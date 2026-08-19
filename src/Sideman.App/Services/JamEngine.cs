using NAudio.Wave;
using Sideman.Core.Analysis;
using Sideman.Core.Diagnostics;
using Sideman.Core.Realtime;
using Sideman.Core.Synthesis;
using Sideman.Media;

namespace Sideman.App.Services;

/// <summary>
/// The band member: listens to the mic (onsets -> tempo PLL, chords via
/// the DSP detector), predicts upcoming beats and schedules drums + bass
/// INTO THE FUTURE on the output clock, compensating device latency.
/// </summary>
public sealed class JamEngine : IDisposable
{
    private const int Sr = 44100;

    private readonly MicrophoneCapture _capture;
    private readonly StreamingOnsetDetector _onsets;
    private readonly TempoPhaseFollower _follower = new();
    private readonly StreamingChordDetector _chords;

    private readonly SampleScheduler _scheduler = new();
    private WaveOutEvent? _output;

    private readonly float[] _kick = DrumKit.Kick();
    private readonly float[] _snare = DrumKit.Snare();
    private readonly float[] _hat = DrumKit.Hat();
    private readonly float[] _click = DrumKit.Click();
    private readonly Dictionary<int, float[]> _bassNotes = new();

    private readonly System.Timers.Timer _planner;
    private long _capturedSamples;
    private long _lastPlannedIndex = long.MinValue;
    private readonly object _planLock = new();

    /// <summary>Extra metronome click on every beat.</summary>
    public bool MetronomeOn { get; set; }

    public bool Running { get; private set; }
    public double Bpm => _follower.Bpm;
    public bool Locked => _follower.Locked;
    public string CurrentChordLabel =>
        _chords.CurrentChord.Label == "N" ? "—" : _chords.CurrentChord.Label;

    /// <summary>Output-path latency compensation, tunable by ear.</summary>
    public double LatencyOffsetMs { get; set; } = 120;

    public float DrumGain { get; set; } = 0.8f;
    public float BassGain { get; set; } = 0.7f;

    public JamEngine(MicrophoneCapture capture)
    {
        _capture = capture;
        _onsets = new StreamingOnsetDetector(Sr);
        _chords = new StreamingChordDetector(Sr);
        // Fixed-tempo mode: we listen only until the BPM is understood,
        // then FREEZE the grid — the band plays like a drum machine and
        // the player follows it. No corrections, no feedback runaway.
        _onsets.OnsetDetected += t =>
        {
            if (!_follower.Locked)
                _follower.OnOnset(t);
        };
        _follower.LockAcquired += () =>
        {
            lock (_planLock)
                _lastPlannedIndex = long.MinValue;
            FileLog.Info($"Jam: tempo fixed at {_follower.Bpm:F0} BPM");
        };

        _planner = new System.Timers.Timer(80);
        _planner.Elapsed += (_, _) => PlanAhead();
    }

    private void OnChunk(float[] chunk)
    {
        Interlocked.Add(ref _capturedSamples, chunk.Length);
        _onsets.AddSamples(chunk);
        _chords.AddSamples(chunk);
    }

    public void Start()
    {
        if (Running)
            return;
        _capture.ChunkAvailable += OnChunk;
        _output = new WaveOutEvent { DesiredLatency = 90, NumberOfBuffers = 3 };
        _output.Init(_scheduler);
        _output.Play();
        lock (_planLock)
            _lastPlannedIndex = long.MinValue;
        _planner.Start();
        Running = true;
        FileLog.Info("Jam engine started");
    }

    public void Stop()
    {
        if (!Running)
            return;
        _planner.Stop();
        _capture.ChunkAvailable -= OnChunk;
        _scheduler.Clear();
        _output?.Dispose();
        _output = null;
        Running = false;
        FileLog.Info("Jam engine stopped");
    }

    private void PlanAhead()
    {
        if (!Running || !_follower.Locked)
            return;
        lock (_planLock)
        {
            double captureNow = Interlocked.Read(ref _capturedSamples) / (double)Sr;
            double horizon = captureNow + 0.9;

            foreach (var beat in _follower.BeatsBetween(captureNow + 0.05, horizon))
            {
                // Dedup by BEAT NUMBER, not by time: PLL corrections move
                // beat times slightly every onset, and a time-based cursor
                // re-schedules the same beat over and over (the "diesel
                // generator" bug).
                long index = _follower.BeatIndex(beat);
                if (index <= _lastPlannedIndex)
                    continue;
                ScheduleBeat(beat, captureNow);
                _lastPlannedIndex = index;
            }
        }
    }

    private void ScheduleBeat(double beatCaptureTime, double captureNow)
    {
        // Map capture-clock time to the output stream position.
        double lead = beatCaptureTime - captureNow - LatencyOffsetMs / 1000.0;
        long startSample = _scheduler.PositionSamples + (long)(lead * Sr);
        if (startSample < _scheduler.PositionSamples)
            return; // would land in the past — skip rather than stutter

        long index = _follower.BeatIndex(beatCaptureTime);
        double period = 60.0 / Math.Max(_follower.Bpm, 1);

        // Drums: kick/snare alternate, hats on beat and off-beat.
        _scheduler.Schedule(startSample, index % 2 == 0 ? _kick : _snare, DrumGain);
        if (MetronomeOn)
            _scheduler.Schedule(startSample, _click, 0.9f);
        _scheduler.Schedule(startSample, _hat, DrumGain * 0.6f);
        _scheduler.Schedule(startSample + (long)(period / 2 * Sr), _hat, DrumGain * 0.45f);

        // Bass: root of the current chord on every beat.
        var chord = _chords.CurrentChord;
        if (chord.Quality != ChordQuality.None)
        {
            var note = BassNote(chord.Root);
            _scheduler.Schedule(startSample, note, BassGain);
        }
    }

    private float[] BassNote(int pitchClass)
    {
        if (!_bassNotes.TryGetValue(pitchClass, out var note))
        {
            // Bass octave E1..D#2 (MIDI 28..39).
            int midi = 28 + ((pitchClass - 4 + 12) % 12);
            double freq = Notes.FrequencyFromMidi(midi);
            _bassNotes[pitchClass] = note =
                PluckSynth.Pluck(freq, 0.5, Sr, decay: 0.9985, seed: pitchClass + 10);
        }
        return note;
    }

    public void Dispose()
    {
        Stop();
        _planner.Dispose();
    }
}
