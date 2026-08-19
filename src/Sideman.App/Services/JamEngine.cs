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
    private readonly Dictionary<int, float[]> _bassNotes = new();

    private readonly System.Timers.Timer _planner;
    private long _capturedSamples;
    private double _lastPlannedBeat;
    private readonly object _planLock = new();

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
        _onsets.OnsetDetected += _follower.OnOnset;
        _follower.LockLost += () => FileLog.Info("Jam: tempo lock lost");
        _follower.LockAcquired += () =>
            FileLog.Info($"Jam: tempo locked at {_follower.Bpm:F0} BPM");

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
        _lastPlannedBeat = 0;
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
            double from = Math.Max(_lastPlannedBeat, captureNow + 0.05);

            foreach (var beat in _follower.BeatsBetween(from, horizon))
            {
                ScheduleBeat(beat, captureNow);
                _lastPlannedBeat = beat;
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
