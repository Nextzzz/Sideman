using NAudio.Wave;
using Strunika.Core.Analysis;
using Strunika.Core.Diagnostics;
using Strunika.Core.Realtime;
using Strunika.Core.Synthesis;
using Strunika.Media;

namespace Strunika.App.Services;

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
    private readonly StreamingChordDetector _chords;

    // Frozen beat grid (drum-machine mode): estimated patiently while
    // listening, then locked for the whole session.
    private bool _locked;
    private double _period;
    private double _anchor;
    private double _listenStart;
    private double _previousEstimate;
    private double _lastEstimateAt;
    private readonly List<(double Time, double Strength)> _recentOnsets = new();

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
    public double Bpm => _locked ? 60.0 / _period : 0;
    public bool Locked => _locked;
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
        _onsets.OnsetDetected += (time, strength) =>
        {
            if (_locked)
                return;
            lock (_planLock)
            {
                _recentOnsets.Add((time, strength));
                _recentOnsets.RemoveAll(o => o.Time < time - 10.0);
            }
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
        {
            _locked = false;
            _previousEstimate = 0;
            _lastPlannedIndex = long.MinValue;
            _recentOnsets.Clear();
            _listenStart = Interlocked.Read(ref _capturedSamples) / (double)Sr;
        }
        _planner.Start();
        Running = true;
        FileLog.Info("Jam engine started");
    }

    public void Stop()
    {
        if (!Running)
            return;
        _locked = false;
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
        if (!Running)
            return;
        lock (_planLock)
        {
            double captureNow = Interlocked.Read(ref _capturedSamples) / (double)Sr;

            if (!_locked)
            {
                TryLock(captureNow);
                return;
            }

            double horizon = captureNow + 0.9;
            double from = captureNow + 0.05;
            double k = Math.Ceiling((from - _anchor) / _period + 1e-9);
            for (double beat = _anchor + k * _period; beat <= horizon; beat += _period)
            {
                // Dedup by BEAT NUMBER: each beat plays exactly once.
                long index = (long)Math.Round((beat - _anchor) / _period);
                if (index <= _lastPlannedIndex)
                    continue;
                ScheduleBeat(beat, captureNow, index);
                _lastPlannedIndex = index;
            }
        }
    }

    /// <summary>
    /// Patient tempo fixing: at least 8 s of listening and 12 attacks,
    /// then the offline-grade estimator (novelty autocorrelation with an
    /// octave prior — the thing that picks 70, not 140, under eighths)
    /// must produce the SAME answer twice in a row before we commit.
    /// </summary>
    private void TryLock(double captureNow)
    {
        if (captureNow - _listenStart < 8.0 || _recentOnsets.Count < 12)
            return;
        if (captureNow - _lastEstimateAt < 1.0)
            return;
        _lastEstimateAt = captureNow;

        var novelty = _onsets.NoveltySnapshot();
        double bpm = new TempoEstimator { PriorBpm = 95, PriorOctaves = 0.8 }
            .Estimate(novelty, _onsets.NoveltyFrameRate);
        if (bpm < 50 || bpm > 190)
            return;

        if (_previousEstimate > 0
            && Math.Abs(bpm - _previousEstimate) / _previousEstimate < 0.03)
        {
            _period = 60.0 / ((bpm + _previousEstimate) / 2);
            // Phase from the strongest recent attack — an accented downstroke.
            var anchor = _recentOnsets
                .Where(o => o.Time > captureNow - 3.0)
                .OrderByDescending(o => o.Strength)
                .FirstOrDefault();
            _anchor = anchor.Time > 0 ? anchor.Time : captureNow;
            _lastPlannedIndex = long.MinValue;
            _locked = true;
            FileLog.Info($"Jam: tempo fixed at {60.0 / _period:F1} BPM " +
                         $"(two stable estimates), anchor {_anchor:F2}");
        }
        else
        {
            _previousEstimate = bpm;
        }
    }

    private void ScheduleBeat(double beatCaptureTime, double captureNow, long index)
    {
        // Map capture-clock time to the output stream position.
        double lead = beatCaptureTime - captureNow - LatencyOffsetMs / 1000.0;
        long startSample = _scheduler.PositionSamples + (long)(lead * Sr);
        if (startSample < _scheduler.PositionSamples)
            return; // would land in the past — skip rather than stutter

        double period = _period;

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
