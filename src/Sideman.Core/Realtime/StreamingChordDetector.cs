using Sideman.Core.Analysis;
using Sideman.Core.Dsp;

namespace Sideman.Core.Realtime;

/// <summary>
/// Live chord detection: feed microphone chunks, read the current chord.
/// Internally a Viterbi-style forward filter over the same emission model
/// as offline recognition, but causal: it only knows the past.
/// A hold-off keeps the display from flickering on transients.
/// </summary>
public sealed class StreamingChordDetector
{
    private readonly ChromaExtractor _chroma;
    private readonly ChordEmissionModel _emissions;
    private readonly Stft _stft;
    private readonly int _sampleRate;

    private readonly float[] _ring;
    private long _written;
    private long _nextFrameAt;

    private readonly double[] _scores;
    private readonly double[] _frameEmissions;
    private readonly float[] _window;
    private readonly double _logStay;
    private readonly double _logSwitch;

    private int _candidate = -1;
    private int _candidateFrames;
    private readonly int _holdFrames;
    private readonly NoiseGate _gate = new();

    // Attack handling: strum transients (pick noise, grazed strings that
    // get damped a moment later) carry misleading chroma. We detect them
    // via spectral flux and simply do not listen until they settle.
    private double _fluxAverage;
    private int _processedFrames;
    private int _transientHold;
    private bool _justAfterTransient;

    /// <summary>Frames to ignore after a detected strum attack (~46 ms each).</summary>
    public int TransientFrames { get; set; } = 3;

    /// <summary>A frame is an attack when its flux exceeds the running
    /// average by this factor.</summary>
    public double OnsetFactor { get; set; } = 3.0;

    public Chord CurrentChord { get; private set; } = Chord.None;
    public double Confidence { get; private set; }

    /// <summary>Live noise-gate sensitivity (dB above the learned room
    /// floor). Exposed so the app can put it on a slider.</summary>
    public double GateMarginDb
    {
        get => _gate.MarginDb;
        set => _gate.MarginDb = value;
    }

    public event Action<Chord>? ChordChanged;

    // Defaults tuned for display stability: the first detection during a
    // strum transient is usually wrong, so a candidate must hold ~230 ms
    // before it is shown. Latency you can feel beats flicker you can see.
    public StreamingChordDetector(
        int sampleRate,
        ChordEmissionModel? emissions = null,
        double selfTransition = 0.95,
        int holdFrames = 5)
    {
        _sampleRate = sampleRate;
        _emissions = emissions ?? new ChordEmissionModel();
        _chroma = new ChromaExtractor();
        _stft = new Stft(_chroma.NFft, _chroma.Hop);
        _ring = new float[_chroma.NFft * 4];
        _window = new float[_chroma.NFft];
        _nextFrameAt = _chroma.NFft;

        int states = _emissions.StateCount;
        _scores = new double[states];
        _frameEmissions = new double[states];
        _logStay = Math.Log(selfTransition);
        _logSwitch = Math.Log((1 - selfTransition) / (states - 1));
        _holdFrames = holdFrames;
    }

    public void AddSamples(ReadOnlySpan<float> chunk)
    {
        foreach (var s in chunk)
            _ring[(int)(_written++ % _ring.Length)] = s;

        while (_written >= _nextFrameAt)
        {
            ProcessFrame(_nextFrameAt);
            _nextFrameAt += _chroma.Hop;
        }
    }

    private void ProcessFrame(long endExclusive)
    {
        // Copy the latest NFft samples out of the ring, oldest first.
        int n = _chroma.NFft;
        long start = endExclusive - n;
        for (int i = 0; i < n; i++)
            _window[i] = _ring[(int)((start + i) % _ring.Length)];

        var magnitude = _stft.MagnitudeOf(_window);
        var frame = _chroma.FoldFrame(magnitude, _sampleRate);

        // Attack detection: high flux vs the running average = a strum is
        // happening right now, and this frame's chroma is transient garbage.
        bool isAttack = _processedFrames > 5
            && frame.Flux > OnsetFactor * _fluxAverage
            && frame.Flux > 1.0;
        _fluxAverage = _processedFrames == 0
            ? frame.Flux
            : 0.95 * _fluxAverage + 0.05 * frame.Flux;
        _processedFrames++;

        if (isAttack)
            _transientHold = TransientFrames;
        if (_transientHold > 0)
        {
            _transientHold--;
            _justAfterTransient = _transientHold == 0;
            return; // don't listen during the attack
        }

        var verdict = _gate.Assess(frame.Rms, frame.Flatness);
        if (verdict == GateVerdict.Quiet)
            return; // a decaying chord: freeze the display, no new evidence
        _emissions.FillEmissions(frame, verdict, _frameEmissions);

        // Forward Viterbi step. Right after a transient the chord may well
        // have just changed — relax the transition for one frame so clean
        // evidence can switch immediately instead of fighting inertia.
        int states = _emissions.StateCount;
        double logStay = _logStay, logSwitch = _logSwitch;
        if (_justAfterTransient)
        {
            logStay = Math.Log(0.5);
            logSwitch = Math.Log(0.5 / (states - 1));
            _justAfterTransient = false;
        }

        int bestPrev = 0;
        for (int s = 1; s < states; s++)
            if (_scores[s] > _scores[bestPrev])
                bestPrev = s;

        for (int s = 0; s < states; s++)
        {
            double stay = _scores[s] + logStay;
            double jump = _scores[bestPrev] + logSwitch;
            _frameEmissions[s] += Math.Max(stay, jump);
        }

        // Renormalize so scores never drift to -infinity.
        double max = _frameEmissions.Max();
        int top = 0;
        double second = double.MinValue;
        for (int s = 0; s < states; s++)
        {
            _scores[s] = _frameEmissions[s] - max;
            if (_scores[s] == 0)
                top = s;
        }
        for (int s = 0; s < states; s++)
            if (s != top && _scores[s] > second)
                second = _scores[s];
        Confidence = second == double.MinValue ? 1 : 1 - Math.Exp(second);

        // Hold-off: a new chord must win several consecutive frames.
        if (top == _candidate)
        {
            _candidateFrames++;
        }
        else
        {
            _candidate = top;
            _candidateFrames = 1;
        }

        var candidateChord = _emissions.ChordOf(_candidate);
        if (_candidateFrames >= _holdFrames && candidateChord != CurrentChord)
        {
            CurrentChord = candidateChord;
            ChordChanged?.Invoke(CurrentChord);
        }
    }
}
