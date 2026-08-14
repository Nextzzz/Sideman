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

    public Chord CurrentChord { get; private set; } = Chord.None;
    public double Confidence { get; private set; }

    public event Action<Chord>? ChordChanged;

    public StreamingChordDetector(
        int sampleRate,
        ChordEmissionModel? emissions = null,
        double selfTransition = 0.9,
        int holdFrames = 3)
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

        _emissions.FillEmissions(frame, _frameEmissions);

        // Forward Viterbi step.
        int states = _emissions.StateCount;
        int bestPrev = 0;
        for (int s = 1; s < states; s++)
            if (_scores[s] > _scores[bestPrev])
                bestPrev = s;

        for (int s = 0; s < states; s++)
        {
            double stay = _scores[s] + _logStay;
            double jump = _scores[bestPrev] + _logSwitch;
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
