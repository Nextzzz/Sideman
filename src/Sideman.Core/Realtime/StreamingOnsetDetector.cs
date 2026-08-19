using Sideman.Core.Dsp;

namespace Sideman.Core.Realtime;

/// <summary>
/// Live attack detection at fine resolution (11.6 ms hop) — the beat
/// follower's ears. Spectral flux with an adaptive median threshold,
/// decisions delayed by one frame so a peak is confirmed as a peak.
/// </summary>
public sealed class StreamingOnsetDetector
{
    private const int NFft = 2048;
    private const int Hop = 512;

    private readonly int _sampleRate;
    private readonly Stft _stft;
    private readonly float[] _ring = new float[NFft * 8];
    private long _written;
    private long _nextFrameAt = NFft;

    private double[]? _previousCompressed;
    private readonly double[] _fluxHistory = new double[256]; // ~3 s
    private int _fluxCount;
    private double _fluxPrev1, _fluxPrev2;
    private long _frameIndex;
    private double _lastOnsetTime = -1;

    /// <summary>Peak must exceed the recent median by this factor.</summary>
    public double Sensitivity { get; set; } = 2.5;

    /// <summary>(time seconds, strength = flux/median) per detected attack.
    /// Strength separates accented downstrokes from grace notes.</summary>
    public event Action<double, double>? OnsetDetected;

    // Rolling novelty for offline-grade tempo estimation on live audio.
    private readonly double[] _noveltyRing = new double[1024]; // ~12 s
    private long _noveltyWritten;

    public double NoveltyFrameRate => _sampleRate / (double)Hop;

    /// <summary>Chronological copy of the recent novelty curve.</summary>
    public double[] NoveltySnapshot()
    {
        long written = _noveltyWritten;
        int count = (int)Math.Min(written, _noveltyRing.Length);
        var result = new double[count];
        for (int i = 0; i < count; i++)
            result[i] = _noveltyRing[(int)((written - count + i) % _noveltyRing.Length)];
        return result;
    }

    public StreamingOnsetDetector(int sampleRate)
    {
        _sampleRate = sampleRate;
        _stft = new Stft(NFft, Hop);
    }

    public void AddSamples(ReadOnlySpan<float> chunk)
    {
        foreach (var s in chunk)
            _ring[(int)(_written++ % _ring.Length)] = s;
        while (_written >= _nextFrameAt)
        {
            ProcessFrame(_nextFrameAt);
            _nextFrameAt += Hop;
        }
    }

    private void ProcessFrame(long endExclusive)
    {
        var window = new float[NFft];
        long start = endExclusive - NFft;
        for (int i = 0; i < NFft; i++)
            window[i] = _ring[(int)((start + i) % _ring.Length)];
        var magnitude = _stft.MagnitudeOf(window);

        double flux = 0;
        var compressed = new double[magnitude.Length];
        for (int k = 0; k < magnitude.Length; k++)
        {
            compressed[k] = Math.Log(1.0 + 100.0 * magnitude[k]);
            if (_previousCompressed != null)
            {
                double rise = compressed[k] - _previousCompressed[k];
                if (rise > 0)
                    flux += rise;
            }
        }
        _previousCompressed = compressed;

        _fluxHistory[(int)(_frameIndex % _fluxHistory.Length)] = flux;
        if (_fluxCount < _fluxHistory.Length)
            _fluxCount++;
        _noveltyRing[(int)(_noveltyWritten % _noveltyRing.Length)] = flux;
        _noveltyWritten++;

        // Peak test on the PREVIOUS frame (needs both neighbors known).
        if (_fluxCount > 40 && _fluxPrev1 > _fluxPrev2 && _fluxPrev1 >= flux)
        {
            double median = Median(_fluxHistory, _fluxCount);
            double time = ((_frameIndex - 1) * (double)Hop + NFft / 2.0) / _sampleRate;
            if (_fluxPrev1 > Sensitivity * median && _fluxPrev1 > 1.0
                && time - _lastOnsetTime > 0.10)
            {
                _lastOnsetTime = time;
                OnsetDetected?.Invoke(time, _fluxPrev1 / Math.Max(median, 1e-6));
            }
        }
        _fluxPrev2 = _fluxPrev1;
        _fluxPrev1 = flux;
        _frameIndex++;
    }

    private static double Median(double[] values, int count)
    {
        var copy = new double[count];
        Array.Copy(values, copy, count);
        Array.Sort(copy);
        return copy[count / 2];
    }
}
