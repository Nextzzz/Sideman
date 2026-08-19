namespace Strunika.Core.Dsp;

/// <summary>
/// Streaming-friendly STFT: iterates windowed frames and yields magnitude
/// spectra one at a time, so long files never hold a full spectrogram in memory.
/// </summary>
public sealed class Stft
{
    public int NFft { get; }
    public int Hop { get; }

    private readonly float[] _window;
    private readonly double[] _re;
    private readonly double[] _im;
    private readonly float[] _frame;

    public Stft(int nFft, int hop)
    {
        NFft = nFft;
        Hop = hop;
        _window = Window.Hann(nFft);
        _re = new double[nFft];
        _im = new double[nFft];
        _frame = new float[nFft];
    }

    public int Bins => NFft / 2 + 1;

    public int FrameCount(int sampleCount) =>
        sampleCount < NFft ? 0 : 1 + (sampleCount - NFft) / Hop;

    /// <summary>Center time (seconds) of frame <paramref name="index"/>.</summary>
    public double FrameTime(int index, int sampleRate) =>
        (index * (double)Hop + NFft / 2.0) / sampleRate;

    public IEnumerable<float[]> Magnitudes(float[] samples)
    {
        int frames = FrameCount(samples.Length);
        for (int f = 0; f < frames; f++)
        {
            int start = f * Hop;
            for (int i = 0; i < NFft; i++)
                _frame[i] = samples[start + i] * _window[i];
            var magnitude = new float[Bins];
            Fft.Magnitude(_frame, _re, _im, magnitude);
            yield return magnitude;
        }
    }

    /// <summary>Magnitude of a single already-positioned window (for streaming use).</summary>
    public float[] MagnitudeOf(ReadOnlySpan<float> window)
    {
        for (int i = 0; i < NFft; i++)
            _frame[i] = window[i] * _window[i];
        var magnitude = new float[Bins];
        Fft.Magnitude(_frame, _re, _im, magnitude);
        return magnitude;
    }
}
