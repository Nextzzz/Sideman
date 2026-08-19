namespace Strunika.Neural;

/// <summary>
/// Streaming 2:1 decimator (44100 -> 22050) with a 31-tap windowed-sinc
/// low-pass, so nothing above the new Nyquist aliases into the analysis.
/// </summary>
public sealed class HalfbandDecimator
{
    private const int Taps = 31;
    private static readonly float[] Kernel = BuildKernel();

    private readonly float[] _delay = new float[Taps];
    private int _parity;

    private static float[] BuildKernel()
    {
        var kernel = new float[Taps];
        int center = Taps / 2;
        double sum = 0;
        for (int n = 0; n < Taps; n++)
        {
            int m = n - center;
            // sinc low-pass at a quarter of the input rate, Hann-windowed.
            double sinc = m == 0 ? 0.5 : Math.Sin(Math.PI * m / 2.0) / (Math.PI * m);
            double window = 0.5 * (1 + Math.Cos(Math.PI * m / center));
            kernel[n] = (float)(sinc * window);
            sum += kernel[n];
        }
        for (int n = 0; n < Taps; n++)
            kernel[n] /= (float)(sum * 0.5); // unity gain after keeping every 2nd sample
        return kernel;
    }

    /// <summary>Feed input samples; returns the decimated samples.</summary>
    public float[] Process(ReadOnlySpan<float> input)
    {
        var output = new List<float>(input.Length / 2 + 1);
        foreach (var sample in input)
        {
            Array.Copy(_delay, 1, _delay, 0, Taps - 1);
            _delay[Taps - 1] = sample;
            _parity ^= 1;
            if (_parity == 0)
                continue;

            float acc = 0;
            for (int n = 0; n < Taps; n++)
                acc += _delay[n] * Kernel[n];
            output.Add(acc);
        }
        return output.ToArray();
    }

    /// <summary>One-shot helper for in-memory recordings.</summary>
    public static float[] Decimate(float[] samples44100)
    {
        return new HalfbandDecimator().Process(samples44100);
    }
}
