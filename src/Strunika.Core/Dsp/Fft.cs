namespace Strunika.Core.Dsp;

/// <summary>
/// In-place iterative radix-2 FFT. Dependency-free on purpose: the same
/// code must eventually run on desktop and mobile.
/// </summary>
public static class Fft
{
    /// <summary>Forward FFT. Lengths must be equal powers of two.</summary>
    public static void Forward(double[] re, double[] im)
    {
        int n = re.Length;
        if (n != im.Length || (n & (n - 1)) != 0)
            throw new ArgumentException("FFT length must be a power of two.");

        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Butterflies.
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = -2.0 * Math.PI / len;
            double wRe = Math.Cos(angle);
            double wIm = Math.Sin(angle);
            for (int i = 0; i < n; i += len)
            {
                double curRe = 1.0, curIm = 0.0;
                int half = len >> 1;
                for (int k = 0; k < half; k++)
                {
                    int a = i + k;
                    int b = a + half;
                    double tRe = re[b] * curRe - im[b] * curIm;
                    double tIm = re[b] * curIm + im[b] * curRe;
                    re[b] = re[a] - tRe;
                    im[b] = im[a] - tIm;
                    re[a] += tRe;
                    im[a] += tIm;
                    double nextRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = nextRe;
                }
            }
        }
    }

    /// <summary>
    /// Magnitude spectrum of a real windowed frame: bins 0..n/2 inclusive.
    /// </summary>
    public static void Magnitude(ReadOnlySpan<float> frame, double[] scratchRe, double[] scratchIm, float[] magnitudeOut)
    {
        int n = scratchRe.Length;
        for (int i = 0; i < n; i++)
        {
            scratchRe[i] = i < frame.Length ? frame[i] : 0.0;
            scratchIm[i] = 0.0;
        }
        Forward(scratchRe, scratchIm);
        int bins = n / 2 + 1;
        for (int k = 0; k < bins; k++)
            magnitudeOut[k] = (float)Math.Sqrt(scratchRe[k] * scratchRe[k] + scratchIm[k] * scratchIm[k]);
    }
}
