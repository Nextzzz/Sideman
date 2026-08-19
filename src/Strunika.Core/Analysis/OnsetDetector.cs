using Strunika.Core.Dsp;

namespace Strunika.Core.Analysis;

/// <summary>
/// Attack detection via spectral flux: how much NEW energy appears in the
/// spectrum between consecutive frames. Positive changes only — a decaying
/// (ringing) string is not an event, a fresh attack is.
/// </summary>
public sealed class OnsetDetector
{
    public int NFft { get; init; } = 2048;
    public int Hop { get; init; } = 512;
    public double Gamma { get; init; } = 100.0;
    public double Delta { get; init; } = 0.05;
    public double MedianWindowSeconds { get; init; } = 0.4;
    public double MinGapSeconds { get; init; } = 0.05;

    public double FrameRate(int sampleRate) => sampleRate / (double)Hop;

    /// <summary>Normalized novelty curve (one value per STFT frame).</summary>
    public double[] NoveltyCurve(float[] samples, int sampleRate)
    {
        var stft = new Stft(NFft, Hop);
        int frames = stft.FrameCount(samples.Length);
        if (frames == 0)
            return Array.Empty<double>();

        var novelty = new double[frames];
        double[]? previous = null;
        int f = 0;
        foreach (var magnitude in stft.Magnitudes(samples))
        {
            var compressed = new double[magnitude.Length];
            for (int k = 0; k < magnitude.Length; k++)
                compressed[k] = Math.Log(1.0 + Gamma * magnitude[k]);

            if (previous != null)
            {
                double flux = 0;
                for (int k = 0; k < compressed.Length; k++)
                {
                    double d = compressed[k] - previous[k];
                    if (d > 0)
                        flux += d; // half-wave rectification
                }
                novelty[f] = flux;
            }
            previous = compressed;
            f++;
        }

        double max = novelty.Max();
        if (max > 0)
            for (int i = 0; i < novelty.Length; i++)
                novelty[i] /= max;
        return novelty;
    }

    /// <summary>Onset times in seconds (window-center convention).</summary>
    public double[] Detect(float[] samples, int sampleRate)
    {
        var novelty = NoveltyCurve(samples, sampleRate);
        var peaks = PickPeaks(novelty, FrameRate(sampleRate));
        return peaks.Select(p => (p * (double)Hop + NFft / 2.0) / sampleRate).ToArray();
    }

    public int[] PickPeaks(double[] novelty, double frameRate)
    {
        if (novelty.Length == 0)
            return Array.Empty<int>();

        // Adaptive threshold: local median + delta, so quiet passages
        // are judged against their own level, not the loudest strum's.
        int window = Math.Max(3, ((int)(MedianWindowSeconds * frameRate)) | 1);
        var threshold = MedianFilter(novelty, window);
        for (int i = 0; i < threshold.Length; i++)
            threshold[i] += Delta;

        int minGap = Math.Max(1, (int)(MinGapSeconds * frameRate));
        var peaks = new List<int>();
        for (int i = 1; i < novelty.Length - 1; i++)
        {
            if (novelty[i] <= threshold[i] || novelty[i] < novelty[i - 1] || novelty[i] < novelty[i + 1])
                continue;
            if (peaks.Count > 0 && i - peaks[^1] < minGap)
            {
                if (novelty[i] > novelty[peaks[^1]])
                    peaks[^1] = i; // keep the stronger of two close peaks
                continue;
            }
            peaks.Add(i);
        }
        return peaks.ToArray();
    }

    private static double[] MedianFilter(double[] x, int window)
    {
        int half = window / 2;
        var result = new double[x.Length];
        var buffer = new double[window];
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = 0; j < window; j++)
            {
                int idx = Math.Clamp(i - half + j, 0, x.Length - 1);
                buffer[j] = x[idx];
            }
            Array.Sort(buffer);
            result[i] = buffer[half];
        }
        return result;
    }
}
