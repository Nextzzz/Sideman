using Strunika.Core.Dsp;

namespace Strunika.Neural;

/// <summary>
/// Decides whether audio is solo-guitar-like or a full mix, to route
/// between the guitar-fine-tuned and generalist chord models.
///
/// The discriminator is a single measured feature: the share of spectral
/// power below 70 Hz. A guitar's lowest string is E2 (82 Hz), so a solo
/// guitar recording has almost nothing down there, while kick drums and
/// bass guitar put plenty. Calibrated on GuitarSet (120 files) vs GTZAN
/// (200 files, 10 genres): guitar 96.7% / mix 90.5% at threshold 0.015.
/// </summary>
public static class AudioDomainClassifier
{
    public const double LowBandHz = 70.0;
    public const double GuitarThreshold = 0.015;

    /// <summary>Share of spectral power below 70 Hz (0..1).</summary>
    public static double LowBandRatio(float[] samples, int sampleRate)
    {
        // Match the calibration resolution (~10.8 Hz bins) at any rate.
        int nFft = sampleRate > 30000 ? 4096 : 2048;
        var stft = new Stft(nFft, nFft / 2);

        int lowBins = (int)(LowBandHz * nFft / sampleRate);
        double low = 0, total = 0;
        foreach (var magnitude in stft.Magnitudes(samples))
        {
            for (int k = 1; k < magnitude.Length; k++)
            {
                double power = (double)magnitude[k] * magnitude[k];
                total += power;
                if (k <= lowBins)
                    low += power;
            }
        }
        return total > 0 ? low / total : 0;
    }

    public static bool IsGuitarLike(float[] samples, int sampleRate) =>
        LowBandRatio(samples, sampleRate) < GuitarThreshold;
}
