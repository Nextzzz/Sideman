namespace Sideman.Core.Analysis;

public readonly record struct PitchResult(double Frequency, double Clarity);

/// <summary>
/// Monophonic pitch detection — the tuner. YIN algorithm
/// (de Cheveigné &amp; Kawahara, 2002): autocorrelation-family difference
/// function with cumulative mean normalization, which avoids the classic
/// octave errors of plain autocorrelation.
/// </summary>
public sealed class PitchDetector
{
    public double MinFrequency { get; init; } = 60.0;    // below guitar drop-C
    public double MaxFrequency { get; init; } = 1200.0;
    public double Threshold { get; init; } = 0.15;

    /// <summary>Returns null when the window does not contain a clear pitch.</summary>
    public PitchResult? Detect(ReadOnlySpan<float> samples, int sampleRate)
    {
        int n = samples.Length;
        int tauMax = Math.Min((int)(sampleRate / MinFrequency), n / 2);
        int tauMin = Math.Max(2, (int)(sampleRate / MaxFrequency));
        if (tauMax <= tauMin)
            return null;

        // Difference function over a fixed comparison window.
        int window = n - tauMax;
        var diff = new double[tauMax + 1];
        for (int tau = 1; tau <= tauMax; tau++)
        {
            double sum = 0;
            for (int i = 0; i < window; i++)
            {
                double d = samples[i] - samples[i + tau];
                sum += d * d;
            }
            diff[tau] = sum;
        }

        // Cumulative mean normalized difference: 1.0 = noise, dips = pitch.
        var cmnd = new double[tauMax + 1];
        cmnd[0] = 1.0;
        double running = 0;
        for (int tau = 1; tau <= tauMax; tau++)
        {
            running += diff[tau];
            cmnd[tau] = running > 0 ? diff[tau] * tau / running : 1.0;
        }

        // First dip below threshold; walk to its local minimum.
        int best = -1;
        for (int tau = tauMin; tau <= tauMax; tau++)
        {
            if (cmnd[tau] < Threshold)
            {
                while (tau + 1 <= tauMax && cmnd[tau + 1] < cmnd[tau])
                    tau++;
                best = tau;
                break;
            }
        }
        if (best < 0)
        {
            // No confident dip — take the global minimum if it is decent.
            double min = double.MaxValue;
            for (int tau = tauMin; tau <= tauMax; tau++)
            {
                if (cmnd[tau] < min)
                {
                    min = cmnd[tau];
                    best = tau;
                }
            }
            if (min > 0.5)
                return null; // unvoiced / silence
        }

        // Parabolic interpolation refines tau below one sample.
        double refined = best;
        if (best > 1 && best < tauMax)
        {
            double a = cmnd[best - 1], b = cmnd[best], c = cmnd[best + 1];
            double denom = a + c - 2 * b;
            if (Math.Abs(denom) > 1e-12)
                refined = best + 0.5 * (a - c) / denom;
        }

        return new PitchResult(sampleRate / refined, 1.0 - cmnd[best]);
    }
}
