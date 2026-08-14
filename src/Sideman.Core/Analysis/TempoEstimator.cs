namespace Sideman.Core.Analysis;

/// <summary>
/// Tempo from autocorrelation of the onset novelty curve. If attacks repeat
/// every T seconds the curve correlates with itself shifted by T. A soft
/// log-Gaussian prior around a comfortable tempo resolves the octave
/// ambiguity (100 vs 50 vs 200 BPM).
/// </summary>
public sealed class TempoEstimator
{
    public double BpmMin { get; init; } = 40.0;
    public double BpmMax { get; init; } = 200.0;
    public double PriorBpm { get; init; } = 120.0;
    public double PriorOctaves { get; init; } = 1.0;

    public double Estimate(double[] novelty, double frameRate)
    {
        int n = novelty.Length;
        if (n < 4)
            return PriorBpm;

        double mean = novelty.Average();
        var centered = novelty.Select(v => v - mean).ToArray();

        int lagMin = Math.Max(1, (int)(60.0 * frameRate / BpmMax));
        int lagMax = Math.Min(n - 1, (int)(60.0 * frameRate / BpmMin));
        if (lagMax <= lagMin)
            return PriorBpm;

        double norm = centered.Sum(v => v * v);
        if (norm <= 0)
            return PriorBpm;

        double bestScore = double.MinValue;
        double bestBpm = PriorBpm;
        for (int lag = lagMin; lag <= lagMax; lag++)
        {
            double ac = 0;
            for (int i = 0; i + lag < n; i++)
                ac += centered[i] * centered[i + lag];
            ac /= norm;

            double bpm = 60.0 * frameRate / lag;
            double octaveDistance = Math.Log2(bpm / PriorBpm) / PriorOctaves;
            double weight = Math.Exp(-0.5 * octaveDistance * octaveDistance);
            double score = ac * weight;
            if (score > bestScore)
            {
                bestScore = score;
                bestBpm = bpm;
            }
        }
        return bestBpm;
    }
}
