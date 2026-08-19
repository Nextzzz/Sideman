namespace Strunika.Core.Analysis;

/// <summary>
/// Beat tracking by dynamic programming (Ellis, 2007). Scores every chain
/// of beat positions: reward for landing on strong novelty, log-squared
/// penalty for gaps deviating from the beat period, and picks the globally
/// best chain — so syncopation cannot derail the grid.
/// </summary>
public sealed class BeatTracker
{
    /// <summary>Rhythm rigidity: high = steady grid, low = follows rubato.</summary>
    public double Tightness { get; init; } = 100.0;

    /// <summary>Returns beat positions as frame indices into the novelty curve.</summary>
    public int[] Track(double[] novelty, double frameRate, double bpm)
    {
        int n = novelty.Length;
        if (n == 0)
            return Array.Empty<int>();

        double period = 60.0 * frameRate / bpm;
        int gapMin = Math.Max(1, (int)Math.Round(period / 2));
        int gapMax = (int)Math.Round(period * 2);

        var penalty = new double[gapMax + 1];
        for (int gap = gapMin; gap <= gapMax; gap++)
        {
            double logRatio = Math.Log(gap / period);
            penalty[gap] = -Tightness * logRatio * logRatio;
        }

        var cumscore = new double[n];
        var backlink = new int[n];
        Array.Fill(backlink, -1);

        for (int i = 0; i < n; i++)
        {
            double best = 0;
            int bestPrev = -1;
            for (int gap = gapMin; gap <= gapMax; gap++)
            {
                int prev = i - gap;
                if (prev < 0)
                    break;
                double value = cumscore[prev] + penalty[gap];
                if (value > best)
                {
                    best = value;
                    bestPrev = prev;
                }
            }
            cumscore[i] = novelty[i] + Math.Max(best, 0);
            backlink[i] = best > 0 ? bestPrev : -1;
        }

        // Backtrack from the strongest chain ending near the end.
        int tail = Math.Max(1, (int)Math.Round(period));
        int end = n - tail;
        for (int i = n - tail; i < n; i++)
            if (cumscore[i] > cumscore[end])
                end = i;

        var chain = new List<int> { end };
        while (backlink[chain[^1]] >= 0)
            chain.Add(backlink[chain[^1]]);
        chain.Reverse();

        // Trim "beats" placed in leading/trailing near-silence.
        int first = 0, last = chain.Count - 1;
        while (first <= last && novelty[chain[first]] < 0.05)
            first++;
        while (last >= first && novelty[chain[last]] < 0.05)
            last--;
        return chain.Skip(first).Take(last - first + 1).ToArray();
    }
}
