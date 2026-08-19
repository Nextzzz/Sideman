namespace Sideman.Core.Analysis;

/// <summary>Post-processing of chord timelines for display correctness.</summary>
public static class ChordTimeline
{
    /// <summary>
    /// Snap segment boundaries to the beat grid: chords change ON beats,
    /// and a boundary a fraction off is an analysis artifact, not music.
    /// Boundaries farther than the tolerance stay untouched; segments that
    /// collapse are dropped and equal neighbors merged.
    /// </summary>
    public static List<(double Start, double End, string Label)> SnapToBeats(
        IReadOnlyList<(double Start, double End, string Label)> segments,
        IReadOnlyList<double> beatTimes)
    {
        if (segments.Count == 0 || beatTimes.Count < 2)
            return segments.ToList();

        var periods = new List<double>();
        for (int i = 1; i < beatTimes.Count; i++)
            periods.Add(beatTimes[i] - beatTimes[i - 1]);
        periods.Sort();
        double tolerance = Math.Min(0.45, 0.40 * periods[periods.Count / 2]);

        // Move each INTERNAL boundary to the nearest beat within tolerance.
        var boundaries = new double[segments.Count + 1];
        boundaries[0] = segments[0].Start;
        boundaries[^1] = segments[^1].End;
        for (int i = 1; i < segments.Count; i++)
        {
            double b = segments[i].Start;
            double nearest = NearestBeat(beatTimes, b);
            boundaries[i] = Math.Abs(nearest - b) <= tolerance ? nearest : b;
        }

        var result = new List<(double, double, string)>();
        for (int i = 0; i < segments.Count; i++)
        {
            double start = boundaries[i];
            double end = boundaries[i + 1];
            if (end - start < 1e-3)
                continue; // collapsed by snapping
            if (result.Count > 0 && result[^1].Item3 == segments[i].Label)
                result[^1] = (result[^1].Item1, end, segments[i].Label);
            else
                result.Add((start, end, segments[i].Label));
        }
        return result;
    }

    private static double NearestBeat(IReadOnlyList<double> beats, double time)
    {
        int lo = 0, hi = beats.Count - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) / 2;
            if (beats[mid] < time)
                lo = mid;
            else
                hi = mid;
        }
        return Math.Abs(beats[lo] - time) <= Math.Abs(beats[hi] - time)
            ? beats[lo]
            : beats[hi];
    }
}
