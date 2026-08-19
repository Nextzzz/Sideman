namespace Strunika.Core.Realtime;

/// <summary>
/// The band member's inner clock: a phase-locked loop over onsets.
/// It listens until the pulse is clear, then predicts future beats and
/// gently corrects period and phase from each on-grid attack — off-grid
/// attacks (eighths, syncopation) are ignored rather than obeyed.
/// </summary>
public sealed class TempoPhaseFollower
{
    private readonly List<double> _onsets = new();
    private double _anchor;   // a grid beat time, kept NEAR the present
    private long _anchorIndex; // global beat number of the anchor
    private double _period;
    private double _lastAcceptedOnset;

    /// <summary>Phase correction gain (fraction of the error applied).</summary>
    public double PhaseGain { get; init; } = 0.35;

    /// <summary>Period correction gain. 0.10 keeps up with a player's
    /// gradual accelerando while adding ~1 ms noise on human jitter.</summary>
    public double PeriodGain { get; init; } = 0.10;

    public double MinPeriod { get; init; } = 60.0 / 180; // 180 BPM
    public double MaxPeriod { get; init; } = 60.0 / 55;  // 55 BPM

    public bool Locked { get; private set; }

    public double Bpm => Locked ? 60.0 / _period : 0;

    public event Action? LockAcquired;
    public event Action? LockLost;

    public void OnOnset(double time)
    {
        _onsets.Add(time);
        _onsets.RemoveAll(t => t < time - 9.0);

        if (!Locked)
        {
            TryLock(time);
            return;
        }

        // Re-anchor to the grid beat nearest this onset FIRST: corrections
        // must rotate the grid around the present, not around the ancient
        // lock point — otherwise a period tweak swings the phase at "now"
        // like a long lever and the loop oscillates.
        double beats = (time - _anchor) / _period;
        long nearest = (long)Math.Round(beats);
        _anchor += nearest * _period;
        _anchorIndex += nearest;

        double error = time - _anchor;
        if (Math.Abs(error) <= 0.22 * _period)
        {
            _anchor += PhaseGain * error;
            _period = Math.Clamp(
                _period + PeriodGain * error, MinPeriod, MaxPeriod);
            _lastAcceptedOnset = time;
        }
        else if (time - _lastAcceptedOnset > 6 * _period)
        {
            // The player stopped or drifted beyond recovery: re-listen.
            Locked = false;
            LockLost?.Invoke();
        }
    }

    private void TryLock(double now)
    {
        if (_onsets.Count < 8)
            return;

        // Histogram of inter-onset intervals (including non-adjacent pairs
        // up to 1.2 s) — the heaviest bucket in the plausible range is the
        // beat period.
        var buckets = new Dictionary<int, (int Count, double Sum)>();
        for (int i = 0; i < _onsets.Count; i++)
        {
            for (int j = i + 1; j < _onsets.Count; j++)
            {
                double interval = _onsets[j] - _onsets[i];
                if (interval > 1.25)
                    break;
                if (interval < MinPeriod * 0.9)
                    continue;
                int bucket = (int)(interval / 0.025);
                var entry = buckets.GetValueOrDefault(bucket);
                buckets[bucket] = (entry.Count + 1, entry.Sum + interval);
            }
        }
        if (buckets.Count == 0)
            return;

        // Merge each bucket with its neighbors: a metronomic player's
        // identical intervals straddle a bucket edge in floating point
        // and would otherwise never reach the threshold.
        int bestBucket = 0;
        int bestCount = -1;
        double bestSum = 0;
        foreach (var bucket in buckets.Keys)
        {
            int count = 0;
            double sum = 0;
            for (int d = -1; d <= 1; d++)
            {
                if (buckets.TryGetValue(bucket + d, out var e))
                {
                    count += e.Count;
                    sum += e.Sum;
                }
            }
            if (count > bestCount)
            {
                bestCount = count;
                bestSum = sum;
                bestBucket = bucket;
            }
        }
        if (bestCount < 6)
            return;
        double period = bestSum / bestCount;
        if (period < MinPeriod || period > MaxPeriod)
            return;

        _period = period;
        _anchor = _onsets[^1];
        _anchorIndex = 0;
        _lastAcceptedOnset = now;
        Locked = true;
        LockAcquired?.Invoke();
    }

    /// <summary>Predicted beat times in (from, to] on the capture clock.</summary>
    public IEnumerable<double> BeatsBetween(double from, double to)
    {
        if (!Locked)
            yield break;
        double k = Math.Ceiling((from - _anchor) / _period + 1e-9);
        for (double t = _anchor + k * _period; t <= to; t += _period)
            yield return t;
    }

    /// <summary>Global index of the beat at the given grid time (for
    /// pattern positions: kick/snare alternation).</summary>
    public long BeatIndex(double beatTime) =>
        _anchorIndex + (long)Math.Round((beatTime - _anchor) / _period);
}
