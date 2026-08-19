namespace Sideman.Neural;

/// <summary>
/// Key detection from a decoded chord path + diatonic bonuses for the
/// second Viterbi pass. Family mapping mirrors the majmin evaluation:
/// maj/maj6/maj7/7 -> major family, min/min6/min7/minmaj7 -> minor family.
/// </summary>
public static class KeyPrior
{
    public readonly record struct Key(int Tonic, bool IsMinor)
    {
        public string Name => Sideman.Core.Analysis.Notes.Names[Tonic] + (IsMinor ? "m" : "");
    }

    private static readonly int[] MajFamilyQualities = { 1, 5, 8, 9 };
    private static readonly int[] MinFamilyQualities = { 0, 4, 6, 7 };

    // Diatonic (root offset, isMinorChord) sets. Minor includes the
    // harmonic-minor major dominant — E in Am is family, not a stranger.
    private static readonly (int, bool)[] MajorKeyChords =
        { (0, false), (2, true), (4, true), (5, false), (7, false), (9, true) };
    private static readonly (int, bool)[] MinorKeyChords =
        { (0, true), (3, false), (5, true), (7, true), (7, false), (8, false), (10, false) };

    /// <summary>(root, isMinor) for a state, or null for N/X/dim/aug/sus.</summary>
    private static (int Root, bool IsMinor)? Family(int state)
    {
        if (state >= 168)
            return null;
        int quality = state % 14;
        if (MajFamilyQualities.Contains(quality))
            return (state / 14, false);
        if (MinFamilyQualities.Contains(quality))
            return (state / 14, true);
        return null;
    }

    /// <summary>Pick the key whose diatonic set covers the most decoded
    /// frames; null when no key explains at least 60% of chord time.
    /// Relative keys (G major vs E minor) cover identical chord sets, so
    /// ties are broken by the TONIC: how long it sounds and whether the
    /// piece opens on it.</summary>
    public static Key? Estimate(int[] path, string[] labels)
    {
        var counts = new Dictionary<(int, bool), int>();
        int total = 0;
        (int, bool)? firstFamily = null;
        foreach (var state in path)
        {
            var family = Family(state);
            if (family == null)
                continue;
            firstFamily ??= family;
            total++;
            counts[family.Value] = counts.GetValueOrDefault(family.Value) + 1;
        }
        if (total < 50) // under ~5 seconds of chords: not enough evidence
            return null;

        Key best = default;
        double bestScore = double.MinValue;
        int bestCovered = -1;
        for (int tonic = 0; tonic < 12; tonic++)
        {
            foreach (bool minor in new[] { false, true })
            {
                var set = minor ? MinorKeyChords : MajorKeyChords;
                int covered = set.Sum(c =>
                    counts.GetValueOrDefault(((tonic + c.Item1) % 12, c.Item2)));
                int tonicFrames = counts.GetValueOrDefault((tonic, minor));
                double score = covered
                               + 0.30 * tonicFrames
                               + (firstFamily == (tonic, minor) ? 0.15 * total : 0);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestCovered = covered;
                    best = new Key(tonic, minor);
                }
            }
        }
        return bestCovered >= 0.6 * total ? best : null;
    }

    /// <summary>Per-state log-space bonuses for the given key.</summary>
    public static float[] StateBonuses(Key key, string[] labels, double strength)
    {
        var diatonic = new HashSet<(int, bool)>(
            (key.IsMinor ? MinorKeyChords : MajorKeyChords)
            .Select(c => (((key.Tonic + c.Item1) % 12), c.Item2)));

        var bonus = new float[labels.Length];
        for (int state = 0; state < labels.Length; state++)
        {
            var family = Family(state);
            if (family != null && diatonic.Contains(family.Value))
                bonus[state] = (float)strength;
        }
        return bonus;
    }
}
