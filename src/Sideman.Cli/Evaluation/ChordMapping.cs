using Sideman.Core.Analysis;

namespace Sideman.Cli.Evaluation;

/// <summary>
/// Maps dataset chord labels ("D#:maj7", "Bb:min/5", "N") onto our
/// major/minor vocabulary — the standard MIREX "majmin" evaluation:
/// maj/maj7/7/6/9 → major family, min/min7/min6 → minor family,
/// dim/aug/sus → excluded from scoring.
/// </summary>
public static class ChordMapping
{
    private static readonly Dictionary<char, int> LetterToPc = new()
    {
        ['C'] = 0, ['D'] = 2, ['E'] = 4, ['F'] = 5, ['G'] = 7, ['A'] = 9, ['B'] = 11,
    };

    /// <summary>Returns our label ("D#", "D#m", "N") or null when the truth
    /// chord has no major/minor equivalent (dim/aug/sus).</summary>
    public static string? ToMajMin(string raw)
    {
        if (raw is "N" or "X")
            return "N";

        int colon = raw.IndexOf(':');
        string rootPart = colon < 0 ? raw : raw[..colon];
        string quality = colon < 0 ? "maj" : raw[(colon + 1)..];

        // Strip bass ("/5") and extensions in parentheses ("maj6(*5)").
        int slash = quality.IndexOf('/');
        if (slash >= 0)
            quality = quality[..slash];
        int paren = quality.IndexOf('(');
        if (paren >= 0)
            quality = quality[..paren];

        if (!LetterToPc.TryGetValue(rootPart[0], out int pc))
            return null;
        foreach (char c in rootPart[1..])
        {
            if (c == '#') pc++;
            else if (c == 'b') pc--;
        }
        pc = ((pc % 12) + 12) % 12;

        if (quality.StartsWith("min"))
            return Notes.Names[pc] + "m";
        if (quality.Length == 0
            || quality.StartsWith("maj")
            || quality is "7" or "9" or "11" or "13" or "6")
            return Notes.Names[pc];

        return null; // dim / hdim / aug / sus — outside the majmin vocabulary
    }
}
