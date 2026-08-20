namespace Strunika.Neural;

/// <summary>Turns model labels ("C:maj", "F#:min7") into display form ("C", "F#m7").</summary>
public static class ChordLabels
{
    public static string Pretty(string label)
    {
        if (label is "N" or "X")
            return "—";
        int colon = label.IndexOf(':');
        if (colon < 0)
            return label;

        string root = label[..colon];
        string quality = label[(colon + 1)..];
        return root + quality switch
        {
            "maj" => "",
            "min" => "m",
            "maj7" => "maj7",
            "min7" => "m7",
            "min6" => "m6",
            "maj6" => "6",
            "minmaj7" => "mMaj7",
            "hdim7" => "m7b5",
            _ => quality, // 7, 9, dim, aug, sus2, sus4...
        };
    }

    private static readonly string[] SharpRoots =
        { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

    /// <summary>Display label moved by <paramref name="semitones"/>
    /// ("Am7" +2 → "Bm7", "F#" −1 → "F"); sharps throughout, as the
    /// models name roots. Display-only, like Simplify.</summary>
    public static string Transpose(string pretty, int semitones)
    {
        if (pretty.Length == 0 || pretty == "—" || semitones % 12 == 0)
            return pretty;
        int rootLength = pretty.Length > 1 && pretty[1] == '#' ? 2 : 1;
        int index = Array.IndexOf(SharpRoots, pretty[..rootLength]);
        if (index < 0)
            return pretty;
        int shifted = ((index + semitones) % 12 + 12) % 12;
        return SharpRoots[shifted] + pretty[rootLength..];
    }

    /// <summary>Beginner view of a display label: extensions and
    /// suspensions collapse to the triad ("Am7"→"Am", "Cmaj7"→"C",
    /// "Dsus4"→"D", "Bm7b5"→"Bdim"). Display-only; analyses keep the
    /// full vocabulary.</summary>
    public static string Simplify(string pretty)
    {
        if (pretty.Length == 0 || pretty == "—")
            return pretty;
        int rootLength = pretty.Length > 1 && pretty[1] == '#' ? 2 : 1;
        string root = pretty[..rootLength];
        string suffix = pretty[rootLength..];
        if (suffix.StartsWith("dim") || suffix.Contains("m7b5"))
            return root + "dim";
        if (suffix.StartsWith("aug"))
            return root + "aug";
        if (suffix.StartsWith("m") && !suffix.StartsWith("maj"))
            return root + "m";
        return root;
    }
}
