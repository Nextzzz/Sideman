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
}
