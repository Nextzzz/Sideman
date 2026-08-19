using System.Text.Json;

namespace Strunika.Cli.Evaluation;

public sealed record TruthSegment(double Start, double End, string RawLabel);

/// <summary>
/// Reads the chord ground truth from a GuitarSet .jams file. GuitarSet has
/// two chord annotations per file: the instructed leadsheet chords (simple
/// "D#:maj" labels — what the player was told to play, and what a chord app
/// should display) and a detailed performance transcription. We take the
/// instructed one: the first chord annotation without a data_source.
/// </summary>
public static class JamsChords
{
    public static (List<TruthSegment> Segments, double Duration) Read(string jamsPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(jamsPath));
        var root = doc.RootElement;

        double duration = root.GetProperty("file_metadata").GetProperty("duration").GetDouble();

        JsonElement? chosen = null;
        foreach (var annotation in root.GetProperty("annotations").EnumerateArray())
        {
            if (annotation.GetProperty("namespace").GetString() != "chord")
                continue;
            string source = annotation
                .GetProperty("annotation_metadata")
                .GetProperty("data_source")
                .GetString() ?? "";
            if (source.Length == 0)
            {
                chosen = annotation;
                break;
            }
            chosen ??= annotation;
        }

        var segments = new List<TruthSegment>();
        if (chosen == null)
            return (segments, duration);

        foreach (var item in chosen.Value.GetProperty("data").EnumerateArray())
        {
            double time = item.GetProperty("time").GetDouble();
            double dur = item.GetProperty("duration").GetDouble();
            string value = item.GetProperty("value").GetString() ?? "N";
            segments.Add(new TruthSegment(time, time + dur, value));
        }
        return (segments, duration);
    }
}
