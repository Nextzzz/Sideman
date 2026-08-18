using System.IO;
using System.Text.Json;
using Sideman.Core.Diagnostics;

namespace Sideman.App.Services;

public sealed record SavedSegment(double Start, double End, string Chord);

public sealed record SavedAnalysis(
    string Source,
    string AudioPath,
    DateTime AnalyzedAt,
    double DurationSeconds,
    double Bpm,
    string Engine,
    List<SavedSegment> Segments);

/// <summary>
/// Persists every analysis as JSON so results can be revisited (and
/// discussed) without re-running anything: one timestamped file per
/// analysis plus last.json always holding the most recent one.
/// </summary>
public static class AnalysisStore
{
    public static string Directory { get; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "Sideman", "analyses");

    public static void Save(SavedAnalysis analysis)
    {
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            string json = JsonSerializer.Serialize(
                analysis, new JsonSerializerOptions { WriteIndented = true });

            string name = $"{analysis.AnalyzedAt:yyyyMMdd_HHmmss}_" +
                          Path.GetFileNameWithoutExtension(analysis.AudioPath) + ".json";
            File.WriteAllText(Path.Combine(Directory, name), json);
            File.WriteAllText(Path.Combine(Directory, "last.json"), json);
            FileLog.Info($"Analysis saved: {name}");
        }
        catch (Exception ex)
        {
            FileLog.Error("Failed to save analysis", ex);
        }
    }
}
