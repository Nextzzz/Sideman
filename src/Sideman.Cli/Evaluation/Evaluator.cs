using System.Collections.Concurrent;
using System.Text;
using Sideman.Core.Analysis;
using Sideman.Media;

namespace Sideman.Cli.Evaluation;

public sealed record FileResult(
    string Name, int Scored, int Correct, double Accuracy, string TopConfusion);

/// <summary>
/// Frame-level chord evaluation over GuitarSet: for every 100 ms of every
/// file, compare the recognized chord against the instructed leadsheet
/// chord (majmin vocabulary). The number that comes out is WCSR — weighted
/// chord symbol recall, the standard chord-recognition benchmark metric.
/// </summary>
public sealed class Evaluator
{
    private const double Step = 0.1;

    public ChordRecognizerOptions Options { get; init; } = new();

    /// <summary>Only files whose name contains this (e.g. "_comp") are used.</summary>
    public string Filter { get; init; } = "_comp";

    public int Limit { get; init; } = int.MaxValue;

    /// <summary>Path to a BTC ONNX model; when set, evaluation runs the
    /// neural recognizer instead of the template engine.</summary>
    public string? NeuralModel { get; init; }

    /// <summary>When set, only files starting with one of these prefixes
    /// are evaluated (e.g. ["04_","05_"] = held-out players).</summary>
    public string[]? FilePrefixes { get; init; }

    public (List<FileResult> Files, Dictionary<string, int> Confusions, long Scored, long Correct)
        Run(string jamsDir, string audioDir)
    {
        var jamsFiles = Directory.GetFiles(jamsDir, "*.jams")
            .Where(f => Path.GetFileNameWithoutExtension(f).Contains(Filter))
            .Where(f => FilePrefixes == null
                || FilePrefixes.Any(p => Path.GetFileName(f).StartsWith(p)))
            .OrderBy(f => f)
            .Take(Limit)
            .ToArray();

        var results = new ConcurrentBag<FileResult>();
        var confusions = new ConcurrentDictionary<string, int>();
        long totalScored = 0, totalCorrect = 0;

        using var neural = NeuralModel == null
            ? null
            : new Sideman.Neural.NeuralChordRecognizer(NeuralModel);

        Parallel.ForEach(jamsFiles, jamsPath =>
        {
            string stem = Path.GetFileNameWithoutExtension(jamsPath);
            string wavPath = Path.Combine(audioDir, stem + "_mic.wav");
            if (!File.Exists(wavPath))
                return;

            var (truth, duration) = JamsChords.Read(jamsPath);

            Func<double, string> predictAt;
            if (neural != null)
            {
                var (samples22k, _) = AudioLoader.LoadMono(
                    wavPath, Sideman.Neural.CqtExtractor.SampleRate);
                var frames = neural.PredictFrames(samples22k);
                double spf = Sideman.Neural.CqtExtractor.ChunkSeconds
                             / Sideman.Neural.CqtExtractor.FramesPerChunk;
                predictAt = t =>
                {
                    int i = Math.Clamp((int)(t / spf), 0, frames.Length - 1);
                    return ChordMapping.ToMajMin(frames[i]) ?? "X";
                };
            }
            else
            {
                var (samples, sampleRate) = AudioLoader.LoadMono(wavPath);
                var segments = new ChordRecognizer(Options).Recognize(samples, sampleRate);
                predictAt = t => PredictedAt(segments, t);
            }

            int scored = 0, correct = 0;
            var localConfusions = new Dictionary<string, int>();

            for (double t = 0; t < duration; t += Step)
            {
                string? truthLabel = MapTruthAt(truth, t);
                if (truthLabel == null)
                    continue; // dim/aug/sus or gap — outside the vocabulary

                string predLabel = predictAt(t);
                scored++;
                if (predLabel == truthLabel)
                {
                    correct++;
                }
                else
                {
                    string key = $"{truthLabel}->{predLabel}";
                    localConfusions[key] = localConfusions.GetValueOrDefault(key) + 1;
                }
            }

            foreach (var (key, count) in localConfusions)
                confusions.AddOrUpdate(key, count, (_, v) => v + count);

            string top = localConfusions.Count == 0
                ? ""
                : localConfusions.MaxBy(kv => kv.Value).Key;
            results.Add(new FileResult(
                stem, scored, correct, scored == 0 ? 0 : (double)correct / scored, top));
            Interlocked.Add(ref totalScored, scored);
            Interlocked.Add(ref totalCorrect, correct);
        });

        return (results.OrderBy(r => r.Accuracy).ToList(),
                confusions.ToDictionary(kv => kv.Key, kv => kv.Value),
                totalScored, totalCorrect);
    }

    private static string? MapTruthAt(List<TruthSegment> truth, double time)
    {
        foreach (var segment in truth)
            if (time >= segment.Start && time < segment.End)
                return ChordMapping.ToMajMin(segment.RawLabel);
        return null;
    }

    private static string PredictedAt(IReadOnlyList<ChordSegment> predicted, double time)
    {
        foreach (var segment in predicted)
            if (time >= segment.Start && time < segment.End)
                return segment.Chord.Label;
        return "N";
    }

    public static string Report(
        List<FileResult> files, Dictionary<string, int> confusions, long scored, long correct)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Files: {files.Count}   Frames scored: {scored}");
        sb.AppendLine($"WCSR (majmin): {(double)correct / scored:P2}");
        sb.AppendLine();

        sb.AppendLine("Worst 10 files:");
        foreach (var f in files.Take(10))
            sb.AppendLine($"  {f.Accuracy:P1}  {f.Name}  (top err: {f.TopConfusion})");
        sb.AppendLine();

        sb.AppendLine("Top 15 confusions (truth->predicted):");
        foreach (var (key, count) in confusions.OrderByDescending(kv => kv.Value).Take(15))
            sb.AppendLine($"  {key,-12} {count,6} frames ({count * Step:F0}s)");
        return sb.ToString();
    }
}
