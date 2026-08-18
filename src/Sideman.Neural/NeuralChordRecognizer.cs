using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Sideman.Neural;

public sealed record NeuralChordSegment(double Start, double End, string Label)
{
    public override string ToString() => $"{Start:F2}-{End:F2}s {Label}";
}

/// <summary>
/// BTC chord recognition through ONNX Runtime: 22050 Hz mono samples in,
/// chord timeline out. Feature normalization is baked into the ONNX graph,
/// so this class only computes log-CQT and windows it per 108 frames.
/// </summary>
public sealed class NeuralChordRecognizer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly CqtExtractor _cqt = new();
    private readonly string[] _labels;
    private readonly int _timestep;

    public NeuralChordRecognizer(string onnxPath)
    {
        _session = new InferenceSession(onnxPath);
        string configPath = Path.ChangeExtension(onnxPath, ".json");
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        _labels = doc.RootElement.GetProperty("labels")
            .EnumerateArray().Select(e => e.GetString()!).ToArray();
        _timestep = doc.RootElement.GetProperty("timestep").GetInt32();
    }

    public IReadOnlyList<string> Labels => _labels;

    /// <summary>Per-frame chord labels (~92.6 ms per frame).</summary>
    public string[] PredictFrames(float[] samples22050)
    {
        var features = _cqt.Extract(samples22050);
        int frames = features.Length;
        if (frames == 0)
            return Array.Empty<string>();

        var result = new string[frames];
        int windows = (frames + _timestep - 1) / _timestep;
        for (int w = 0; w < windows; w++)
        {
            var tensor = new DenseTensor<float>(new[] { 1, _timestep, CqtExtractor.Bins });
            for (int t = 0; t < _timestep; t++)
            {
                int src = w * _timestep + t;
                if (src >= frames)
                    break; // zero padding for the tail window
                for (int b = 0; b < CqtExtractor.Bins; b++)
                    tensor[0, t, b] = features[src][b];
            }

            using var output = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("features", tensor),
            });
            var logits = (DenseTensor<float>)output[0].Value;

            for (int t = 0; t < _timestep; t++)
            {
                int dst = w * _timestep + t;
                if (dst >= frames)
                    break;
                int best = 0;
                for (int c = 1; c < _labels.Length; c++)
                    if (logits[0, t, c] > logits[0, t, best])
                        best = c;
                result[dst] = _labels[best];
            }
        }
        return result;
    }

    /// <summary>
    /// Labels for one window of frames (zero-padded to the model's
    /// timestep). Used by the live sliding-window detector.
    /// </summary>
    public string[] PredictWindow(IReadOnlyList<float[]> window)
    {
        var tensor = new DenseTensor<float>(new[] { 1, _timestep, CqtExtractor.Bins });
        int count = Math.Min(window.Count, _timestep);
        for (int t = 0; t < count; t++)
            for (int b = 0; b < CqtExtractor.Bins; b++)
                tensor[0, t, b] = window[t][b];

        using var output = _session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("features", tensor),
        });
        var logits = (DenseTensor<float>)output[0].Value;

        var labels = new string[count];
        for (int t = 0; t < count; t++)
        {
            int best = 0;
            for (int c = 1; c < _labels.Length; c++)
                if (logits[0, t, c] > logits[0, t, best])
                    best = c;
            labels[t] = _labels[best];
        }
        return labels;
    }

    /// <summary>Merged chord timeline for a full recording. Blips shorter
    /// than <paramref name="minSegmentSeconds"/> are absorbed into their
    /// neighbor — frame-level flips are not musical events.</summary>
    public IReadOnlyList<NeuralChordSegment> Recognize(
        float[] samples22050, double minSegmentSeconds = 0.3)
    {
        var frames = PredictFrames(samples22050);
        double spf = _cqt.SecondsPerFrame;

        var segments = new List<NeuralChordSegment>();
        int start = 0;
        for (int t = 1; t <= frames.Length; t++)
        {
            if (t == frames.Length || frames[t] != frames[start])
            {
                segments.Add(new NeuralChordSegment(start * spf, t * spf, frames[start]));
                start = t;
            }
        }

        var merged = new List<NeuralChordSegment>();
        foreach (var segment in segments)
        {
            if (merged.Count > 0 && segment.End - segment.Start < minSegmentSeconds)
                merged[^1] = merged[^1] with { End = segment.End };
            else if (merged.Count > 0 && merged[^1].Label == segment.Label)
                merged[^1] = merged[^1] with { End = segment.End };
            else
                merged.Add(segment);
        }
        return merged;
    }

    public void Dispose() => _session.Dispose();
}
