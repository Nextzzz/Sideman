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

    /// <summary>Merged chord timeline for a full recording.</summary>
    public IReadOnlyList<NeuralChordSegment> Recognize(float[] samples22050)
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
        return segments;
    }

    public void Dispose() => _session.Dispose();
}
