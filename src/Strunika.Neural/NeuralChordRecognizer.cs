using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Strunika.Neural;

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

    /// <summary>Probability that the chord stays the same between frames
    /// (~93 ms) in the Viterbi smoothing pass. During section transitions
    /// in a mix, single frames vote for in-between chords — raw argmax
    /// turns those into spurious half-second segments.</summary>
    public double ViterbiSelfTransition { get; init; } = 0.9;

    /// <summary>Log-space bonus for chords diatonic to the detected key
    /// in a second decoding pass. Targets parallel major/minor confusions:
    /// in A minor, an A major triad is a stranger while Am is family.
    /// 0 disables the second pass.</summary>
    public double KeyPriorStrength { get; init; } = 0.5;

    /// <summary>Detected key of the last recognized recording ("Am", "C"),
    /// or null when detection was not confident.</summary>
    public string? DetectedKey { get; private set; }

    /// <summary>Per-frame chord labels (~92.6 ms per frame).
    /// <paramref name="smooth"/> false = raw argmax (golden-file parity
    /// with the Python reference); true = Viterbi-smoothed (product).</summary>
    public string[] PredictFrames(float[] samples22050, bool smooth = true)
    {
        var logProbs = PredictLogProbs(samples22050);
        DetectedKey = null;
        if (logProbs.Length == 0)
            return Array.Empty<string>();

        if (!smooth || ViterbiSelfTransition <= 0)
            return logProbs.Select(ArgMax).Select(i => _labels[i]).ToArray();

        var path = ViterbiPath(logProbs);

        // Second pass with a diatonic prior when the key is clear.
        if (KeyPriorStrength > 0)
        {
            var key = KeyPrior.Estimate(path, _labels);
            if (key != null)
            {
                DetectedKey = key.Value.Name;
                var bonus = KeyPrior.StateBonuses(key.Value, _labels, KeyPriorStrength);
                for (int t = 0; t < logProbs.Length; t++)
                    for (int c = 0; c < bonus.Length; c++)
                        logProbs[t][c] += bonus[c];
                path = ViterbiPath(logProbs);
            }
        }
        return path.Select(i => _labels[i]).ToArray();
    }

    private float[][] PredictLogProbs(float[] samples22050)
    {
        var features = _cqt.Extract(samples22050);
        int frames = features.Length;
        var result = new float[frames][];

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
                // Log-softmax keeps Viterbi's reward scale comparable
                // across frames regardless of raw logit magnitudes.
                var row = new float[_labels.Length];
                double max = double.MinValue;
                for (int c = 0; c < _labels.Length; c++)
                    max = Math.Max(max, logits[0, t, c]);
                double sum = 0;
                for (int c = 0; c < _labels.Length; c++)
                    sum += Math.Exp(logits[0, t, c] - max);
                double logSum = max + Math.Log(sum);
                for (int c = 0; c < _labels.Length; c++)
                    row[c] = (float)(logits[0, t, c] - logSum);
                result[dst] = row;
            }
        }
        return result;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int c = 1; c < row.Length; c++)
            if (row[c] > row[best])
                best = c;
        return best;
    }

    private int[] ViterbiPath(float[][] logProbs)
    {
        int frames = logProbs.Length;
        int states = _labels.Length;
        double logStay = Math.Log(ViterbiSelfTransition);
        double logSwitch = Math.Log((1.0 - ViterbiSelfTransition) / (states - 1));

        var score = new double[states];
        var backlink = new int[frames][];
        for (int s = 0; s < states; s++)
            score[s] = logProbs[0][s];

        for (int t = 1; t < frames; t++)
        {
            backlink[t] = new int[states];
            int bestPrev = 0;
            for (int s = 1; s < states; s++)
                if (score[s] > score[bestPrev])
                    bestPrev = s;

            var next = new double[states];
            for (int s = 0; s < states; s++)
            {
                double stay = score[s] + logStay;
                double jump = score[bestPrev] + logSwitch;
                if (stay >= jump || bestPrev == s)
                {
                    next[s] = stay + logProbs[t][s];
                    backlink[t][s] = s;
                }
                else
                {
                    next[s] = jump + logProbs[t][s];
                    backlink[t][s] = bestPrev;
                }
            }
            score = next;
        }

        var path = new int[frames];
        int cur = 0;
        for (int s = 1; s < states; s++)
            if (score[s] > score[cur])
                cur = s;
        for (int t = frames - 1; t >= 0; t--)
        {
            path[t] = cur;
            if (t > 0)
                cur = backlink[t][cur];
        }
        return path;
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
