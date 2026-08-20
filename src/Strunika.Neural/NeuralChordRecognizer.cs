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

    /// <summary>Second window pass offset by half a window, averaged with
    /// the first: frames near a window edge get two-sided context. One
    /// extra inference pass (~0.25 s per 3-minute song).</summary>
    public bool OverlapWindows { get; init; }

    /// <summary>Pitch-shift test-time augmentation: extra passes on the
    /// CQT rolled ±1..N semitones, predictions rolled back to the true
    /// root and averaged. Two passes per semitone. 0 disables.</summary>
    public int PitchTtaSemitones { get; init; }

    private const float LogFloor = -13.8155106f; // log(1e-6): CQT silence level

    /// <summary>Per-frame log-probabilities averaged over all enabled
    /// passes (probability space), log-softmax for Viterbi's reward scale.</summary>
    private float[][] PredictLogProbs(float[] samples22050)
    {
        var features = _cqt.Extract(samples22050);
        int frames = features.Length;
        if (frames == 0)
            return Array.Empty<float[]>();
        int states = _labels.Length;

        // Label layout root*qualities+quality, then X, N — needed to roll
        // shifted predictions back. Models without it get no TTA.
        int qualities = (states - 2) % 12 == 0 ? (states - 2) / 12 : 0;

        var passes = new List<(int Offset, int Shift)> { (0, 0) };
        if (OverlapWindows)
            passes.Add((_timestep / 2, 0));
        if (qualities > 0)
            for (int k = 1; k <= PitchTtaSemitones; k++)
            {
                passes.Add((0, k));
                passes.Add((0, -k));
            }

        var sum = new double[frames][];
        var cover = new int[frames];
        for (int t = 0; t < frames; t++)
            sum[t] = new double[states];

        var probs = new double[states];
        foreach (var (offset, shift) in passes)
        {
            var input = shift == 0 ? features : ShiftBins(features, shift);
            for (int start = offset; start < frames; start += _timestep)
            {
                int valid = Math.Min(_timestep, frames - start);
                var tensor = new DenseTensor<float>(new[] { 1, _timestep, CqtExtractor.Bins });
                for (int t = 0; t < valid; t++)
                    for (int b = 0; b < CqtExtractor.Bins; b++)
                        tensor[0, t, b] = input[start + t][b];
                // (tail window stays zero-padded, as in training)

                using var output = _session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("features", tensor),
                });
                var logits = (DenseTensor<float>)output[0].Value;

                for (int t = 0; t < valid; t++)
                {
                    double max = double.MinValue;
                    for (int c = 0; c < states; c++)
                        max = Math.Max(max, logits[0, t, c]);
                    double total = 0;
                    for (int c = 0; c < states; c++)
                        total += probs[c] = Math.Exp(logits[0, t, c] - max);

                    var row = sum[start + t];
                    for (int c = 0; c < states; c++)
                    {
                        // Model heard audio `shift` semitones up, so its
                        // root r is the true root r - shift.
                        int dst = c;
                        if (shift != 0 && c < qualities * 12)
                        {
                            int root = (c / qualities - shift + 12) % 12;
                            dst = root * qualities + c % qualities;
                        }
                        row[dst] += probs[c] / total;
                    }
                    cover[start + t]++;
                }
            }
        }

        var result = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[states];
            for (int c = 0; c < states; c++)
                row[c] = (float)Math.Log(sum[t][c] / cover[t] + 1e-12);
            result[t] = row;
        }
        return result;
    }

    /// <summary>CQT rolled by 2 bins per semitone (24 bins/octave);
    /// vacated bins get the silence floor, as in training augmentation.</summary>
    private static float[][] ShiftBins(float[][] features, int semitones)
    {
        int bins = CqtExtractor.Bins;
        int delta = 2 * semitones;
        var shifted = new float[features.Length][];
        for (int t = 0; t < features.Length; t++)
        {
            var row = new float[bins];
            Array.Fill(row, LogFloor);
            for (int b = 0; b < bins; b++)
            {
                int src = b - delta;
                if (src >= 0 && src < bins)
                    row[b] = features[t][src];
            }
            shifted[t] = row;
        }
        return shifted;
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
