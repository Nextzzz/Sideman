namespace Sideman.Core.Analysis;

public sealed record ChordSegment(double Start, double End, Chord Chord, double Confidence)
{
    public double Duration => End - Start;
    public override string ToString() => $"{Start:F2}-{End:F2}s {Chord.Label}";
}

public sealed class ChordRecognizerOptions
{
    /// <summary>Probability of staying on the same chord between frames.
    /// Higher = steadier output, slower to react to real changes.</summary>
    public double SelfTransition { get; init; } = 0.97;

    /// <summary>Segments shorter than this are merged into their neighbor.</summary>
    public double MinSegmentSeconds { get; init; } = 0.4;

    public ChordEmissionModel Emissions { get; init; } = new();
}

/// <summary>
/// Frame chroma → chord timeline. Per-frame template matching is noisy
/// (transients, harmonics), so a Viterbi pass finds the globally best
/// chord sequence: it trades "this frame looks like Am" against
/// "chords don't change every 46 ms".
/// </summary>
public sealed class ChordRecognizer
{
    private readonly ChromaExtractor _chroma;
    private readonly ChordRecognizerOptions _options;
    private readonly ChordEmissionModel _emissions;

    public ChordRecognizer(ChordRecognizerOptions? options = null, ChromaExtractor? chroma = null)
    {
        _options = options ?? new ChordRecognizerOptions();
        _emissions = _options.Emissions;
        _chroma = chroma ?? new ChromaExtractor();
    }

    public IReadOnlyList<ChordSegment> Recognize(float[] samples, int sampleRate)
    {
        var chroma = _chroma.Extract(samples, sampleRate);
        if (chroma.Length == 0)
            return Array.Empty<ChordSegment>();

        int frames = chroma.Length;
        int states = _emissions.StateCount;

        var emissions = new double[frames][];
        for (int t = 0; t < frames; t++)
        {
            emissions[t] = new double[states];
            _emissions.FillEmissions(chroma[t], emissions[t]);
        }

        var path = ViterbiPath(emissions, states);
        return ToSegments(path, emissions, sampleRate);
    }

    private int[] ViterbiPath(double[][] emissions, int states)
    {
        int frames = emissions.Length;
        double logStay = Math.Log(_options.SelfTransition);
        double logSwitch = Math.Log((1.0 - _options.SelfTransition) / (states - 1));

        var score = new double[states];
        var backlink = new int[frames][];
        for (int s = 0; s < states; s++)
            score[s] = emissions[0][s];

        for (int t = 1; t < frames; t++)
        {
            backlink[t] = new int[states];
            // With a uniform switch cost the best predecessor is either
            // "stay" or "come from the globally best previous state".
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
                    next[s] = stay + emissions[t][s];
                    backlink[t][s] = s;
                }
                else
                {
                    next[s] = jump + emissions[t][s];
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

    private IReadOnlyList<ChordSegment> ToSegments(int[] path, double[][] emissions, int sampleRate)
    {
        var segments = new List<ChordSegment>();
        int start = 0;
        for (int t = 1; t <= path.Length; t++)
        {
            if (t == path.Length || path[t] != path[start])
            {
                double confidence = 0;
                for (int i = start; i < t; i++)
                    confidence += Math.Exp(emissions[i][path[start]] / _emissions.EmissionSharpness);
                confidence /= t - start;

                segments.Add(new ChordSegment(
                    Start: start == 0 ? 0 : _chroma.FrameTime(start, sampleRate),
                    End: t == path.Length
                        ? _chroma.FrameTime(path.Length - 1, sampleRate) + 1.0 / _chroma.FrameRate(sampleRate)
                        : _chroma.FrameTime(t, sampleRate),
                    Chord: _emissions.ChordOf(path[start]),
                    Confidence: confidence));
                start = t;
            }
        }
        return MergeShort(segments);
    }

    private IReadOnlyList<ChordSegment> MergeShort(List<ChordSegment> segments)
    {
        // Absorb blips shorter than MinSegmentSeconds into the previous segment.
        var result = new List<ChordSegment>();
        foreach (var seg in segments)
        {
            if (seg.Duration < _options.MinSegmentSeconds && result.Count > 0)
            {
                var prev = result[^1];
                result[^1] = prev with { End = seg.End };
            }
            else if (result.Count > 0 && result[^1].Chord == seg.Chord)
            {
                result[^1] = result[^1] with { End = seg.End };
            }
            else
            {
                result.Add(seg);
            }
        }
        return result;
    }
}
