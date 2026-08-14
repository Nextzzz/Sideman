namespace Sideman.Core.Analysis;

public enum ChordQuality { None, Major, Minor }

public readonly record struct Chord(int Root, ChordQuality Quality)
{
    public static readonly Chord None = new(-1, ChordQuality.None);

    public string Label => Quality switch
    {
        ChordQuality.Major => Notes.Names[Root],
        ChordQuality.Minor => Notes.Names[Root] + "m",
        _ => "N",
    };

    public override string ToString() => Label;
}

/// <summary>
/// The chord vocabulary: 12 major + 12 minor triads. "No chord" is NOT a
/// template — a flat vector would beat every real chord on noisy frames.
/// Instead it is a separate state with a fixed quality floor (see
/// <see cref="ChordEmissionModel"/>): to be recognized, a chord must beat it.
/// Harmonic weights: a plucked string also sounds its fifth (3rd harmonic),
/// so chord tones get graded weights instead of a flat binary mask.
/// </summary>
public sealed class ChordTemplates
{
    public IReadOnlyList<Chord> Chords { get; }
    public IReadOnlyList<float[]> Vectors { get; }

    public ChordTemplates()
    {
        var chords = new List<Chord>();
        var vectors = new List<float[]>();

        for (int root = 0; root < 12; root++)
        {
            chords.Add(new Chord(root, ChordQuality.Major));
            vectors.Add(Normalize(Build(root, third: 4)));

            chords.Add(new Chord(root, ChordQuality.Minor));
            vectors.Add(Normalize(Build(root, third: 3)));
        }

        Chords = chords;
        Vectors = vectors;
    }

    public int Count => Chords.Count;

    // A real string sounds overtones: pitch-class offsets of harmonics 1..6
    // are 0, 0, +7, 0, +4, +7 semitones. The template must EXPECT them —
    // e.g. an E string always leaks some G# (5th harmonic), and without
    // modeling that, every Em gets misread as E major.
    private static readonly int[] HarmonicOffsets = { 0, 0, 7, 0, 4, 7 };
    private static readonly double[] HarmonicWeights = { 1.0, 0.4, 0.2, 0.1, 0.08, 0.05 };

    private static float[] Build(int root, int third)
    {
        var v = new float[12];
        AddTone(v, root, 1.0);                // root
        AddTone(v, (root + third) % 12, 0.8); // third defines major/minor
        AddTone(v, (root + 7) % 12, 0.9);     // fifth (0.6 was tried: -8pp WCSR)

        // Anti-third: energy at the OTHER third is evidence against this
        // chord. Symmetric -0.3 measured best on GuitarSet (asymmetric
        // variants were tried and scored worse).
        int otherThird = third == 4 ? 3 : 4;
        v[(root + otherThird) % 12] -= 0.3f;
        return v;
    }

    private static void AddTone(float[] v, int pitchClass, double weight)
    {
        for (int h = 0; h < HarmonicOffsets.Length; h++)
            v[(pitchClass + HarmonicOffsets[h]) % 12] += (float)(weight * HarmonicWeights[h]);
    }

    private static float[] Normalize(float[] v)
    {
        double norm = Math.Sqrt(v.Sum(x => (double)x * x));
        if (norm > 0)
            for (int i = 0; i < v.Length; i++)
                v[i] = (float)(v[i] / norm);
        return v;
    }

    public double Similarity(float[] chroma, int templateIndex)
    {
        var t = Vectors[templateIndex];
        double dot = 0;
        for (int i = 0; i < 12; i++)
            dot += chroma[i] * t[i];
        return dot;
    }
}

/// <summary>
/// Shared emission model for offline and streaming recognition, so the two
/// paths can never drift apart. State space = 24 chords + 1 "no chord".
/// </summary>
public sealed class ChordEmissionModel
{
    public ChordTemplates Templates { get; } = new();

    // Defaults below are calibrated on GuitarSet (180 real mic recordings,
    // frame-level WCSR sweep) — not on synthetic signals.

    /// <summary>Sharpens cosine similarities into emissions: sim^beta.</summary>
    public double EmissionSharpness { get; init; } = 2.5;

    /// <summary>Quality floor: a chord must be at least this similar to beat
    /// "no chord". Real-recording chroma is messy; 0.72 (synthetic-derived)
    /// sent a third of all real frames to "N".</summary>
    public double NoChordSimilarity { get; init; } = 0.45;

    /// <summary>Bonus when the chord's root is sounding in the bass. On
    /// strummed guitar the bass note is almost always the root — this is
    /// what separates G from its relatives Em/Bm that share most tones.</summary>
    public double BassRootWeight { get; init; } = 0.3;

    /// <summary>Bonus proportional to the root's strength in the full
    /// chroma. Targets the "C# heard as Fm" family: a chord whose root is
    /// missing from the spectrum is probably not the chord being played.</summary>
    public double RootChromaWeight { get; init; } = 0.0;

    /// <summary>Frames with chroma energy below this are treated as silence.</summary>
    public double SilenceEnergy { get; init; } = 1.0;

    public int StateCount => Templates.Count + 1;
    public int NoneState => Templates.Count;

    public Chord ChordOf(int state) =>
        state == NoneState ? Chord.None : Templates.Chords[state];

    /// <summary>Log-emissions for one frame into <paramref name="dest"/> (length StateCount).</summary>
    public void FillEmissions(in ChromaFrame frame, double[] dest)
    {
        if (frame.Energy < SilenceEnergy)
        {
            // Silence: only "no chord" is plausible.
            for (int s = 0; s < Templates.Count; s++)
                dest[s] = -10.0;
            dest[NoneState] = 0.0;
            return;
        }

        for (int s = 0; s < Templates.Count; s++)
        {
            int root = Templates.Chords[s].Root;
            double sim = Templates.Similarity(frame.Chroma, s)
                         + BassRootWeight * frame.Bass[root]
                         + RootChromaWeight * frame.Chroma[root];
            dest[s] = EmissionSharpness * Math.Log(Math.Max(sim, 1e-3));
        }
        dest[NoneState] = EmissionSharpness * Math.Log(NoChordSimilarity);
    }
}
