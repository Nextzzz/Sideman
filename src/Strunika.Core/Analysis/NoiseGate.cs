namespace Strunika.Core.Analysis;

/// <summary>What the noise gate concluded about a frame.</summary>
public enum GateVerdict
{
    /// <summary>Playing — run chord matching.</summary>
    Active,
    /// <summary>Quiet but tonal (a chord decaying into the floor) —
    /// no new evidence; keep whatever chord was already sounding.</summary>
    Quiet,
    /// <summary>Noise or silence — force "no chord".</summary>
    Noise,
}

/// <summary>
/// Decides per frame whether actual playing is happening, so room noise,
/// typing and breathing never reach chord matching. Two independent tests:
///
/// 1. Tonality (spectral flatness): a guitar is strongly harmonic
///    (flatness near 0), room noise is broadband (near 1). Clearly tonal
///    frames always pass, clearly noisy ones always fail — regardless of
///    loudness.
/// 2. Adaptive level: the gate learns the room's noise floor as a low
///    percentile of recent frame RMS and requires the signal to rise
///    MarginDb above it. This is what adapts to any microphone and room.
/// </summary>
public sealed class NoiseGate
{
    private readonly double[] _history;
    private int _count;
    private int _position;

    /// <summary>How far above the learned noise floor a frame must rise.
    /// The user-facing sensitivity knob: lower = more sensitive.</summary>
    public double MarginDb { get; set; } = 12.0;

    /// <summary>Frames flatter than this are noise no matter how loud.</summary>
    public double MaxFlatness { get; set; } = 0.35;

    /// <summary>Frames more tonal than this pass no matter how quiet.</summary>
    public double StrongTonality { get; set; } = 0.05;

    /// <param name="historyLength">Frames of RMS history for the floor
    /// estimate (~20 s at the chroma frame rate by default).</param>
    public NoiseGate(int historyLength = 430)
    {
        _history = new double[historyLength];
    }

    /// <summary>Streaming decision; also feeds the rolling floor estimate.</summary>
    public GateVerdict Assess(double rms, double flatness)
    {
        _history[_position] = rms;
        _position = (_position + 1) % _history.Length;
        if (_count < _history.Length)
            _count++;

        double floor = Percentile(_history.AsSpan(0, _count), 0.1);
        return Decide(rms, flatness, floor, MarginDb, StrongTonality, MaxFlatness);
    }

    /// <summary>Shared rule, also used by offline recognition where the
    /// floor is computed over the whole file at once.</summary>
    public static GateVerdict Decide(
        double rms, double flatness, double noiseFloor,
        double marginDb, double strongTonality = 0.05, double maxFlatness = 0.35)
    {
        if (rms <= 1e-6)
            return GateVerdict.Noise;  // digital silence
        if (flatness > maxFlatness)
            return GateVerdict.Noise;  // unmistakably broadband noise
        if (flatness < strongTonality)
            return GateVerdict.Active; // unmistakably harmonic content
        return rms > noiseFloor * Math.Pow(10.0, marginDb / 20.0)
            ? GateVerdict.Active
            : GateVerdict.Quiet;       // tonal-ish but near the floor: a
                                       // decaying chord, not a new event
    }

    public static double Percentile(ReadOnlySpan<double> values, double p)
    {
        if (values.Length == 0)
            return 0;
        var sorted = values.ToArray();
        Array.Sort(sorted);
        int index = Math.Clamp((int)(p * sorted.Length), 0, sorted.Length - 1);
        return sorted[index];
    }
}
