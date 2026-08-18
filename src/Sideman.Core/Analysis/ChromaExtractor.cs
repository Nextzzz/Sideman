using Sideman.Core.Dsp;

namespace Sideman.Core.Analysis;

/// <summary>
/// One analysis frame: full-range chroma, bass-range chroma, plus the raw
/// signal statistics downstream logic needs — band-limited RMS (how loud),
/// spectral flatness (how noise-like: ~1 = broadband noise, ~0 = tonal)
/// and spectral flux (how much NEW energy vs the previous frame — high
/// during strum attacks, when the chroma is full of transient garbage).
/// </summary>
public readonly record struct ChromaFrame(
    float[] Chroma, float[] Bass, float Energy, float Rms, float Flatness, float Flux);

/// <summary>
/// Folds a magnitude spectrum into 12-bin pitch-class (chroma) vectors:
/// all Cs of every octave into bin 0, all C#s into bin 1, etc.
/// Two vectors per frame: full range (what harmony is sounding) and bass
/// range (which note is in the bass — on guitar almost always the root).
/// </summary>
public sealed class ChromaExtractor
{
    // A long FFT is required: at E2 (82 Hz) adjacent semitones are only
    // ~5 Hz apart, so we need ~5 Hz bins to separate bass notes.
    public int NFft { get; init; } = 8192;
    public int Hop { get; init; } = 2048;
    public double FMin { get; init; } = 55.0;
    public double FMax { get; init; } = 2200.0;
    public double BassFMax { get; init; } = 180.0;

    /// <summary>Log-compression strength. Kept moderate on purpose: strong
    /// compression (γ=100) inflates leakage skirts and weak harmonics until
    /// they rival real notes — measured, not theoretical.</summary>
    public double Gamma { get; init; } = 5.0;

    public double FrameRate(int sampleRate) => sampleRate / (double)Hop;

    public double FrameTime(int frameIndex, int sampleRate) =>
        (frameIndex * (double)Hop + NFft / 2.0) / sampleRate;

    /// <summary>Global tuning deviation in semitones (-0.5..0.5), subtracted
    /// before pitch-class mapping. A guitar tuned 30 cents flat otherwise
    /// lands between semitone bands and the whole chroma collapses.</summary>
    public double TuningOffset { get; set; }

    // Previous frame's compressed spectrum, for the flux computation.
    // Makes FoldFrame stateful: frames must be fed in temporal order.
    private double[]? _previousCompressed;

    public ChromaFrame[] Extract(float[] samples, int sampleRate)
    {
        _previousCompressed = null; // fresh flux baseline per recording
        var stft = new Stft(NFft, Hop);
        int frames = stft.FrameCount(samples.Length);
        var result = new ChromaFrame[frames];

        int f = 0;
        foreach (var magnitude in stft.Magnitudes(samples))
            result[f++] = FoldFrame(magnitude, sampleRate);
        return result;
    }

    /// <summary>
    /// Estimates how far the recording is from A440, as the circular mean
    /// of every bin's deviation from the nearest semitone, weighted by
    /// magnitude. Call before Extract, assign to <see cref="TuningOffset"/>.
    /// </summary>
    public double EstimateTuning(float[] samples, int sampleRate)
    {
        var stft = new Stft(NFft, Hop * 4); // sparse pass is enough
        double binHz = sampleRate / (double)NFft;
        int kMin = Math.Max(1, (int)Math.Ceiling(100.0 / binHz));
        int kMax = (int)Math.Floor(Math.Min(FMax, 2000.0) / binHz);

        double x = 0, y = 0;
        foreach (var magnitude in stft.Magnitudes(samples))
        {
            for (int k = kMin; k <= kMax && k < magnitude.Length; k++)
            {
                double midi = Notes.MidiFromFrequency(k * binHz);
                double deviation = midi - Math.Round(midi); // -0.5..0.5
                // Circular statistics: deviation is an angle on a semitone circle.
                x += magnitude[k] * Math.Cos(2 * Math.PI * deviation);
                y += magnitude[k] * Math.Sin(2 * Math.PI * deviation);
            }
        }
        double offset = x == 0 && y == 0 ? 0 : Math.Atan2(y, x) / (2 * Math.PI);

        // Real guitars drift by cents, not quarter-tones. An estimate near
        // ±0.5 is the ambiguous boundary where a wrong sign shifts EVERY
        // chord by a semitone (measured: whole files dropping to 0%
        // accuracy) — better to apply no correction at all.
        return Math.Abs(offset) <= 0.25 ? offset : 0.0;
    }

    /// <summary>Fold one magnitude spectrum into chroma + bass chroma.</summary>
    public ChromaFrame FoldFrame(float[] magnitude, int sampleRate)
    {
        var full = new double[12];
        var bassNotes = new double[128]; // per MIDI note, bass band only
        double binHz = sampleRate / (double)NFft;
        int kMin = Math.Max(1, (int)Math.Ceiling(FMin / binHz));
        int kMax = Math.Min(magnitude.Length - 1, (int)Math.Floor(FMax / binHz));

        // Band-limited power statistics for the noise gate. The band also
        // conveniently excludes mains hum and rumble below FMin.
        double sumPower = 0, sumLogPower = 0;
        int bins = 0;
        _previousCompressed ??= new double[magnitude.Length];
        double flux = 0;
        for (int k = kMin; k <= kMax; k++)
        {
            double p = (double)magnitude[k] * magnitude[k];
            sumPower += p;
            sumLogPower += Math.Log(p + 1e-12);
            bins++;

            // Spectral flux on compressed magnitudes (positive changes only).
            double compressed = Math.Log(1.0 + Gamma * magnitude[k]);
            double rise = compressed - _previousCompressed[k];
            if (rise > 0)
                flux += rise;
            _previousCompressed[k] = compressed;
        }
        float rms = (float)Math.Sqrt(sumPower / Math.Max(bins, 1));
        float flatness = bins == 0
            ? 1f
            : (float)(Math.Exp(sumLogPower / bins) / (sumPower / bins + 1e-12));

        for (int k = kMin; k <= kMax; k++)
        {
            double freq = k * binHz;
            double midi = Notes.MidiFromFrequency(freq) - TuningOffset;
            int nearest = (int)Math.Round(midi);
            int pc = ((nearest % 12) + 12) % 12;

            // Semitone-proximity weight: at low frequencies FFT bins are
            // wider than a semitone, and leakage from a loud bass note
            // lands in bins halfway to the NEXT pitch class (a strong B2
            // fakes a C). A bin only counts as much as it is unambiguous.
            double offset = Math.Abs(midi - nearest); // 0 = center, 0.5 = boundary
            double weight = Math.Max(0.0, 1.0 - 2.0 * offset);

            // Log compression: quiet harmonics matter to the ear.
            full[pc] += weight * Math.Log(1.0 + Gamma * magnitude[k]);

            // Bass stays LINEAR: here we ask which note physically carries
            // the energy. Only LOCAL PEAKS count: the mainlobe skirt of a
            // loud C3 slopes right through the B2 semitone band, and
            // without this check the skirt registers as a phantom bass B.
            bool isLocalPeak = k > 0 && k + 1 < magnitude.Length
                && magnitude[k] >= magnitude[k - 1]
                && magnitude[k] >= magnitude[k + 1];
            if (isLocalPeak && freq <= BassFMax && nearest is >= 0 and < 128)
                bassNotes[nearest] += weight * magnitude[k];
        }

        double rawNorm = 0;
        for (int i = 0; i < 12; i++)
            rawNorm += full[i] * full[i];
        float energy = (float)Math.Sqrt(rawNorm);

        // Log compression leaves a near-uniform pedestal in every pitch class
        // (noise floor, leakage). Subtract the median so only what actually
        // STANDS OUT survives — otherwise a flat "no chord" pattern beats
        // every real chord.
        var sorted = (double[])full.Clone();
        Array.Sort(sorted);
        double median = (sorted[5] + sorted[6]) / 2;
        for (int i = 0; i < 12; i++)
            full[i] = Math.Max(full[i] - median, 0);

        double norm = 0;
        for (int i = 0; i < 12; i++)
            norm += full[i] * full[i];
        norm = Math.Sqrt(norm);

        var chroma = new float[12];
        if (norm > 1e-9)
            for (int i = 0; i < 12; i++)
                chroma[i] = (float)(full[i] / norm);

        // The bass NOTE is the lowest note holding a significant share of
        // the strongest note's energy in the band. Taking the loudest class
        // instead misroots chords whose voicing has two notes in the band
        // (C3+E3 in open C: a decayed C3 would hand the root to E).
        var bassOut = new float[12];
        double bassMax = bassNotes.Max();
        if (bassMax > 1e-9)
        {
            for (int note = 0; note < bassNotes.Length; note++)
            {
                if (bassNotes[note] >= 0.35 * bassMax)
                {
                    bassOut[note % 12] = 1f;
                    break;
                }
            }
        }

        return new ChromaFrame(chroma, bassOut, energy, rms, flatness, (float)flux);
    }
}
