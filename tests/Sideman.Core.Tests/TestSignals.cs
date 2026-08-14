using Sideman.Core.Analysis;

namespace Sideman.Core.Tests;

/// <summary>
/// Synthetic guitar signals with exactly known ground truth.
/// Karplus-Strong (1983): a noise burst circulating through a one-period
/// delay line with averaging decays into a convincing plucked string.
/// </summary>
public static class TestSignals
{
    public const int SampleRate = 44100;

    public static float[] Sine(double frequency, double seconds, double amplitude = 0.5)
    {
        var x = new float[(int)(seconds * SampleRate)];
        for (int i = 0; i < x.Length; i++)
            x[i] = (float)(amplitude * Math.Sin(2 * Math.PI * frequency * i / SampleRate));
        return x;
    }

    public static float[] Pluck(double frequency, double seconds, int seed = 1, double decay = 0.996)
    {
        var rng = new Random(seed);
        int period = (int)Math.Round(SampleRate / frequency);

        // Ideal-string excitation: harmonic k starts at amplitude 1/k with
        // a random phase — the classic plucked-string spectrum. Raw white
        // noise makes the harmonic balance a seed lottery (a string may
        // barely contain its own fundamental); a flat spectrum overloads
        // the highs. Both were measured to break chord tests.
        var buffer = new float[period];
        for (int k = 1; k <= period / 2; k++)
        {
            double phase = rng.NextDouble() * 2 * Math.PI;
            for (int i = 0; i < period; i++)
                buffer[i] += (float)(Math.Cos(2 * Math.PI * k * i / period + phase) / k);
        }
        float peak = buffer.Max(Math.Abs);
        if (peak > 0)
            for (int i = 0; i < period; i++)
                buffer[i] /= peak;

        var output = new float[(int)(seconds * SampleRate)];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = buffer[i % period];
            buffer[i % period] = (float)(decay * 0.5 * (buffer[i % period] + buffer[(i + 1) % period]));
        }

        // Release fade — a raw cutoff would be an audible click.
        int fade = Math.Min((int)(0.1 * SampleRate), output.Length);
        for (int i = 0; i < fade; i++)
        {
            double g = 0.5 * (1 + Math.Cos(Math.PI * i / (fade - 1.0)));
            output[output.Length - fade + i] *= (float)g;
        }
        for (int i = 0; i < output.Length; i++)
            output[i] *= 0.7f;
        return output;
    }

    /// <summary>Strummed chord: strings started ~15 ms apart, like a real strum.</summary>
    public static float[] Strum(IReadOnlyList<int> midiNotes, double seconds, int seed = 1)
    {
        var x = new float[(int)(seconds * SampleRate)];
        int strumDelay = (int)(0.015 * SampleRate);
        for (int n = 0; n < midiNotes.Count; n++)
        {
            double freq = Notes.FrequencyFromMidi(midiNotes[n]);
            var pluck = Pluck(freq, seconds - n * strumDelay / (double)SampleRate, seed + n);
            int offset = n * strumDelay;
            for (int i = 0; i < pluck.Length && offset + i < x.Length; i++)
                x[offset + i] += pluck[i] / midiNotes.Count;
        }
        return x;
    }

    /// <summary>Common open/barre guitar voicings, bottom string first (MIDI numbers).</summary>
    public static readonly Dictionary<string, int[]> Voicings = new()
    {
        ["C"] = new[] { 48, 52, 55, 60, 64 },        // x32010
        ["G"] = new[] { 43, 47, 50, 55, 59, 67 },    // 320003
        ["D"] = new[] { 50, 57, 62, 66 },            // xx0232
        ["E"] = new[] { 40, 47, 52, 56, 59, 64 },    // 022100
        ["A"] = new[] { 45, 52, 57, 61, 64 },        // x02220
        ["F"] = new[] { 41, 48, 53, 57, 60, 65 },    // 133211 barre
        ["Am"] = new[] { 45, 52, 57, 60, 64 },       // x02210
        ["Em"] = new[] { 40, 47, 52, 55, 59, 64 },   // 022000
        ["Dm"] = new[] { 50, 57, 62, 65 },           // xx0231
    };

    /// <summary>
    /// A chord progression with known boundaries. Each chord is strummed
    /// twice — on the boundary and halfway through, slightly softer — the
    /// way players actually keep a chord alive instead of letting it decay
    /// into ambiguity.
    /// </summary>
    public static (float[] Samples, (double Start, string Label)[] Truth) Progression(
        string[] chords, double secondsPerChord, int seed = 1)
    {
        var samples = new float[(int)(chords.Length * secondsPerChord * SampleRate)];
        var truth = new (double, string)[chords.Length];
        for (int c = 0; c < chords.Length; c++)
        {
            double start = c * secondsPerChord;
            truth[c] = (start, chords[c]);
            // Second strum REPLACES the first one's tail (a real strum damps
            // the old vibration). Overlapping identical synthetic strings
            // phase-cancel and fake a different chord — measured, not theory.
            AddStrum(samples, Voicings[chords[c]], start, secondsPerChord / 2, 1.0, seed + c * 10);
            AddStrum(samples, Voicings[chords[c]], start + secondsPerChord / 2, secondsPerChord / 2, 0.9, seed + c * 10 + 5);
        }
        return (samples, truth);
    }

    private static void AddStrum(
        float[] samples, int[] voicing, double start, double seconds, double gain, int seed)
    {
        var strum = Strum(voicing, seconds, seed);
        int offset = (int)(start * SampleRate);
        for (int i = 0; i < strum.Length && offset + i < samples.Length; i++)
            samples[offset + i] += (float)(strum[i] * gain);
    }

    public static void AddNoise(float[] samples, double amplitude, int seed = 99)
    {
        var rng = new Random(seed);
        for (int i = 0; i < samples.Length; i++)
            samples[i] += (float)((rng.NextDouble() * 2 - 1) * amplitude);
    }
}
