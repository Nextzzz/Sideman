namespace Strunika.Cli;

/// <summary>
/// Minimal Karplus-Strong synthesizer for the `demo` command: an audible
/// end-to-end check of the analysis pipeline without recording anything.
/// </summary>
public static class DemoSynth
{
    private const int SampleRate = 44100;

    private static readonly Dictionary<string, int[]> Voicings = new()
    {
        ["G"] = new[] { 43, 47, 50, 55, 59, 67 },
        ["C"] = new[] { 48, 52, 55, 60, 64 },
        ["D"] = new[] { 50, 57, 62, 66 },
        ["Em"] = new[] { 40, 47, 52, 55, 59, 64 },
    };

    public static float[] CampfireProgression(double secondsPerChord = 2.0)
    {
        string[] chords = { "G", "C", "D", "Em" };
        var samples = new float[(int)(chords.Length * secondsPerChord * SampleRate)];
        var rng = new Random(1);
        for (int c = 0; c < chords.Length; c++)
        {
            AddStrum(samples, Voicings[chords[c]], c * secondsPerChord, secondsPerChord / 2, 1.0, rng);
            AddStrum(samples, Voicings[chords[c]], (c + 0.5) * secondsPerChord, secondsPerChord / 2, 0.9, rng);
        }
        return samples;
    }

    private static void AddStrum(
        float[] samples, int[] voicing, double start, double seconds, double gain, Random rng)
    {
        int strumDelay = (int)(0.015 * SampleRate);
        for (int n = 0; n < voicing.Length; n++)
        {
            double freq = 440.0 * Math.Pow(2.0, (voicing[n] - 69) / 12.0);
            var pluck = Pluck(freq, seconds, rng);
            int offset = (int)(start * SampleRate) + n * strumDelay;
            for (int i = 0; i < pluck.Length && offset + i < samples.Length; i++)
                samples[offset + i] += (float)(pluck[i] * gain / voicing.Length);
        }
    }

    private static float[] Pluck(double frequency, double seconds, Random rng)
    {
        int period = (int)Math.Round(SampleRate / frequency);

        // Ideal-string excitation: harmonic k at amplitude 1/k, random phase.
        var buffer = new float[period];
        for (int k = 1; k <= period / 2; k++)
        {
            double phase = rng.NextDouble() * 2 * Math.PI;
            for (int i = 0; i < period; i++)
                buffer[i] += (float)(Math.Cos(2 * Math.PI * k * i / period + phase) / k);
        }
        float peak = buffer.Max(Math.Abs);
        for (int i = 0; i < period && peak > 0; i++)
            buffer[i] /= peak;

        var output = new float[(int)(seconds * SampleRate)];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = buffer[i % period];
            buffer[i % period] = (float)(0.996 * 0.5 * (buffer[i % period] + buffer[(i + 1) % period]));
        }

        int fade = Math.Min((int)(0.1 * SampleRate), output.Length);
        for (int i = 0; i < fade; i++)
            output[output.Length - fade + i] *= 0.5f * (1f + (float)Math.Cos(Math.PI * i / (fade - 1.0)));
        return output;
    }
}
