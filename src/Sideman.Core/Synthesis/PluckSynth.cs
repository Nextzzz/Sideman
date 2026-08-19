namespace Sideman.Core.Synthesis;

/// <summary>
/// Karplus-Strong plucked string — the same three lines of 1983 math that
/// powered our test fixtures, now promoted to the accompaniment engine.
/// </summary>
public static class PluckSynth
{
    public static float[] Pluck(
        double frequency, double seconds, int sampleRate = 44100,
        double decay = 0.996, int seed = 1)
    {
        var rng = new Random(seed);
        int period = Math.Max(2, (int)Math.Round(sampleRate / frequency));

        // Ideal-string excitation: harmonic k at amplitude 1/k, random phase.
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

        var output = new float[(int)(seconds * sampleRate)];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = buffer[i % period];
            buffer[i % period] = (float)(decay * 0.5 *
                (buffer[i % period] + buffer[(i + 1) % period]));
        }

        // Release fade so the tail never clicks.
        int fade = Math.Min((int)(0.05 * sampleRate), output.Length);
        for (int i = 0; i < fade; i++)
            output[output.Length - fade + i] *=
                0.5f * (1f + (float)Math.Cos(Math.PI * i / (fade - 1.0)));
        return output;
    }
}
