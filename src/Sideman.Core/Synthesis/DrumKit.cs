namespace Sideman.Core.Synthesis;

/// <summary>Procedural drum one-shots: kick, snare, closed hat.</summary>
public static class DrumKit
{
    public static float[] Kick(int sampleRate = 44100)
    {
        // Pitch sweep 150 -> 45 Hz with a fast exponential decay.
        int n = (int)(0.16 * sampleRate);
        var samples = new float[n];
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)sampleRate;
            double freq = 45 + 105 * Math.Exp(-t * 28);
            phase += 2 * Math.PI * freq / sampleRate;
            samples[i] = (float)(Math.Sin(phase) * Math.Exp(-t * 22));
        }
        return samples;
    }

    public static float[] Snare(int sampleRate = 44100)
    {
        int n = (int)(0.14 * sampleRate);
        var samples = new float[n];
        var rng = new Random(3);
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)sampleRate;
            phase += 2 * Math.PI * 190 / sampleRate;
            double tone = 0.4 * Math.Sin(phase) * Math.Exp(-t * 30);
            double noise = 0.8 * (rng.NextDouble() * 2 - 1) * Math.Exp(-t * 24);
            samples[i] = (float)(tone + noise) * 0.8f;
        }
        return samples;
    }

    public static float[] Click(int sampleRate = 44100)
    {
        // Metronome tick: a short bright sine burst.
        int n = (int)(0.03 * sampleRate);
        var samples = new float[n];
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)sampleRate;
            samples[i] = (float)(Math.Sin(2 * Math.PI * 1400 * t) * Math.Exp(-t * 120)) * 0.8f;
        }
        return samples;
    }

    public static float[] Hat(int sampleRate = 44100)
    {
        int n = (int)(0.05 * sampleRate);
        var samples = new float[n];
        var rng = new Random(4);
        float previous = 0;
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)sampleRate;
            // Crude high-pass: differentiated noise.
            float white = (float)(rng.NextDouble() * 2 - 1);
            samples[i] = (float)((white - previous) * Math.Exp(-t * 80)) * 0.5f;
            previous = white;
        }
        return samples;
    }
}
