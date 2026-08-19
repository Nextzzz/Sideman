namespace Strunika.Core.Dsp;

public static class Window
{
    /// <summary>Hann window: smooth taper that suppresses spectral leakage.</summary>
    public static float[] Hann(int size)
    {
        var w = new float[size];
        for (int i = 0; i < size; i++)
            w[i] = 0.5f * (1f - (float)Math.Cos(2.0 * Math.PI * i / size));
        return w;
    }
}
