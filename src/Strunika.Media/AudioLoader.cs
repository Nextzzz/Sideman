using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Strunika.Media;

/// <summary>
/// Decodes an audio file (wav/mp3/m4a/anything MediaFoundation knows)
/// into mono float samples at the analysis rate.
/// </summary>
public static class AudioLoader
{
    public const int TargetSampleRate = 44100;

    public static (float[] Samples, int SampleRate) LoadMono(string path, int targetRate = TargetSampleRate)
    {
        using var reader = new AudioFileReader(path);
        ISampleProvider provider = reader;

        if (provider.WaveFormat.Channels == 2)
            provider = new StereoToMonoSampleProvider(provider) { LeftVolume = 0.5f, RightVolume = 0.5f };
        else if (provider.WaveFormat.Channels > 2)
            throw new NotSupportedException($"{provider.WaveFormat.Channels}-channel audio is not supported.");

        if (provider.WaveFormat.SampleRate != targetRate)
            provider = new WdlResamplingSampleProvider(provider, targetRate);

        var chunks = new List<float[]>();
        var buffer = new float[targetRate]; // 1 second per read
        int total = 0;
        int read;
        while ((read = provider.Read(buffer, 0, buffer.Length)) > 0)
        {
            var chunk = new float[read];
            Array.Copy(buffer, chunk, read);
            chunks.Add(chunk);
            total += read;
        }

        var samples = new float[total];
        int offset = 0;
        foreach (var chunk in chunks)
        {
            chunk.CopyTo(samples, offset);
            offset += chunk.Length;
        }
        return (samples, targetRate);
    }

    public static void SaveWav(string path, float[] samples, int sampleRate)
    {
        using var writer = new WaveFileWriter(path, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1));
        writer.WriteSamples(samples, 0, samples.Length);
    }
}
