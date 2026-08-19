using NUnit.Framework;
using Strunika.Neural;

namespace Strunika.Neural.Tests;

[TestFixture]
public class DomainClassifierTests
{
    private const int Sr = 44100;

    private static float[] GuitarChord()
    {
        // Plucked-string synthesis: no energy below E2 (82 Hz).
        var samples = new float[Sr * 3];
        var rng = new Random(5);
        foreach (var freq in new[] { 98.0, 123.47, 146.83, 196.0, 246.94 })
        {
            int period = (int)Math.Round(Sr / freq);
            var buf = new float[period];
            for (int i = 0; i < period; i++)
                buf[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] += buf[i % period] / 5f;
                buf[i % period] = 0.996f * 0.5f * (buf[i % period] + buf[(i + 1) % period]);
            }
        }
        return samples;
    }

    [Test]
    public void IsGuitarLike_SoloGuitarChord_True()
    {
        // Arrange
        var samples = GuitarChord();

        // Act + Assert
        Assert.That(AudioDomainClassifier.IsGuitarLike(samples, Sr), Is.True,
            $"lowBand={AudioDomainClassifier.LowBandRatio(samples, Sr):F4}");
    }

    [Test]
    public void IsGuitarLike_GuitarPlusKickAndBass_False()
    {
        // Arrange: same guitar + a 50 Hz bass line and kick-like thumps.
        var samples = GuitarChord();
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] += (float)(0.25 * Math.Sin(2 * Math.PI * 50.0 * i / Sr));
            if (i % (Sr / 2) < Sr / 20) // 100 ms kick burst twice a second
                samples[i] += (float)(0.3 * Math.Sin(2 * Math.PI * 55.0 * i / Sr));
        }

        // Act + Assert
        Assert.That(AudioDomainClassifier.IsGuitarLike(samples, Sr), Is.False,
            $"lowBand={AudioDomainClassifier.LowBandRatio(samples, Sr):F4}");
    }
}
