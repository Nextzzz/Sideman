using Sideman.Core.Dsp;

namespace Sideman.Core.Tests;

[TestFixture]
public class FftTests
{
    [Test]
    public void Forward_ImpulseInput_GivesFlatSpectrum()
    {
        // Arrange: a unit impulse contains every frequency equally.
        var re = new double[64];
        var im = new double[64];
        re[0] = 1.0;

        // Act
        Fft.Forward(re, im);

        // Assert: every bin has magnitude 1.
        for (int k = 0; k < 64; k++)
        {
            double magnitude = Math.Sqrt(re[k] * re[k] + im[k] * im[k]);
            Assert.That(magnitude, Is.EqualTo(1.0).Within(1e-9), $"bin {k}");
        }
    }

    [Test]
    public void Forward_PureSine_PeaksAtItsBin()
    {
        // Arrange: sine at exactly bin 5 of a 256-point FFT.
        int n = 256, bin = 5;
        var re = new double[n];
        var im = new double[n];
        for (int i = 0; i < n; i++)
            re[i] = Math.Sin(2 * Math.PI * bin * i / n);

        // Act
        Fft.Forward(re, im);

        // Assert: energy sits in bin 5 (and its mirror), nowhere else.
        double peak = Math.Sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
        Assert.That(peak, Is.EqualTo(n / 2.0).Within(1e-6));
        for (int k = 0; k < n / 2; k++)
        {
            if (k == bin)
                continue;
            double magnitude = Math.Sqrt(re[k] * re[k] + im[k] * im[k]);
            Assert.That(magnitude, Is.LessThan(1e-6), $"bin {k} should be empty");
        }
    }

    [Test]
    public void Forward_NonPowerOfTwo_Throws()
    {
        // Arrange
        var re = new double[100];
        var im = new double[100];

        // Act + Assert
        Assert.Throws<ArgumentException>(() => Fft.Forward(re, im));
    }
}
