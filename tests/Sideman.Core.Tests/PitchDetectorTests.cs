using Sideman.Core.Analysis;

namespace Sideman.Core.Tests;

[TestFixture]
public class PitchDetectorTests
{
    // All six standard-tuning open strings.
    [TestCase(82.41)]   // E2
    [TestCase(110.00)]  // A2
    [TestCase(146.83)]  // D3
    [TestCase(196.00)]  // G3
    [TestCase(246.94)]  // B3
    [TestCase(329.63)]  // E4
    public void Detect_PureSineOfOpenString_IsAccurateWithinTwoCents(double frequency)
    {
        // Arrange
        var detector = new PitchDetector();
        var samples = TestSignals.Sine(frequency, 0.2);

        // Act
        var result = detector.Detect(samples.AsSpan(0, 4096), TestSignals.SampleRate);

        // Assert
        Assert.That(result, Is.Not.Null);
        double cents = 1200 * Math.Log2(result!.Value.Frequency / frequency);
        Assert.That(Math.Abs(cents), Is.LessThan(2.0), $"detected {result.Value.Frequency:F2} Hz");
    }

    [Test]
    public void Detect_KarplusPluckE2_FindsFundamental()
    {
        // Arrange: synthetic pluck is harmonically rich, like a real string.
        var detector = new PitchDetector();
        var samples = TestSignals.Pluck(82.41, 1.0);

        // Act: analyze a window shortly after the attack noise settles.
        var result = detector.Detect(samples.AsSpan(4410, 4096), TestSignals.SampleRate);

        // Assert: within half a semitone (synthesis period is rounded to samples).
        Assert.That(result, Is.Not.Null);
        double cents = 1200 * Math.Log2(result!.Value.Frequency / 82.41);
        Assert.That(Math.Abs(cents), Is.LessThan(50.0), $"detected {result.Value.Frequency:F2} Hz");
    }

    [Test]
    public void Detect_Silence_ReturnsNull()
    {
        // Arrange
        var detector = new PitchDetector();
        var silence = new float[4096];

        // Act
        var result = detector.Detect(silence, TestSignals.SampleRate);

        // Assert
        Assert.That(result, Is.Null);
    }

    [Test]
    public void Detect_DetunedString_ReportsCentsOffset()
    {
        // Arrange: A2 tuned 20 cents flat, as a tuner must report.
        double detuned = 110.0 * Math.Pow(2, -20 / 1200.0);
        var detector = new PitchDetector();
        var samples = TestSignals.Sine(detuned, 0.2);

        // Act
        var result = detector.Detect(samples.AsSpan(0, 4096), TestSignals.SampleRate);
        var (name, _, cents) = Notes.Describe(result!.Value.Frequency);

        // Assert
        Assert.That(name, Is.EqualTo("A"));
        Assert.That(cents, Is.EqualTo(-20).Within(3));
    }
}
