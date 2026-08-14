using Sideman.Core.Analysis;

namespace Sideman.Core.Tests;

[TestFixture]
public class ChordRecognizerTests
{
    private static string[] RecognizedLabels(float[] samples)
    {
        var recognizer = new ChordRecognizer();
        var segments = recognizer.Recognize(samples, TestSignals.SampleRate);
        // Ignore "no chord" gaps (lead-in, decay tails) — compare the music.
        return segments
            .Where(s => s.Chord.Quality != ChordQuality.None)
            .Select(s => s.Chord.Label)
            .ToArray();
    }

    [Test]
    public void Recognize_SingleStrummedGMajor_IsG()
    {
        // Arrange
        var samples = TestSignals.Strum(TestSignals.Voicings["G"], 2.0);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(new[] { "G" }));
    }

    [Test]
    public void Recognize_SingleStrummedAMinor_IsAm()
    {
        // Arrange
        var samples = TestSignals.Strum(TestSignals.Voicings["Am"], 2.0);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(new[] { "Am" }));
    }

    [Test]
    public void Recognize_CampfireProgression_AllChordsInOrder()
    {
        // Arrange: G - C - D - Em, two seconds each.
        var (samples, truth) = TestSignals.Progression(
            new[] { "G", "C", "D", "Em" }, secondsPerChord: 2.0);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(truth.Select(t => t.Label).ToArray()));
    }

    [Test]
    public void Recognize_MinorProgression_AllChordsInOrder()
    {
        // Arrange: Am - F - C - E, the "house of the rising sun" family.
        var (samples, truth) = TestSignals.Progression(
            new[] { "Am", "F", "C", "E" }, secondsPerChord: 2.0);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(truth.Select(t => t.Label).ToArray()));
    }

    [Test]
    public void Recognize_ProgressionBoundaries_AreCloseToTruth()
    {
        // Arrange
        var (samples, truth) = TestSignals.Progression(
            new[] { "G", "C", "D", "Em" }, secondsPerChord: 2.0);
        var recognizer = new ChordRecognizer();

        // Act
        var segments = recognizer
            .Recognize(samples, TestSignals.SampleRate)
            .Where(s => s.Chord.Quality != ChordQuality.None)
            .ToArray();

        // Assert: each chord's segment starts within 0.35 s of the strum.
        for (int i = 0; i < truth.Length; i++)
        {
            double deviation = Math.Abs(segments[i].Start - truth[i].Start);
            // First segment may start at 0 because lead-in is merged.
            if (i > 0)
                Assert.That(deviation, Is.LessThan(0.35),
                    $"{truth[i].Label} expected near {truth[i].Start:F2}s, got {segments[i].Start:F2}s");
        }
    }

    [Test]
    public void Recognize_ProgressionWithNoise_StillCorrect()
    {
        // Arrange: same progression drowned in -26 dB white noise.
        var (samples, truth) = TestSignals.Progression(
            new[] { "G", "C", "D", "Em" }, secondsPerChord: 2.0);
        TestSignals.AddNoise(samples, 0.05);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(truth.Select(t => t.Label).ToArray()));
    }

    [Test]
    public void Recognize_Silence_IsNoChordOnly()
    {
        // Arrange
        var silence = new float[TestSignals.SampleRate * 2];

        // Act
        var recognizer = new ChordRecognizer();
        var segments = recognizer.Recognize(silence, TestSignals.SampleRate);

        // Assert
        Assert.That(segments.All(s => s.Chord.Quality == ChordQuality.None), Is.True);
    }
}
