using Strunika.Core.Analysis;

namespace Strunika.Core.Tests;

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

    [TestCase(0.25)] // low F2 barely rings (bad barre pressure)
    [TestCase(0.0)]  // low F2 fully muted
    public void Recognize_BarreFWithWeakBassString_IsStillF(double bassGain)
    {
        // Arrange: barre F (133211) where the low F2 is weak or dead — the
        // hardest note of the hardest beginner chord. The lowest CLEAN
        // note is then C3, and a naive bass detector hands the chord to C.
        var voicing = TestSignals.Voicings["F"]; // F2 C3 F3 A3 C4 F4
        var samples = new float[(int)(2.0 * TestSignals.SampleRate)];
        int strumDelay = (int)(0.015 * TestSignals.SampleRate);
        for (int n = 0; n < voicing.Length; n++)
        {
            double freq = Notes.FrequencyFromMidi(voicing[n]);
            double gain = n == 0 ? bassGain : 1.0;
            var pluck = TestSignals.Pluck(freq, 1.8, seed: 40 + n);
            int offset = n * strumDelay;
            for (int i = 0; i < pluck.Length && offset + i < samples.Length; i++)
                samples[offset + i] += (float)(pluck[i] * gain / voicing.Length);
        }

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(new[] { "F" }));
    }

    [Test]
    public void Recognize_SmallFWithCInBass_IsStillF()
    {
        // Arrange: the common no-barre F (x33211): C3 F3 A3 C4 F4.
        // The bass note genuinely IS C — a legitimate F/C inversion.
        // The bass bonus must credit the fifth, not only the root.
        var samples = TestSignals.Strum(new[] { 48, 53, 57, 60, 65 }, 2.0, seed: 7);

        // Act
        var labels = RecognizedLabels(samples);

        // Assert
        Assert.That(labels, Is.EqualTo(new[] { "F" }));
    }

    [Test]
    public void Recognize_NoiseOnly_IsNoChordOnly()
    {
        // Arrange: three seconds of broadband room noise, no playing.
        var noise = new float[TestSignals.SampleRate * 3];
        TestSignals.AddNoise(noise, 0.05);

        // Act
        var segments = new ChordRecognizer().Recognize(noise, TestSignals.SampleRate);

        // Assert
        Assert.That(segments.All(s => s.Chord.Quality == ChordQuality.None), Is.True,
            "got: " + string.Join(", ", segments));
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
