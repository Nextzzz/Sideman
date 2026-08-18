using Sideman.Core.Analysis;
using Sideman.Core.Realtime;

namespace Sideman.Core.Tests;

[TestFixture]
public class StreamingChordDetectorTests
{
    private static void FeedInChunks(StreamingChordDetector detector, float[] samples, int chunkSize = 1024)
    {
        // Simulate a microphone: audio arrives in small buffers.
        for (int i = 0; i < samples.Length; i += chunkSize)
            detector.AddSamples(samples.AsSpan(i, Math.Min(chunkSize, samples.Length - i)));
    }

    [Test]
    public void AddSamples_StrummedC_SettlesOnC()
    {
        // Arrange
        var detector = new StreamingChordDetector(TestSignals.SampleRate);
        var samples = TestSignals.Strum(TestSignals.Voicings["C"], 2.0);

        // Act
        FeedInChunks(detector, samples);

        // Assert
        Assert.That(detector.CurrentChord.Label, Is.EqualTo("C"));
    }

    [Test]
    public void AddSamples_ChordChange_IsTrackedInOrder()
    {
        // Arrange
        var detector = new StreamingChordDetector(TestSignals.SampleRate);
        var history = new List<string>();
        detector.ChordChanged += chord => history.Add(chord.Label);
        var (samples, _) = TestSignals.Progression(new[] { "G", "C", "Em" }, 2.0);

        // Act
        FeedInChunks(detector, samples);

        // Assert: the three chords appear in playing order (transient
        // detections may add extras, but order must hold).
        var indexes = new[] { "G", "C", "Em" }.Select(l => history.IndexOf(l)).ToArray();
        Assert.That(indexes.All(i => i >= 0), Is.True,
            $"missing chords; history: {string.Join(",", history)}");
        Assert.That(indexes, Is.Ordered);
    }

    [Test]
    public void AddSamples_RoomNoiseOnly_ProducesNoChords()
    {
        // Arrange: five seconds of broadband noise — nobody is playing.
        var detector = new StreamingChordDetector(TestSignals.SampleRate);
        var fired = new List<string>();
        detector.ChordChanged += c => fired.Add(c.Label);
        var noise = new float[TestSignals.SampleRate * 5];
        TestSignals.AddNoise(noise, 0.05);

        // Act
        FeedInChunks(detector, noise);

        // Assert: the gate keeps every noise frame away from chord matching.
        Assert.That(detector.CurrentChord, Is.EqualTo(Chord.None));
        Assert.That(fired.Where(l => l != "N"), Is.Empty);
    }

    [Test]
    public void AddSamples_ChordAfterNoise_IsStillDetected()
    {
        // Arrange: room noise first, then a strummed chord over that noise.
        var detector = new StreamingChordDetector(TestSignals.SampleRate);
        var noise = new float[TestSignals.SampleRate * 3];
        TestSignals.AddNoise(noise, 0.02);
        var chord = TestSignals.Strum(TestSignals.Voicings["E"], 2.0);
        var mixed = new float[chord.Length];
        TestSignals.AddNoise(mixed, 0.02);
        for (int i = 0; i < chord.Length; i++)
            mixed[i] += chord[i];

        // Act
        FeedInChunks(detector, noise);
        FeedInChunks(detector, mixed);

        // Assert: playing punches through the gate.
        Assert.That(detector.CurrentChord.Label, Is.EqualTo("E"));
    }

    [Test]
    public void AddSamples_Silence_StaysOnNoChord()
    {
        // Arrange
        var detector = new StreamingChordDetector(TestSignals.SampleRate);
        var silence = new float[TestSignals.SampleRate];

        // Act
        FeedInChunks(detector, silence);

        // Assert
        Assert.That(detector.CurrentChord, Is.EqualTo(Chord.None));
    }
}
