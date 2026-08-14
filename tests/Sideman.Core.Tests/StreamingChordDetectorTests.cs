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
