using NUnit.Framework;
using Sideman.Core.Analysis;
using Sideman.Media;
using Sideman.Neural;

namespace Sideman.Neural.Tests;

[TestFixture]
public class SlidingDetectorTests
{
    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "Sideman.slnx")))
                dir = dir.Parent;
            return dir!.FullName;
        }
    }

    [Test]
    public void Recognize_CampfireProgressionWithKeyPrior_StaysCorrect()
    {
        // Arrange: G-C-D-Em are all diatonic to G major — the key prior
        // must reinforce, never distort, an already-correct timeline.
        var (samples, _) = AudioLoader.LoadMono(
            Path.Combine(RepoRoot, "output", "demo_progression.wav"), 22050);
        using var recognizer = new NeuralChordRecognizer(
            Path.Combine(RepoRoot, "ml", "models", "btc_large_voca.onnx"));

        // Act
        var labels = recognizer.Recognize(samples)
            .Where(s => s.Label is not ("N" or "X"))
            .Select(s => ChordLabels.Pretty(s.Label))
            .ToArray();

        // Assert
        Assert.That(labels, Is.EqualTo(new[] { "G", "C", "D", "Em" }));
        Assert.That(recognizer.DetectedKey, Is.EqualTo("G"));
    }

    [Test]
    public void Decimator_Sine440At44100_StaysAt440After2xDecimation()
    {
        // Arrange: a pure A4 at the capture rate.
        var input = new float[44100];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(0.5 * Math.Sin(2 * Math.PI * 440.0 * i / 44100.0));

        // Act
        var output = HalfbandDecimator.Decimate(input);

        // Assert: same pitch at 22050 Hz, half the samples.
        Assert.That(output.Length, Is.EqualTo(input.Length / 2));
        var pitch = new PitchDetector().Detect(output.AsSpan(2048, 4096), 22050);
        Assert.That(pitch, Is.Not.Null);
        Assert.That(pitch!.Value.Frequency, Is.EqualTo(440.0).Within(1.0));
    }

    [Test]
    public void Tick_DemoProgressionStreamed_ConfirmsChordsInOrder()
    {
        // Arrange: the demo G-C-D-Em progression streamed like a mic feed.
        var (samples, _) = AudioLoader.LoadMono(
            Path.Combine(RepoRoot, "output", "demo_progression.wav"), 44100);
        using var detector = new SlidingNeuralChordDetector(
            Path.Combine(RepoRoot, "ml", "models", "btc_large_voca.onnx"));
        var confirmed = new List<string>();
        detector.ConfirmedChanged += label => confirmed.Add(ChordLabels.Pretty(label));

        // Act: 250 ms chunks, a Tick after each — real-time cadence.
        int chunk = 11025;
        for (int offset = 0; offset < samples.Length; offset += chunk)
        {
            detector.AddSamples(samples.AsSpan(offset, Math.Min(chunk, samples.Length - offset)));
            detector.Tick();
        }

        // Assert: all four chords confirmed, in playing order, and the
        // stability gate let NOTHING else through.
        var indexes = new[] { "G", "C", "D", "Em" }
            .Select(l => confirmed.IndexOf(l))
            .ToArray();
        Assert.That(indexes.All(i => i >= 0), Is.True,
            "missing chords; confirmed: " + string.Join(",", confirmed));
        Assert.That(indexes, Is.Ordered, string.Join(",", confirmed));
        Assert.That(confirmed.Where(l => l is not ("G" or "C" or "D" or "Em" or "—")), Is.Empty,
            "spurious confirmations: " + string.Join(",", confirmed));
    }
}
