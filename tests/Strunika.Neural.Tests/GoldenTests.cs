using NUnit.Framework;
using Strunika.Media;
using Strunika.Neural;

namespace Strunika.Neural.Tests;

/// <summary>
/// Validates the C# port against golden files produced by the Python
/// reference pipeline (ml/make_goldens.py): same audio in, features and
/// chord decisions must match within tolerance.
/// </summary>
[TestFixture]
public class GoldenTests
{
    private static string RepoRoot
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "Strunika.slnx")))
                dir = dir.Parent;
            Assert.That(dir, Is.Not.Null, "repo root not found");
            return dir!.FullName;
        }
    }

    private static string Goldens => Path.Combine(RepoRoot, "ml", "goldens");
    private static string Models => Path.Combine(RepoRoot, "ml", "models");

    private static float[][] LoadCsv(string path) =>
        File.ReadAllLines(path)
            .Select(line => line.Split(',')
                .Select(v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray())
            .ToArray();

    private static float[] LoadAudio22k(string wavPath)
    {
        var (samples, _) = AudioLoader.LoadMono(wavPath, CqtExtractor.SampleRate);
        return samples;
    }

    [Test]
    public void Cqt_DemoProgression_MatchesPythonReference()
    {
        // Arrange
        var reference = LoadCsv(Path.Combine(Goldens, "demo_progression.features.csv"));
        var audio = LoadAudio22k(Path.Combine(RepoRoot, "output", "demo_progression.wav"));

        // Act
        var features = new CqtExtractor().Extract(audio);

        // Assert: same frame count; per-bin agreement within tolerance.
        Assert.That(features.Length, Is.EqualTo(reference.Length),
            "frame count mismatch");

        double sumAbs = 0, sumOffset = 0;
        int count = 0;
        for (int t = 0; t < reference.Length; t++)
        {
            for (int b = 0; b < CqtExtractor.Bins; b++)
            {
                double diff = features[t][b] - reference[t][b];
                sumOffset += diff;
                sumAbs += Math.Abs(diff);
                count++;
            }
        }
        double meanAbs = sumAbs / count;
        double meanOffset = sumOffset / count;
        TestContext.Out.WriteLine($"mean|diff|={meanAbs:F3} meanOffset={meanOffset:F3} (log domain)");

        // A direct CQT never matches librosa's recursive one to hundredths;
        // what must hold: no global offset (saturation!) and bounded noise.
        // Decision-level parity is enforced by the end-to-end tests below.
        Assert.That(Math.Abs(meanOffset), Is.LessThan(0.2),
            $"global scale drift: offset={meanOffset:F3}");
        Assert.That(meanAbs, Is.LessThan(1.0),
            $"log-CQT deviates too much from librosa: mean|diff|={meanAbs:F3}");
    }

    [Test]
    public void EndToEnd_DemoProgression_FrameLabelsMatchPython()
    {
        // Arrange
        var expected = File.ReadAllLines(Path.Combine(Goldens, "demo_progression.labels.txt"));
        var audio = LoadAudio22k(Path.Combine(RepoRoot, "output", "demo_progression.wav"));
        using var recognizer = new NeuralChordRecognizer(
            Path.Combine(Models, "btc_large_voca.onnx"));

        // Act: raw argmax — the golden files were produced without smoothing.
        var frames = recognizer.PredictFrames(audio, smooth: false);

        // Assert: high per-frame agreement with the Python pipeline
        // (small feature deviations may flip a few transient frames).
        int window = Math.Min(expected.Length, frames.Length);
        int agree = 0;
        for (int t = 0; t < window; t++)
            if (frames[t] == expected[t])
                agree++;
        double agreement = (double)agree / window;
        TestContext.Out.WriteLine($"frame agreement: {agreement:P1} ({agree}/{window})");

        Assert.That(agreement, Is.GreaterThan(0.85),
            $"only {agreement:P1} of frames match the Python reference");
    }

    [Test]
    public void EndToEnd_GuitarSetJazzClip_FrameLabelsMatchPython()
    {
        // Arrange: real guitar recording with 7th chords.
        var expected = File.ReadAllLines(
            Path.Combine(Goldens, "00_BN1-129-Eb_comp_mic.labels.txt"));
        var audio = LoadAudio22k(Path.Combine(
            RepoRoot, "datasets", "guitarset", "audio_mono-mic", "00_BN1-129-Eb_comp_mic.wav"));
        using var recognizer = new NeuralChordRecognizer(
            Path.Combine(Models, "btc_large_voca.onnx"));

        // Act: raw argmax — the golden files were produced without smoothing.
        var frames = recognizer.PredictFrames(audio, smooth: false);

        // Assert
        int window = Math.Min(expected.Length, frames.Length);
        int agree = 0;
        for (int t = 0; t < window; t++)
            if (frames[t] == expected[t])
                agree++;
        double agreement = (double)agree / window;
        TestContext.Out.WriteLine($"frame agreement: {agreement:P1} ({agree}/{window})");

        Assert.That(agreement, Is.GreaterThan(0.80),
            $"only {agreement:P1} of frames match the Python reference");
    }
}
