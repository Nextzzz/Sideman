using Strunika.Core.Analysis;

namespace Strunika.Core.Tests;

[TestFixture]
public class RhythmTests
{
    /// <summary>Plucks at a steady tempo with two off-beat eighth notes.</summary>
    private static (float[] Samples, double[] BeatTruth) Groove(double bpm)
    {
        double period = 60.0 / bpm;
        var beats = Enumerable.Range(0, 7).Select(i => 0.5 + i * period).ToArray();
        var extras = new[] { beats[2] + period / 2, beats[4] + period / 2 };
        var onsets = beats.Concat(extras).OrderBy(t => t).ToArray();

        var samples = new float[(int)((onsets[^1] + 1.2) * TestSignals.SampleRate)];
        double[] freqs = { 82.41, 110.0, 146.83, 196.0 };
        for (int k = 0; k < onsets.Length; k++)
        {
            var pluck = TestSignals.Pluck(freqs[k % 4], 0.9, seed: k);
            int offset = (int)(onsets[k] * TestSignals.SampleRate);
            for (int i = 0; i < pluck.Length && offset + i < samples.Length; i++)
                samples[offset + i] += pluck[i];
        }
        return (samples, beats);
    }

    [Test]
    public void OnsetDetector_SixPlucks_FindsAllWithoutFalseAlarms()
    {
        // Arrange: plucks at known times.
        double[] truth = { 0.50, 1.10, 1.55, 2.20, 2.65, 3.40 };
        var samples = new float[(int)(4.7 * TestSignals.SampleRate)];
        for (int k = 0; k < truth.Length; k++)
        {
            var pluck = TestSignals.Pluck(110.0, 1.2, seed: k);
            int offset = (int)(truth[k] * TestSignals.SampleRate);
            for (int i = 0; i < pluck.Length && offset + i < samples.Length; i++)
                samples[offset + i] += pluck[i];
        }

        // Act
        var detected = new OnsetDetector().Detect(samples, TestSignals.SampleRate);

        // Assert: every pluck found within 50 ms, nothing extra.
        Assert.That(detected.Length, Is.EqualTo(truth.Length));
        for (int i = 0; i < truth.Length; i++)
            Assert.That(Math.Abs(detected[i] - truth[i]), Is.LessThan(0.05),
                $"onset {i}: expected {truth[i]:F2}, got {detected[i]:F2}");
    }

    [TestCase(90.0)]
    [TestCase(120.0)]
    public void TempoEstimator_SteadyGroove_FindsBpmWithinFivePercent(double bpm)
    {
        // Arrange
        var (samples, _) = Groove(bpm);
        var onsets = new OnsetDetector();

        // Act
        var novelty = onsets.NoveltyCurve(samples, TestSignals.SampleRate);
        double estimated = new TempoEstimator().Estimate(novelty, onsets.FrameRate(TestSignals.SampleRate));

        // Assert
        Assert.That(estimated, Is.EqualTo(bpm).Within(bpm * 0.05));
    }

    [Test]
    public void BeatTracker_GrooveWithEighthNotes_BeatsLandOnQuartersOnly()
    {
        // Arrange
        var (samples, beatTruth) = Groove(100.0);
        var onsets = new OnsetDetector();
        var novelty = onsets.NoveltyCurve(samples, TestSignals.SampleRate);
        double frameRate = onsets.FrameRate(TestSignals.SampleRate);
        double bpm = new TempoEstimator().Estimate(novelty, frameRate);

        // Act
        var frames = new BeatTracker().Track(novelty, frameRate, bpm);
        var beatTimes = frames
            .Select(f => (f * (double)onsets.Hop + onsets.NFft / 2.0) / TestSignals.SampleRate)
            .ToArray();

        // Assert: one detected beat near every true quarter note.
        Assert.That(beatTimes.Length, Is.EqualTo(beatTruth.Length));
        for (int i = 0; i < beatTruth.Length; i++)
            Assert.That(Math.Abs(beatTimes[i] - beatTruth[i]), Is.LessThan(0.07),
                $"beat {i}: expected {beatTruth[i]:F2}, got {beatTimes[i]:F2}");
    }
}
