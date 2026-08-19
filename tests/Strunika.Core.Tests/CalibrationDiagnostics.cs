using Strunika.Core.Analysis;

namespace Strunika.Core.Tests;

/// <summary>Not assertions — a data dump used to calibrate recognizer constants.</summary>
[TestFixture, Explicit("Calibration tool, run by hand")]
public class CalibrationDiagnostics
{
    private static ChordEmissionModel Model => new();

    private static IEnumerable<(string Label, double Score)> Scores(ChromaFrame frame)
    {
        var model = Model;
        var dest = new double[model.StateCount];
        model.FillEmissions(frame, GateVerdict.Active, dest);
        return Enumerable.Range(0, model.StateCount)
            .Select(s => (model.ChordOf(s).Label, Math.Exp(dest[s] / model.EmissionSharpness)))
            .OrderByDescending(x => x.Item2);
    }

    [Test]
    public void DumpCampfireSegmentsAndFrames()
    {
        var (samples, _) = TestSignals.Progression(new[] { "G", "C", "D", "Em" }, 2.0);

        var segments = new ChordRecognizer().Recognize(samples, TestSignals.SampleRate);
        TestContext.Out.WriteLine("segments: " + string.Join(" | ", segments));

        var extractor = new ChromaExtractor();
        var frames = extractor.Extract(samples, TestSignals.SampleRate);
        for (int f = 0; f < frames.Length; f += 3)
        {
            double time = extractor.FrameTime(f, TestSignals.SampleRate);
            var top = Scores(frames[f]).First();
            TestContext.Out.WriteLine($"t={time:F2} e={frames[f].Energy:F0} {top.Label}:{top.Score:F3}");
        }
    }

    [Test]
    public void DumpStrummedChords()
    {
        foreach (var name in new[] { "G", "C", "D", "E", "A", "F", "Em", "Am", "Dm" })
        {
            var samples = TestSignals.Strum(TestSignals.Voicings[name], 2.0);
            var frames = new ChromaExtractor().Extract(samples, TestSignals.SampleRate);
            var frame = frames[frames.Length / 2];
            var top = Scores(frame).Take(3)
                .Select(x => $"{x.Label}:{x.Score:F3}");
            TestContext.Out.WriteLine($"{name} -> {string.Join(" ", top)}");
            if (name == "C")
            {
                TestContext.Out.WriteLine("  C chroma: " + string.Join(" ",
                    frame.Chroma.Select((v, i) => $"{Notes.Names[i]}={v:F2}")));
                TestContext.Out.WriteLine("  C bass:   " + string.Join(" ",
                    frame.Bass.Select((v, i) => v > 0 ? Notes.Names[i] : "").Where(s => s != "")));
            }
        }
    }

    [Test]
    public void DumpNoiseFrame()
    {
        var samples = new float[TestSignals.SampleRate];
        TestSignals.AddNoise(samples, 0.05);
        var frames = new ChromaExtractor().Extract(samples, TestSignals.SampleRate);
        var top = Scores(frames[frames.Length / 2]).First();
        TestContext.Out.WriteLine($"noise best: {top.Label} {top.Score:F4}");
    }
}
