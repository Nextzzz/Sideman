using Strunika.Core.Analysis;

namespace Strunika.Core.Tests;

[TestFixture]
public class ChromaTests
{
    [Test]
    public void Extract_PureA440_PutsEnergyInPitchClassA()
    {
        // Arrange
        var extractor = new ChromaExtractor();
        var samples = TestSignals.Sine(440.0, 1.0);

        // Act
        var frames = extractor.Extract(samples, TestSignals.SampleRate);

        // Assert: bin 9 (A) dominates the middle frame.
        var chroma = frames[frames.Length / 2].Chroma;
        int strongest = Array.IndexOf(chroma, chroma.Max());
        Assert.That(strongest, Is.EqualTo(9));
    }

    [Test]
    public void Extract_CMajorTriadSines_TopBinsAreChordTones()
    {
        // Arrange: C4 + E4 + G4 as pure tones.
        var c = TestSignals.Sine(261.63, 1.0, 0.3);
        var e = TestSignals.Sine(329.63, 1.0, 0.3);
        var g = TestSignals.Sine(392.00, 1.0, 0.3);
        var mix = new float[c.Length];
        for (int i = 0; i < mix.Length; i++)
            mix[i] = c[i] + e[i] + g[i];
        var extractor = new ChromaExtractor();

        // Act
        var frames = extractor.Extract(mix, TestSignals.SampleRate);
        var chroma = frames[frames.Length / 2].Chroma;

        // Assert: C(0), E(4), G(7) are the three strongest pitch classes.
        var top3 = chroma
            .Select((value, index) => (value, index))
            .OrderByDescending(x => x.value)
            .Take(3)
            .Select(x => x.index)
            .ToHashSet();
        Assert.That(top3, Is.EquivalentTo(new[] { 0, 4, 7 }));
    }

    [Test]
    public void Extract_LowEString_BassChromaPointsAtE()
    {
        // Arrange: open low E — the bass detector must name it.
        var samples = TestSignals.Pluck(82.41, 1.0);
        var extractor = new ChromaExtractor();

        // Act
        var frames = extractor.Extract(samples, TestSignals.SampleRate);
        var bass = frames[frames.Length / 3].Bass;

        // Assert: E (4) is the strongest bass pitch class.
        Assert.That(Array.IndexOf(bass, bass.Max()), Is.EqualTo(4));
    }

    [Test]
    public void Extract_Silence_HasZeroEnergy()
    {
        // Arrange
        var extractor = new ChromaExtractor();
        var silence = new float[TestSignals.SampleRate];

        // Act
        var frames = extractor.Extract(silence, TestSignals.SampleRate);

        // Assert
        Assert.That(frames.All(f => f.Energy < 1e-3), Is.True);
    }
}
