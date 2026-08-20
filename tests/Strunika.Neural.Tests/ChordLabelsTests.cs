using NUnit.Framework;

namespace Strunika.Neural.Tests;

public class ChordLabelsTests
{
    [TestCase("Am7", "Am")]
    [TestCase("Cmaj7", "C")]
    [TestCase("G7", "G")]
    [TestCase("Dsus4", "D")]
    [TestCase("F#m", "F#m")]
    [TestCase("Bm7b5", "Bdim")]
    [TestCase("C#dim7", "C#dim")]
    [TestCase("Eaug", "Eaug")]
    [TestCase("AmMaj7", "Am")]
    [TestCase("E6", "E")]
    [TestCase("—", "—")]
    public void Simplify_collapses_extensions_to_triads(string pretty, string expected)
    {
        // Act
        string simplified = ChordLabels.Simplify(pretty);

        // Assert
        Assert.That(simplified, Is.EqualTo(expected));
    }

    [TestCase("C:maj", "C")]
    [TestCase("A:min7", "Am7")]
    [TestCase("F#:hdim7", "F#m7b5")]
    [TestCase("N", "—")]
    public void Pretty_formats_model_labels(string label, string expected)
    {
        // Act
        string pretty = ChordLabels.Pretty(label);

        // Assert
        Assert.That(pretty, Is.EqualTo(expected));
    }
}
