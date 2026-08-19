using Sideman.Core.Analysis;

namespace Sideman.Core.Tests;

[TestFixture]
public class ChordTimelineTests
{
    private static readonly double[] Beats = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 };

    [Test]
    public void SnapToBeats_BoundaryNearBeat_MovesOntoBeat()
    {
        // Arrange: the chord change at 1.07 s is an analysis artifact —
        // the actual strum was on the beat at 1.0.
        var segments = new List<(double, double, string)>
        {
            (0.0, 1.07, "C"), (1.07, 3.0, "G"),
        };

        // Act
        var snapped = ChordTimeline.SnapToBeats(segments, Beats);

        // Assert
        Assert.That(snapped[0].End, Is.EqualTo(1.0));
        Assert.That(snapped[1].Start, Is.EqualTo(1.0));
    }

    [Test]
    public void SnapToBeats_BoundaryFarFromAnyBeat_StaysPut()
    {
        // Arrange: a change at 1.25 s sits exactly between beats.
        var segments = new List<(double, double, string)>
        {
            (0.0, 1.25, "C"), (1.25, 3.0, "G"),
        };

        // Act
        var snapped = ChordTimeline.SnapToBeats(segments, Beats);

        // Assert
        Assert.That(snapped[0].End, Is.EqualTo(1.25));
    }

    [Test]
    public void SnapToBeats_CollapsedBlip_IsRemovedAndNeighborsMerged()
    {
        // Arrange: a 60 ms "Em" blip straddling the beat collapses once
        // both its boundaries snap to the same beat.
        var segments = new List<(double, double, string)>
        {
            (0.0, 0.97, "C"), (0.97, 1.03, "Em"), (1.03, 3.0, "C"),
        };

        // Act
        var snapped = ChordTimeline.SnapToBeats(segments, Beats);

        // Assert: single continuous C.
        Assert.That(snapped, Has.Count.EqualTo(1));
        Assert.That(snapped[0], Is.EqualTo((0.0, 3.0, "C")));
    }
}
