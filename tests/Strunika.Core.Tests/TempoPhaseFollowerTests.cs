using Strunika.Core.Realtime;

namespace Strunika.Core.Tests;

[TestFixture]
public class TempoPhaseFollowerTests
{
    [Test]
    public void OnOnset_SteadyBeatsWithJitter_LocksNearTrueTempo()
    {
        // Arrange: 120 BPM strums with ±12 ms human jitter.
        var follower = new TempoPhaseFollower();
        var rng = new Random(11);

        // Act
        for (int i = 0; i < 24; i++)
            follower.OnOnset(1.0 + i * 0.5 + (rng.NextDouble() - 0.5) * 0.024);

        // Assert
        Assert.That(follower.Locked, Is.True);
        Assert.That(follower.Bpm, Is.EqualTo(120).Within(4));
    }

    [Test]
    public void BeatsBetween_AfterLock_PredictsOnGrid()
    {
        // Arrange: perfect 100 BPM (0.6 s period).
        var follower = new TempoPhaseFollower();
        for (int i = 0; i < 16; i++)
            follower.OnOnset(2.0 + i * 0.6);

        // Act: predictions for the next 2 seconds after the last onset.
        double last = 2.0 + 15 * 0.6;
        var beats = follower.BeatsBetween(last + 0.01, last + 2.0).ToArray();

        // Assert: each predicted beat sits on a multiple of the period.
        Assert.That(beats.Length, Is.GreaterThanOrEqualTo(3));
        foreach (var beat in beats)
        {
            double offGrid = Math.Abs((beat - 2.0) / 0.6
                - Math.Round((beat - 2.0) / 0.6)) * 0.6;
            Assert.That(offGrid, Is.LessThan(0.03), $"beat {beat:F3} off grid");
        }
    }

    [Test]
    public void OnOnset_GradualSpeedUp_TempoFollows()
    {
        // Arrange: 100 -> ~115 BPM over 30 beats.
        var follower = new TempoPhaseFollower();
        double time = 1.0;
        for (int i = 0; i < 16; i++)
        {
            follower.OnOnset(time);
            time += 0.6;
        }
        Assert.That(follower.Bpm, Is.EqualTo(100).Within(4));

        // Act: shrink the period toward 0.52 s.
        double period = 0.6;
        for (int i = 0; i < 30; i++)
        {
            period = Math.Max(0.52, period - 0.004);
            time += period;
            follower.OnOnset(time);
        }

        // Assert: the follower sped up with the player.
        Assert.That(follower.Bpm, Is.GreaterThan(108));
    }

    [Test]
    public void OnOnset_OffbeatEighths_DoNotDeraiTheGrid()
    {
        // Arrange: locked at 100 BPM, then eighth notes appear between beats.
        var follower = new TempoPhaseFollower();
        double time = 1.0;
        for (int i = 0; i < 16; i++)
        {
            follower.OnOnset(time);
            time += 0.6;
        }
        double bpmBefore = follower.Bpm;

        // Act: beats continue with off-beat eighths interleaved.
        for (int i = 0; i < 8; i++)
        {
            follower.OnOnset(time + 0.3); // eighth — off grid
            time += 0.6;
            follower.OnOnset(time);       // the real beat
        }

        // Assert
        Assert.That(follower.Locked, Is.True);
        Assert.That(follower.Bpm, Is.EqualTo(bpmBefore).Within(3));
    }
}
