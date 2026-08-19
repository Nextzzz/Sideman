using NUnit.Framework;
using Strunika.Media;

namespace Strunika.Neural.Tests;

[TestFixture]
public class SampleSchedulerTests
{
    [Test]
    public void Read_ScheduledEvent_LandsAtExactPosition()
    {
        // Arrange: a 4-sample blip scheduled at absolute position 10.
        var scheduler = new SampleScheduler();
        scheduler.Schedule(10, new float[] { 1, 1, 1, 1 }, gain: 0.5f);
        var buffer = new float[16];

        // Act
        scheduler.Read(buffer, 0, 16);

        // Assert
        for (int i = 0; i < 16; i++)
            Assert.That(buffer[i], Is.EqualTo(i is >= 10 and < 14 ? 0.5f : 0f),
                $"sample {i}");
    }

    [Test]
    public void Read_EventSpanningTwoReads_ContinuesSeamlessly()
    {
        // Arrange: event crosses the boundary between two Read calls.
        var scheduler = new SampleScheduler();
        scheduler.Schedule(6, new float[] { 1, 2, 3, 4 }, gain: 1f);
        var first = new float[8];
        var second = new float[8];

        // Act
        scheduler.Read(first, 0, 8);
        scheduler.Read(second, 0, 8);

        // Assert
        Assert.That(first[6], Is.EqualTo(1));
        Assert.That(first[7], Is.EqualTo(2));
        Assert.That(second[0], Is.EqualTo(3));
        Assert.That(second[1], Is.EqualTo(4));
        Assert.That(second[2], Is.EqualTo(0));
    }

    [Test]
    public void Read_OverlappingEvents_Mix()
    {
        // Arrange: two events overlapping at position 4.
        var scheduler = new SampleScheduler();
        scheduler.Schedule(3, new float[] { 1, 1 }, gain: 1f);
        scheduler.Schedule(4, new float[] { 1, 1 }, gain: 1f);
        var buffer = new float[8];

        // Act
        scheduler.Read(buffer, 0, 8);

        // Assert: 3 -> 1, 4 -> 1+1, 5 -> 1.
        Assert.That(buffer[3], Is.EqualTo(1));
        Assert.That(buffer[4], Is.EqualTo(2));
        Assert.That(buffer[5], Is.EqualTo(1));
    }

    [Test]
    public void Schedule_EntirelyInThePast_IsIgnored()
    {
        // Arrange: advance the clock past 8 samples first.
        var scheduler = new SampleScheduler();
        scheduler.Read(new float[8], 0, 8);
        scheduler.Schedule(2, new float[] { 1, 1 }, gain: 1f);
        var buffer = new float[8];

        // Act
        scheduler.Read(buffer, 0, 8);

        // Assert
        Assert.That(buffer.All(v => v == 0), Is.True);
    }
}
