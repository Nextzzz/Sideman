using NAudio.Wave;

namespace Sideman.Media;

/// <summary>
/// An endless silent stream into which one-shot samples are scheduled at
/// absolute sample positions — the band's sequencer track. Thread-safe:
/// the audio thread reads while the jam engine schedules ahead.
/// </summary>
public sealed class SampleScheduler : ISampleProvider
{
    private sealed record Event(long Start, float[] Data, float Gain);

    private readonly List<Event> _events = new();
    private readonly object _lock = new();
    private long _position;

    public WaveFormat WaveFormat { get; } =
        WaveFormat.CreateIeeeFloatWaveFormat(44100, 1);

    /// <summary>Samples handed to the sound card so far (the output clock).</summary>
    public long PositionSamples => Interlocked.Read(ref _position);

    public void Schedule(long startSample, float[] data, float gain)
    {
        if (startSample + data.Length <= PositionSamples)
            return; // entirely in the past
        lock (_lock)
            _events.Add(new Event(startSample, data, gain));
    }

    public void Clear()
    {
        lock (_lock)
            _events.Clear();
    }

    public int Read(float[] buffer, int offset, int count)
    {
        Array.Clear(buffer, offset, count);
        long start = PositionSamples;
        long end = start + count;

        lock (_lock)
        {
            for (int e = _events.Count - 1; e >= 0; e--)
            {
                var ev = _events[e];
                long evEnd = ev.Start + ev.Data.Length;
                if (evEnd <= start)
                {
                    _events.RemoveAt(e);
                    continue;
                }
                if (ev.Start >= end)
                    continue;

                long from = Math.Max(ev.Start, start);
                long to = Math.Min(evEnd, end);
                for (long s = from; s < to; s++)
                    buffer[offset + (int)(s - start)] +=
                        ev.Data[(int)(s - ev.Start)] * ev.Gain;
            }
        }

        Interlocked.Add(ref _position, count);
        return count;
    }
}
