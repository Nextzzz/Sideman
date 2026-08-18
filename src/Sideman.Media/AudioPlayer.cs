using NAudio.Wave;

namespace Sideman.Media;

/// <summary>Simple playback of an analyzed file (wav/mp3/m4a) so the user
/// can listen and verify the chord timeline against their ears.</summary>
public sealed class AudioPlayer : IDisposable
{
    private WaveOutEvent? _output;
    private AudioFileReader? _reader;

    public string? LoadedPath { get; private set; }

    public bool IsPlaying => _output?.PlaybackState == PlaybackState.Playing;

    public double DurationSeconds => _reader?.TotalTime.TotalSeconds ?? 0;

    public double PositionSeconds
    {
        get => _reader?.CurrentTime.TotalSeconds ?? 0;
        set
        {
            if (_reader != null)
                _reader.CurrentTime = TimeSpan.FromSeconds(
                    Math.Clamp(value, 0, DurationSeconds));
        }
    }

    public void Load(string path)
    {
        Stop();
        _reader = new AudioFileReader(path);
        _output = new WaveOutEvent();
        _output.Init(_reader);
        LoadedPath = path;
    }

    public void Play() => _output?.Play();

    public void Pause() => _output?.Pause();

    public void Stop()
    {
        _output?.Dispose();
        _reader?.Dispose();
        _output = null;
        _reader = null;
        LoadedPath = null;
    }

    public void Dispose() => Stop();
}
