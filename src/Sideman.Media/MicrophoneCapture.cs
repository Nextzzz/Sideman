using NAudio.Wave;

namespace Sideman.Media;

/// <summary>
/// Microphone input as a stream of float chunks at the analysis rate.
/// One instance = one open device; tuner, live chords and the recorder
/// all subscribe to the same stream.
/// </summary>
public sealed class MicrophoneCapture : IDisposable
{
    public const int SampleRate = 44100;

    private WaveInEvent? _waveIn;
    private readonly List<float> _recording = new();
    private bool _isRecording;

    public event Action<float[]>? ChunkAvailable;

    public bool IsRunning => _waveIn != null;

    public static IReadOnlyList<(int Index, string Name)> Devices()
    {
        var devices = new List<(int, string)>();
        for (int i = 0; i < WaveInEvent.DeviceCount; i++)
            devices.Add((i, WaveInEvent.GetCapabilities(i).ProductName));
        return devices;
    }

    public void Start(int deviceIndex = 0)
    {
        Stop();
        _waveIn = new WaveInEvent
        {
            DeviceNumber = deviceIndex,
            WaveFormat = new WaveFormat(SampleRate, 16, 1),
            BufferMilliseconds = 46, // ~one chroma hop
        };
        _waveIn.DataAvailable += OnData;
        _waveIn.StartRecording();
    }

    public void Stop()
    {
        if (_waveIn == null)
            return;
        _waveIn.DataAvailable -= OnData;
        _waveIn.StopRecording();
        _waveIn.Dispose();
        _waveIn = null;
    }

    /// <summary>Begin accumulating the incoming stream for later analysis.</summary>
    public void BeginRecording()
    {
        lock (_recording)
        {
            _recording.Clear();
            _isRecording = true;
        }
    }

    /// <summary>Stop accumulating and return everything captured.</summary>
    public float[] EndRecording()
    {
        lock (_recording)
        {
            _isRecording = false;
            var result = _recording.ToArray();
            _recording.Clear();
            return result;
        }
    }

    private void OnData(object? sender, WaveInEventArgs e)
    {
        int count = e.BytesRecorded / 2;
        var chunk = new float[count];
        for (int i = 0; i < count; i++)
            chunk[i] = BitConverter.ToInt16(e.Buffer, i * 2) / 32768f;

        if (_isRecording)
        {
            lock (_recording)
            {
                if (_isRecording)
                    _recording.AddRange(chunk);
            }
        }
        ChunkAvailable?.Invoke(chunk);
    }

    public void Dispose() => Stop();
}
