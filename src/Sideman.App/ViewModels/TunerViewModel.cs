using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using Sideman.Core.Analysis;
using Sideman.Media;

namespace Sideman.App.ViewModels;

public partial class TunerViewModel : ObservableObject
{
    private const int WindowSize = 4096;

    private readonly MicrophoneCapture _capture;
    private readonly PitchDetector _detector = new();
    private readonly float[] _buffer = new float[WindowSize * 2];
    private int _filled;
    private readonly object _lock = new();

    [ObservableProperty]
    private string noteName = "—";

    [ObservableProperty]
    private string details = "Увімкни мікрофон і зіграй ноту";

    /// <summary>Deviation in cents (-50..50) for the needle.</summary>
    [ObservableProperty]
    private double cents;

    [ObservableProperty]
    private bool inTune;

    public TunerViewModel(MicrophoneCapture capture)
    {
        _capture = capture;
        _capture.ChunkAvailable += OnChunk;

        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
        timer.Tick += (_, _) => Update();
        timer.Start();
    }

    private void OnChunk(float[] chunk)
    {
        lock (_lock)
        {
            int keep = Math.Max(0, _buffer.Length - chunk.Length);
            if (chunk.Length >= _buffer.Length)
            {
                Array.Copy(chunk, chunk.Length - _buffer.Length, _buffer, 0, _buffer.Length);
            }
            else
            {
                Array.Copy(_buffer, chunk.Length, _buffer, 0, keep);
                Array.Copy(chunk, 0, _buffer, keep, chunk.Length);
            }
            _filled = Math.Min(_buffer.Length, _filled + chunk.Length);
        }
    }

    private void Update()
    {
        if (!_capture.IsRunning || _filled < WindowSize)
            return;

        var window = new float[WindowSize];
        lock (_lock)
            Array.Copy(_buffer, _buffer.Length - WindowSize, window, 0, WindowSize);

        var pitch = _detector.Detect(window, MicrophoneCapture.SampleRate);
        if (pitch == null)
        {
            NoteName = "—";
            Details = "…";
            Cents = 0;
            InTune = false;
            return;
        }

        var (name, octave, cents) = Notes.Describe(pitch.Value.Frequency);
        NoteName = $"{name}{octave}";
        Cents = Math.Clamp(cents, -50, 50);
        InTune = Math.Abs(cents) < 5;
        Details = $"{pitch.Value.Frequency:F1} Гц   {cents:+0.0;-0.0} центів";
    }
}
