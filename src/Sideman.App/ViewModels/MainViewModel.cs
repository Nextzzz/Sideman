using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sideman.Media;

namespace Sideman.App.ViewModels;

public sealed record DeviceItem(int Index, string Name)
{
    public override string ToString() => Name;
}

public partial class MainViewModel : ObservableObject, IDisposable
{
    public MicrophoneCapture Capture { get; } = new();

    public TunerViewModel Tuner { get; }
    public LiveChordsViewModel Live { get; }
    public SongViewModel Song { get; }

    public IReadOnlyList<DeviceItem> Devices { get; }

    [ObservableProperty]
    private DeviceItem? selectedDevice;

    [ObservableProperty]
    private bool micRunning;

    [ObservableProperty]
    private string micButtonText = "▶ Увімкнути мікрофон";

    public MainViewModel()
    {
        Devices = MicrophoneCapture.Devices()
            .Select(d => new DeviceItem(d.Index, d.Name))
            .ToList();
        SelectedDevice = Devices.FirstOrDefault();

        string? neuralModel = FindNeuralModel();
        Tuner = new TunerViewModel(Capture);
        Live = new LiveChordsViewModel(Capture, neuralModel);
        Song = new SongViewModel(this, neuralModel);
    }

    private static string? FindNeuralModel()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "models", "btc_large_voca.onnx");
        return File.Exists(path) ? path : null;
    }

    [RelayCommand]
    private void ToggleMic()
    {
        if (MicRunning)
        {
            Capture.Stop();
            MicRunning = false;
        }
        else
        {
            EnsureMicRunning();
        }
        MicButtonText = MicRunning ? "■ Вимкнути мікрофон" : "▶ Увімкнути мікрофон";
    }

    public void EnsureMicRunning()
    {
        if (Capture.IsRunning)
            return;
        Capture.Start(SelectedDevice?.Index ?? 0);
        MicRunning = true;
        MicButtonText = "■ Вимкнути мікрофон";
    }

    public void Dispose()
    {
        Live.Dispose();
        Capture.Dispose();
    }
}
