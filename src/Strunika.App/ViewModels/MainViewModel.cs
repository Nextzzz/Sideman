using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Strunika.Media;

namespace Strunika.App.ViewModels;

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

        // Model roster: base generalist (always kept for A/B comparison),
        // GuitarSet fine-tune for live/mic, Billboard fine-tune for mixes
        // (optional until trained).
        string? baseModel = FindModel("btc_large_voca.onnx");
        string? guitarModel = FindModel("btc_guitar.onnx") ?? baseModel;
        string? mixModel = FindModel("btc_mix.onnx");

        Tuner = new TunerViewModel(Capture);
        Live = new LiveChordsViewModel(Capture, guitarModel);
        Song = new SongViewModel(this, baseModel, guitarModel, mixModel);
        // Jam mode is shelved: the engine lives on in Services/JamEngine
        // until the scheduling bug is beaten on a simulation bench.
    }

    private static string? FindModel(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "models", fileName);
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
        Song.Dispose();
        Capture.Dispose();
    }
}
