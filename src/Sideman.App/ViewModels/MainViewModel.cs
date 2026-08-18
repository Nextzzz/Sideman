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

        // Two domains, two models: the GuitarSet-fine-tuned one hears solo
        // guitar better (live, mic recordings); the original generalist is
        // safer on full mixes (files, YouTube).
        string? mixModel = FindModel("btc_large_voca.onnx");
        string? guitarModel = FindModel("btc_guitar.onnx") ?? mixModel;

        Tuner = new TunerViewModel(Capture);
        Live = new LiveChordsViewModel(Capture, guitarModel);
        Song = new SongViewModel(this, mixModel, guitarModel);
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
