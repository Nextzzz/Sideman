using System.Collections.ObjectModel;
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

    public ObservableCollection<DeviceItem> Devices { get; } = new();

    [ObservableProperty]
    private DeviceItem? selectedDevice;

    [ObservableProperty]
    private bool micRunning;

    [ObservableProperty]
    private string micButtonText = "▶ Увімкнути мікрофон";

    public MainViewModel()
    {
        RefreshDevices();

        // Model roster: base generalist (always kept for A/B comparison)
        // and the mic-robust guitar fine-tune for live/mic/solo files.
        string? baseModel = FindModel("btc_large_voca.onnx");
        // guitar2 = consumer-mic-robust re-train; v1 kept only as a disk fallback.
        // Mix model retired after HookTheory-591 (no edge over base).
        string? guitarModel = FindModel("btc_guitar2.onnx")
                              ?? FindModel("btc_guitar.onnx") ?? baseModel;

        Tuner = new TunerViewModel(Capture);
        Live = new LiveChordsViewModel(Capture, guitarModel);
        Song = new SongViewModel(this, baseModel, guitarModel);
        // Jam mode is shelved: the engine lives on in Services/JamEngine
        // until the scheduling bug is beaten on a simulation bench.
    }

    /// <summary>Re-enumerate inputs — virtual mics (e.g. a phone) appear
    /// after the app has started, so the list must be refreshable.</summary>
    public void RefreshDevices()
    {
        var current = SelectedDevice?.Name;
        Devices.Clear();
        foreach (var (index, name) in MicrophoneCapture.Devices())
            Devices.Add(new DeviceItem(index, name));
        SelectedDevice = Devices.FirstOrDefault(d => d.Name == current)
                         ?? Devices.FirstOrDefault();
    }

    partial void OnSelectedDeviceChanged(DeviceItem? value)
    {
        // Switching device while capturing: restart on the new input.
        if (value != null && MicRunning)
            Capture.Start(value.Index);
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
