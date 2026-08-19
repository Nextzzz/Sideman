using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sideman.App.Services;

namespace Sideman.App.ViewModels;

public partial class JamViewModel : ObservableObject, IDisposable
{
    private readonly MainViewModel _main;
    private readonly JamEngine _engine;

    [ObservableProperty]
    private string jamButtonText = "▶ Джемувати";

    [ObservableProperty]
    private string status = "Одягни навушники, натисни старт і грай рівним боєм";

    [ObservableProperty]
    private string bpmText = "—";

    [ObservableProperty]
    private string chordText = "—";

    [ObservableProperty]
    private double drumVolume = 0.8;

    [ObservableProperty]
    private double bassVolume = 0.7;

    [ObservableProperty]
    private double latencyMs = 120;

    [ObservableProperty]
    private bool metronomeOn;

    partial void OnDrumVolumeChanged(double value) => _engine.DrumGain = (float)value;
    partial void OnBassVolumeChanged(double value) => _engine.BassGain = (float)value;
    partial void OnLatencyMsChanged(double value) => _engine.LatencyOffsetMs = value;
    partial void OnMetronomeOnChanged(bool value) => _engine.MetronomeOn = value;

    public JamViewModel(MainViewModel main)
    {
        _main = main;
        _engine = new JamEngine(main.Capture);

        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
        timer.Tick += (_, _) => Refresh();
        timer.Start();
    }

    [RelayCommand]
    private void ToggleJam()
    {
        if (_engine.Running)
        {
            _engine.Stop();
            JamButtonText = "▶ Джемувати";
            Status = "Зупинено.";
        }
        else
        {
            _main.EnsureMicRunning();
            _engine.Start();
            JamButtonText = "■ Стоп";
        }
    }

    private void Refresh()
    {
        if (!_engine.Running)
            return;
        if (_engine.Locked)
        {
            Status = "Темп зафіксовано — грай! 🥁";
            BpmText = $"{_engine.Bpm:F0} BPM";
        }
        else
        {
            Status = "Слухаю темп — грай рівні удари…";
            BpmText = "—";
        }
        ChordText = _engine.CurrentChordLabel;
    }

    public void Dispose() => _engine.Dispose();
}
