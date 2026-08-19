using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using Strunika.Core.Analysis;
using Strunika.Core.Realtime;
using Strunika.Media;
using Strunika.Neural;

namespace Strunika.App.ViewModels;

/// <summary>
/// Two-tier live display: the DSP engine gives an instant provisional
/// guess (~0.3 s, grey), the neural sliding window confirms ~1 s later
/// (big and bright). Without a model file the DSP tier drives the big
/// display alone.
/// </summary>
public partial class LiveChordsViewModel : ObservableObject, IDisposable
{
    private readonly MicrophoneCapture _capture;
    private readonly StreamingChordDetector _dsp;
    private readonly SlidingNeuralChordDetector? _neural;
    private readonly System.Timers.Timer? _tickTimer;

    [ObservableProperty]
    private string provisionalChord = "";

    [ObservableProperty]
    private string confirmedChord = "—";

    [ObservableProperty]
    private bool neuralAvailable;

    /// <summary>Noise-gate sensitivity in dB; bound to the UI slider.</summary>
    [ObservableProperty]
    private double gateMarginDb = 12.0;

    partial void OnGateMarginDbChanged(double value) => _dsp.GateMarginDb = value;

    public ObservableCollection<string> History { get; } = new();

    public LiveChordsViewModel(MicrophoneCapture capture, string? neuralModelPath)
    {
        _capture = capture;
        _dsp = new StreamingChordDetector(MicrophoneCapture.SampleRate);
        _dsp.ChordChanged += OnProvisional;

        if (neuralModelPath != null && File.Exists(neuralModelPath))
        {
            _neural = new SlidingNeuralChordDetector(neuralModelPath);
            _neural.ConfirmedChanged += OnConfirmed;
            NeuralAvailable = true;

            _tickTimer = new System.Timers.Timer(250);
            _tickTimer.Elapsed += (_, _) =>
            {
                if (_capture.IsRunning)
                    _neural.Tick();
            };
            _tickTimer.Start();
        }

        capture.ChunkAvailable += chunk =>
        {
            _dsp.AddSamples(chunk);
            _neural?.AddSamples(chunk);
        };
    }

    private void OnProvisional(Chord chord)
    {
        Application.Current?.Dispatcher.BeginInvoke(() =>
        {
            string label = chord.Label == "N" ? "—" : chord.Label;
            if (NeuralAvailable)
            {
                ProvisionalChord = label;
            }
            else
            {
                // No model: the DSP tier is the whole display.
                ConfirmedChord = label;
                if (chord != Chord.None)
                    PushHistory(chord.Label);
            }
        });
    }

    private void OnConfirmed(string rawLabel)
    {
        Application.Current?.Dispatcher.BeginInvoke(() =>
        {
            string pretty = ChordLabels.Pretty(rawLabel);
            ConfirmedChord = pretty;
            if (pretty != "—")
                PushHistory(pretty);
        });
    }

    private string? _lastHistoryLabel;

    private void PushHistory(string label)
    {
        if (label == _lastHistoryLabel)
            return; // the same chord re-confirmed is not a new event
        _lastHistoryLabel = label;
        History.Insert(0, $"{DateTime.Now:HH:mm:ss}   {label}");
        while (History.Count > 50)
            History.RemoveAt(History.Count - 1);
    }

    public void Dispose()
    {
        _tickTimer?.Stop();
        _tickTimer?.Dispose();
        _neural?.Dispose();
    }
}
