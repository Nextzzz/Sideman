using System.Collections.ObjectModel;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using Sideman.Core.Analysis;
using Sideman.Core.Realtime;
using Sideman.Media;

namespace Sideman.App.ViewModels;

public partial class LiveChordsViewModel : ObservableObject
{
    private readonly StreamingChordDetector _detector;

    [ObservableProperty]
    private string currentChord = "—";

    [ObservableProperty]
    private string confidence = "";

    /// <summary>Noise-gate sensitivity in dB; bound to the UI slider.</summary>
    [ObservableProperty]
    private double gateMarginDb = 12.0;

    partial void OnGateMarginDbChanged(double value) => _detector.GateMarginDb = value;

    public ObservableCollection<string> History { get; } = new();

    public LiveChordsViewModel(MicrophoneCapture capture)
    {
        _detector = new StreamingChordDetector(MicrophoneCapture.SampleRate);
        capture.ChunkAvailable += chunk => _detector.AddSamples(chunk);
        _detector.ChordChanged += OnChordChanged;
    }

    private void OnChordChanged(Chord chord)
    {
        Application.Current?.Dispatcher.BeginInvoke(() =>
        {
            CurrentChord = chord.Label == "N" ? "—" : chord.Label;
            Confidence = $"впевненість {_detector.Confidence:P0}";
            if (chord != Chord.None)
            {
                History.Insert(0, $"{DateTime.Now:HH:mm:ss}   {chord.Label}");
                while (History.Count > 50)
                    History.RemoveAt(History.Count - 1);
            }
        });
    }
}
