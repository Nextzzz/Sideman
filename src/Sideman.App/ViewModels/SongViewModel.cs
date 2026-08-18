using System.Collections.ObjectModel;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using Sideman.Core.Analysis;
using Sideman.Media;

namespace Sideman.App.ViewModels;

public sealed record SegmentRow(string Start, string End, string Chord, string Confidence);

public partial class SongViewModel : ObservableObject
{
    private readonly MainViewModel _main;
    private bool _recording;

    [ObservableProperty]
    private string source = "";

    [ObservableProperty]
    private bool busy;

    [ObservableProperty]
    private string status = "Відкрий файл, встав YouTube-посилання або запиши гру";

    [ObservableProperty]
    private string summary = "";

    [ObservableProperty]
    private string recordButtonText = "● Записати";

    public ObservableCollection<SegmentRow> Segments { get; } = new();

    public SongViewModel(MainViewModel main)
    {
        _main = main;
    }

    [RelayCommand]
    private void Browse()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "Аудіо|*.wav;*.mp3;*.m4a;*.aac;*.wma|Всі файли|*.*",
        };
        if (dialog.ShowDialog() == true)
            Source = dialog.FileName;
    }

    [RelayCommand]
    private async Task AnalyzeAsync()
    {
        if (string.IsNullOrWhiteSpace(Source))
        {
            Status = "Вкажи файл або посилання.";
            return;
        }

        Busy = true;
        try
        {
            string path = Source.Trim();
            if (path.StartsWith("http", StringComparison.OrdinalIgnoreCase))
            {
                Status = "Завантаження аудіо з YouTube…";
                path = await new YoutubeAudioService().DownloadAudioAsync(
                    path, Path.Combine(Path.GetTempPath(), "sideman"));
            }

            Status = "Аналіз…";
            var result = await Task.Run(() =>
            {
                var (samples, sampleRate) = AudioLoader.LoadMono(path);
                return new SongAnalyzer().Analyze(samples, sampleRate);
            });
            ShowAnalysis(result, Path.GetFileName(path));
        }
        catch (Exception ex)
        {
            Status = "Помилка: " + ex.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    [RelayCommand]
    private async Task ToggleRecordAsync()
    {
        if (!_recording)
        {
            _main.EnsureMicRunning();
            _main.Capture.BeginRecording();
            _recording = true;
            RecordButtonText = "■ Стоп і аналіз";
            Status = "Запис… грай!";
            return;
        }

        var samples = _main.Capture.EndRecording();
        _recording = false;
        RecordButtonText = "● Записати";

        if (samples.Length < MicrophoneCapture.SampleRate)
        {
            Status = "Запис закороткий.";
            return;
        }

        Busy = true;
        try
        {
            // Keep every take on disk — real recordings from the user's
            // own mic are the most valuable calibration material there is.
            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Sideman");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, $"take_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
            AudioLoader.SaveWav(path, samples, MicrophoneCapture.SampleRate);

            Status = "Аналіз запису…";
            var result = await Task.Run(() =>
                new SongAnalyzer().Analyze(samples, MicrophoneCapture.SampleRate));
            ShowAnalysis(result, Path.GetFileName(path));
            Status = $"Готово. Збережено: {path}";
        }
        finally
        {
            Busy = false;
        }
    }

    private void ShowAnalysis(SongAnalysis analysis, string title)
    {
        Segments.Clear();
        foreach (var segment in analysis.Chords)
        {
            Segments.Add(new SegmentRow(
                FormatTime(segment.Start),
                FormatTime(segment.End),
                segment.Chord.Label == "N" ? "—" : segment.Chord.Label,
                segment.Confidence.ToString("F2")));
        }
        Summary = $"{title}   •   {analysis.DurationSeconds:F0} с   •   {analysis.Bpm:F0} BPM";
        Status = "Готово.";
    }

    private static string FormatTime(double seconds) =>
        $"{(int)seconds / 60}:{seconds % 60:00.0}";
}
