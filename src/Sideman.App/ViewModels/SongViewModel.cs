using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using Sideman.Core.Analysis;
using Sideman.Core.Diagnostics;
using Sideman.Media;
using Sideman.Neural;

namespace Sideman.App.ViewModels;

public sealed record SegmentRow(string Start, string End, string Chord);

/// <summary>
/// Song analysis: chords come from the neural recognizer (full vocabulary,
/// ~76% benchmark accuracy) when the model file is present, falling back
/// to the template engine otherwise. Tempo always comes from the DSP
/// rhythm pipeline.
/// </summary>
public partial class SongViewModel : ObservableObject
{
    private readonly MainViewModel _main;
    private readonly string? _neuralModelPath;
    private NeuralChordRecognizer? _neural;
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

    public SongViewModel(MainViewModel main, string? neuralModelPath)
    {
        _main = main;
        _neuralModelPath = neuralModelPath;
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
            FileLog.Info($"Analyze requested: {path}");
            if (path.StartsWith("http", StringComparison.OrdinalIgnoreCase))
            {
                Status = "Завантаження аудіо з YouTube…";
                path = await new YoutubeAudioService().DownloadAudioAsync(
                    path, Path.Combine(Path.GetTempPath(), "sideman"));
            }

            Status = "Аналіз…";
            var (rows, bpm, duration) = await Task.Run(() => AnalyzeFile(path));
            ShowRows(rows, Path.GetFileName(path), bpm, duration);
            FileLog.Info($"Analyze done: {rows.Count} segments, {bpm:F0} BPM");
        }
        catch (Exception ex)
        {
            FileLog.Error($"Analyze failed for '{Source}'", ex);
            Status = "Помилка: " + ex.Message + "   (повний стек — у лозі, кнопка «Лог»)";
        }
        finally
        {
            Busy = false;
        }
    }

    [RelayCommand]
    private void OpenLog()
    {
        try
        {
            if (File.Exists(FileLog.CurrentFile))
                Process.Start(new ProcessStartInfo(FileLog.CurrentFile) { UseShellExecute = true });
            else if (Directory.Exists(FileLog.Directory))
                Process.Start(new ProcessStartInfo(FileLog.Directory) { UseShellExecute = true });
            else
                Status = "Лог ще порожній.";
        }
        catch (Exception ex)
        {
            Status = "Не вдалось відкрити лог: " + ex.Message;
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
            var (rows, bpm, duration) = await Task.Run(() => AnalyzeSamples(samples));
            ShowRows(rows, Path.GetFileName(path), bpm, duration);
            Status = $"Готово. Збережено: {path}";
        }
        finally
        {
            Busy = false;
        }
    }

    private (List<SegmentRow> Rows, double Bpm, double Duration) AnalyzeFile(string path)
    {
        var (samples44, _) = AudioLoader.LoadMono(path);
        if (_neuralModelPath == null)
            return AnalyzeWithTemplates(samples44);

        var (samples22, _) = AudioLoader.LoadMono(path, CqtExtractor.SampleRate);
        return AnalyzeNeural(samples22, samples44);
    }

    private (List<SegmentRow> Rows, double Bpm, double Duration) AnalyzeSamples(float[] samples44)
    {
        if (_neuralModelPath == null)
            return AnalyzeWithTemplates(samples44);
        return AnalyzeNeural(HalfbandDecimator.Decimate(samples44), samples44);
    }

    private (List<SegmentRow>, double, double) AnalyzeNeural(float[] samples22, float[] samples44)
    {
        _neural ??= new NeuralChordRecognizer(_neuralModelPath!);
        var segments = _neural.Recognize(samples22);
        var rows = segments
            .Select(s => new SegmentRow(
                FormatTime(s.Start), FormatTime(s.End), ChordLabels.Pretty(s.Label)))
            .ToList();

        var onsets = new OnsetDetector();
        var novelty = onsets.NoveltyCurve(samples44, MicrophoneCapture.SampleRate);
        double bpm = new TempoEstimator().Estimate(
            novelty, onsets.FrameRate(MicrophoneCapture.SampleRate));

        return (rows, bpm, samples44.Length / (double)MicrophoneCapture.SampleRate);
    }

    private (List<SegmentRow>, double, double) AnalyzeWithTemplates(float[] samples44)
    {
        var analysis = new SongAnalyzer().Analyze(samples44, MicrophoneCapture.SampleRate);
        var rows = analysis.Chords
            .Select(s => new SegmentRow(
                FormatTime(s.Start), FormatTime(s.End),
                s.Chord.Label == "N" ? "—" : s.Chord.Label))
            .ToList();
        return (rows, analysis.Bpm, analysis.DurationSeconds);
    }

    private void ShowRows(List<SegmentRow> rows, string title, double bpm, double duration)
    {
        Segments.Clear();
        foreach (var row in rows)
            Segments.Add(row);
        string engine = _neuralModelPath != null ? "нейро" : "шаблони";
        Summary = $"{title}   •   {duration:F0} с   •   {bpm:F0} BPM   •   {engine}";
        Status = "Готово.";
    }

    private static string FormatTime(double seconds) =>
        $"{(int)seconds / 60}:{seconds % 60:00.0}";
}
