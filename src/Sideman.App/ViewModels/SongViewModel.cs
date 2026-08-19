using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using Sideman.Core.Analysis;
using Sideman.Core.Diagnostics;
using Sideman.Media;
using Sideman.Neural;

namespace Sideman.App.ViewModels;

/// <summary>One chord segment row; IsCurrent lights up during playback.</summary>
public partial class SegmentRowVm : ObservableObject
{
    public required string Start { get; init; }
    public required string End { get; init; }
    public required string Chord { get; init; }
    public double StartSec { get; init; }
    public double EndSec { get; init; }

    [ObservableProperty]
    private bool isCurrent;
}

/// <summary>
/// Song analysis: neural chords (full vocabulary) + DSP tempo, plus a
/// built-in player with a chord timeline that follows the sound — listen
/// and verify which chords the model got wrong.
/// </summary>
public partial class SongViewModel : ObservableObject, IDisposable
{
    private readonly MainViewModel _main;
    private readonly string? _baseModelPath;   // original generalist (kept for A/B)
    private readonly string? _guitarModelPath; // GuitarSet fine-tune (mic/solo)
    private readonly string? _mixModelPath;    // Billboard fine-tune (full mixes)
    private readonly Dictionary<string, NeuralChordRecognizer> _recognizers = new();
    private bool _recording;

    private readonly AudioPlayer _player = new();
    private string? _audioPath;
    private bool _syncingPosition;

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

    /// <summary>Which model analyzes. Авто routes by domain; the explicit
    /// options exist for A/B comparison of the same song across models.</summary>
    public string[] EngineModes { get; } = { "Авто", "Гітара", "Мікс", "Базова" };

    [ObservableProperty]
    private string engineMode = "Авто";

    [ObservableProperty]
    private bool playerAvailable;

    [ObservableProperty]
    private string playButtonText = "▶";

    [ObservableProperty]
    private double positionSeconds;

    [ObservableProperty]
    private double durationSeconds;

    [ObservableProperty]
    private string timeText = "";

    /// <summary>The chord sounding right now during playback.</summary>
    [ObservableProperty]
    private string nowChord = "";

    [ObservableProperty]
    private SegmentRowVm? selectedRow;

    public ObservableCollection<SegmentRowVm> Segments { get; } = new();

    public SongViewModel(
        MainViewModel main, string? baseModelPath, string? guitarModelPath, string? mixModelPath)
    {
        _main = main;
        _baseModelPath = baseModelPath;
        _guitarModelPath = guitarModelPath;
        _mixModelPath = mixModelPath;

        var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
        timer.Tick += (_, _) => SyncPlayback();
        timer.Start();
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
            var (samples44, _) = await Task.Run(() => AudioLoader.LoadMono(path));
            var result = await Task.Run(() => Analyze(path, samples44, micRecording: false));
            ShowAnalysis(result, path, Source.Trim());
            FileLog.Info($"Analyze done: {result.Segments.Count} segments, {result.Bpm:F0} BPM");
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
            FileLog.Info($"Recording saved: {path}");

            Status = "Аналіз запису…";
            var result = await Task.Run(() => Analyze(path, samples, micRecording: true));
            ShowAnalysis(result, path, "запис із мікрофона");
            Status = $"Готово. Збережено: {path}";
        }
        catch (Exception ex)
        {
            FileLog.Error("Recording analysis failed", ex);
            Status = "Помилка: " + ex.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    [RelayCommand]
    private void PlayPause()
    {
        if (_audioPath == null)
            return;
        try
        {
            if (_player.LoadedPath != _audioPath)
            {
                _player.Load(_audioPath);
                DurationSeconds = _player.DurationSeconds;
            }
            if (_player.IsPlaying)
            {
                _player.Pause();
                PlayButtonText = "▶";
            }
            else
            {
                _player.Play();
                PlayButtonText = "⏸";
            }
        }
        catch (Exception ex)
        {
            FileLog.Error("Playback failed", ex);
            Status = "Не вдалось відтворити: " + ex.Message;
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

    partial void OnPositionSecondsChanged(double value)
    {
        if (!_syncingPosition && _player.LoadedPath != null)
            _player.PositionSeconds = value;
    }

    private void SyncPlayback()
    {
        if (_player.LoadedPath == null)
            return;

        if (!_player.IsPlaying && PlayButtonText == "⏸")
            PlayButtonText = "▶"; // reached the end

        _syncingPosition = true;
        double position = _player.PositionSeconds;
        PositionSeconds = position;
        TimeText = $"{FormatClock(position)} / {FormatClock(DurationSeconds)}";
        _syncingPosition = false;

        var current = Segments.FirstOrDefault(
            s => position >= s.StartSec && position < s.EndSec);
        if (current != SelectedRow)
        {
            if (SelectedRow != null)
                SelectedRow.IsCurrent = false;
            if (current != null)
            {
                current.IsCurrent = true;
                NowChord = current.Chord;
            }
            SelectedRow = current;
        }
    }

    private sealed record AnalysisResult(
        List<(double Start, double End, string Chord)> Segments,
        double Bpm, double Duration, string Engine);

    private AnalysisResult Analyze(string audioPath, float[] samples44, bool micRecording)
    {
        double duration = samples44.Length / (double)MicrophoneCapture.SampleRate;

        // Model choice: Авто routes by domain (mic takes and bass-free
        // files -> guitar model; everything else -> mix model when trained,
        // base otherwise). Explicit modes exist for A/B comparison — the
        // base generalist is deliberately kept available forever.
        bool autoGuitar = micRecording || AudioDomainClassifier.IsGuitarLike(
            samples44, MicrophoneCapture.SampleRate);
        (string? modelPath, string engineName) = EngineMode switch
        {
            "Гітара" => (_guitarModelPath, "нейро · гітарна"),
            "Базова" => (_baseModelPath, "нейро · базова"),
            "Мікс" => _mixModelPath != null
                ? (_mixModelPath, "нейро · мікс (Billboard)")
                : (_baseModelPath, "нейро · базова (мікс ще не натреновано)"),
            _ => autoGuitar
                ? (_guitarModelPath ?? _baseModelPath, "нейро · гітарна (авто)")
                : _mixModelPath != null
                    ? (_mixModelPath, "нейро · мікс (авто)")
                    : (_baseModelPath, "нейро · базова (авто)"),
        };
        if (EngineMode == "Авто" && !micRecording)
            FileLog.Info($"Auto domain probe: lowBand=" +
                $"{AudioDomainClassifier.LowBandRatio(samples44, MicrophoneCapture.SampleRate):F4}" +
                $" -> {engineName}");

        if (modelPath != null)
        {
            if (!_recognizers.TryGetValue(modelPath, out var neural))
                _recognizers[modelPath] = neural = new NeuralChordRecognizer(modelPath);
            var samples22 = audioPath.EndsWith(".wav", StringComparison.OrdinalIgnoreCase)
                            || audioPath.EndsWith(".m4a", StringComparison.OrdinalIgnoreCase)
                            || audioPath.EndsWith(".mp3", StringComparison.OrdinalIgnoreCase)
                ? AudioLoader.LoadMono(audioPath, CqtExtractor.SampleRate).Samples
                : HalfbandDecimator.Decimate(samples44);
            var segments = neural.Recognize(samples22)
                .Select(s => (s.Start, s.End, ChordLabels.Pretty(s.Label)))
                .ToList();

            var onsets = new OnsetDetector();
            var novelty = onsets.NoveltyCurve(samples44, MicrophoneCapture.SampleRate);
            double bpm = new TempoEstimator().Estimate(
                novelty, onsets.FrameRate(MicrophoneCapture.SampleRate));
            return new AnalysisResult(segments, bpm, duration, engineName);
        }

        var analysis = new SongAnalyzer().Analyze(samples44, MicrophoneCapture.SampleRate);
        var rows = analysis.Chords
            .Select(s => (s.Start, s.End,
                s.Chord.Label == "N" ? "—" : s.Chord.Label))
            .ToList();
        return new AnalysisResult(rows, analysis.Bpm, duration, "шаблони");
    }

    private void ShowAnalysis(AnalysisResult result, string audioPath, string sourceDescription)
    {
        Services.AnalysisStore.Save(new Services.SavedAnalysis(
            sourceDescription,
            audioPath,
            DateTime.Now,
            result.Duration,
            result.Bpm,
            result.Engine,
            result.Segments.Select(s => new Services.SavedSegment(s.Start, s.End, s.Chord)).ToList()));

        _player.Stop();
        PlayButtonText = "▶";
        NowChord = "";
        _audioPath = audioPath;
        PlayerAvailable = true;
        DurationSeconds = result.Duration;
        PositionSeconds = 0;
        TimeText = $"0:00 / {FormatClock(result.Duration)}";

        Segments.Clear();
        foreach (var (start, end, chord) in result.Segments)
        {
            Segments.Add(new SegmentRowVm
            {
                Start = FormatTime(start),
                End = FormatTime(end),
                Chord = chord,
                StartSec = start,
                EndSec = end,
            });
        }
        Summary = $"{Path.GetFileName(audioPath)}   •   {result.Duration:F0} с   •   " +
                  $"{result.Bpm:F0} BPM   •   {result.Engine}";
        Status = "Готово.";
    }

    private static string FormatTime(double seconds) =>
        $"{(int)seconds / 60}:{seconds % 60:00.0}";

    private static string FormatClock(double seconds) =>
        $"{(int)seconds / 60}:{(int)seconds % 60:00}";

    public void Dispose() => _player.Dispose();
}
