using Sideman.Cli;
using Sideman.Core.Analysis;
using Sideman.Media;

if (args.Length == 0)
{
    Console.WriteLine("""
        Sideman CLI — chord & rhythm analysis.

        usage:
          sideman analyze <file.wav|mp3|m4a>   analyze a local audio file
          sideman analyze <youtube-url>        download audio and analyze
          sideman demo [out.wav]               synthesize a test progression and analyze it
        """);
    return 0;
}

switch (args[0])
{
    case "analyze":
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("analyze: expected a file path or YouTube URL");
            return 1;
        }
        string target = args[1];
        if (target.StartsWith("http", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine("Downloading audio from YouTube...");
            var service = new YoutubeAudioService();
            target = await service.DownloadAudioAsync(
                target, Path.Combine(Path.GetTempPath(), "sideman"));
            Console.WriteLine($"Saved: {target}");
        }
        Analyze(target);
        return 0;
    }

    case "demo":
    {
        string output = args.Length > 1 ? args[1] : "demo_progression.wav";
        var samples = DemoSynth.CampfireProgression();
        AudioLoader.SaveWav(output, samples, AudioLoader.TargetSampleRate);
        Console.WriteLine($"Synthesized G-C-D-Em progression -> {output}");
        Analyze(output);
        return 0;
    }

    default:
        Console.Error.WriteLine($"unknown command: {args[0]}");
        return 1;
}

static void Analyze(string path)
{
    Console.WriteLine($"Analyzing {Path.GetFileName(path)}...");
    var (samples, sampleRate) = AudioLoader.LoadMono(path);
    var analysis = new SongAnalyzer().Analyze(samples, sampleRate);

    Console.WriteLine($"Duration: {analysis.DurationSeconds:F1}s   Tempo: {analysis.Bpm:F0} BPM   Beats: {analysis.BeatTimes.Length}");
    Console.WriteLine("Chords:");
    foreach (var segment in analysis.Chords)
        Console.WriteLine($"  {Format(segment.Start),7} - {Format(segment.End),-7} {segment.Chord.Label,-4} ({segment.Confidence:F2})");

    static string Format(double seconds) =>
        $"{(int)seconds / 60}:{seconds % 60:00.0}";
}
