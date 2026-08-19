using Strunika.Cli;
using Strunika.Core.Analysis;
using Strunika.Media;

if (args.Length == 0)
{
    Console.WriteLine("""
        Strunika CLI — chord & rhythm analysis.

        usage:
          strunika analyze <file.wav|mp3|m4a>   analyze a local audio file
          strunika analyze <youtube-url>        download audio and analyze
          strunika demo [out.wav]               synthesize a test progression and analyze it
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
                target, Path.Combine(Path.GetTempPath(), "strunika"));
            Console.WriteLine($"Saved: {target}");
        }
        var neuralArg = args.FirstOrDefault(a => a.StartsWith("--neural"));
        var priorArg = args.FirstOrDefault(a => a.StartsWith("--keyprior="));
        double keyPrior = priorArg == null
            ? 0.5
            : double.Parse(priorArg.Split('=')[1], System.Globalization.CultureInfo.InvariantCulture);
        if (neuralArg != null)
            AnalyzeNeural(target, neuralArg.Contains('=')
                ? neuralArg.Split('=', 2)[1]
                : Path.Combine("ml", "models", "btc_large_voca.onnx"), keyPrior);
        else
            Analyze(target);
        return 0;
    }

    case "eval":
    {
        // strunika eval [datasetRoot] [limit] [--floor=0.6] [--self=0.97] [--sharp=6] [--bass=0.12]
        string root = args.Length > 1 ? args[1] : "datasets/guitarset";
        int limit = args.Length > 2 && !args[2].StartsWith("--") ? int.Parse(args[2]) : int.MaxValue;

        double Param(string name, double fallback)
        {
            var arg = args.FirstOrDefault(a => a.StartsWith($"--{name}="));
            return arg == null
                ? fallback
                : double.Parse(arg.Split('=')[1], System.Globalization.CultureInfo.InvariantCulture);
        }

        // Fallbacks mirror the Core defaults (GuitarSet-calibrated).
        var options = new ChordRecognizerOptions
        {
            SelfTransition = Param("self", 0.995),
            GateMarginDb = Param("gate", 3.0),
            GateMaxFlatness = Param("maxflat", 0.35),
            Emissions = new ChordEmissionModel
            {
                NoChordSimilarity = Param("floor", 0.45),
                EmissionSharpness = Param("sharp", 2.5),
                BassRootWeight = Param("bass", 0.5),
                BassFifthCredit = Param("bassfifth", 0.9),
                RootChromaWeight = Param("rootw", 0.0),
                SilenceEnergy = Param("silence", 1.0),
            },
        };
        string? neuralModel = args.FirstOrDefault(a => a.StartsWith("--neural"));
        if (neuralModel != null)
        {
            neuralModel = neuralModel.Contains('=')
                ? neuralModel.Split('=', 2)[1]
                : Path.Combine("ml", "models", "btc_large_voca.onnx");
            Console.WriteLine($"neural model: {neuralModel}");
        }
        else
        {
            Console.WriteLine($"floor={options.Emissions.NoChordSimilarity} self={options.SelfTransition} " +
                              $"sharp={options.Emissions.EmissionSharpness} bass={options.Emissions.BassRootWeight}");
        }
        string[]? prefixes = args
            .FirstOrDefault(a => a.StartsWith("--players="))
            ?.Split('=', 2)[1].Split(',');
        var evaluator = new Strunika.Cli.Evaluation.Evaluator
        {
            Limit = limit,
            Options = options,
            NeuralModel = neuralModel,
            NeuralKeyPrior = Param("keyprior", 0.5),
            FilePrefixes = prefixes,
        };
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (files, confusions, scored, correct) = evaluator.Run(
            Path.Combine(root, "annotation"), Path.Combine(root, "audio_mono-mic"));
        Console.WriteLine(Strunika.Cli.Evaluation.Evaluator.Report(files, confusions, scored, correct));
        Console.WriteLine($"Elapsed: {sw.Elapsed.TotalSeconds:F0}s");

        Directory.CreateDirectory("output");
        File.WriteAllLines("output/eval_results.csv",
            new[] { "file,scored,correct,accuracy,top_confusion" }
            .Concat(files.Select(f => $"{f.Name},{f.Scored},{f.Correct},{f.Accuracy:F4},{f.TopConfusion}")));
        Console.WriteLine("Per-file results: output/eval_results.csv");
        return 0;
    }

    case "probe":
    {
        // strunika probe <file> — domain classification diagnostics
        var (samples, sr) = AudioLoader.LoadMono(args[1]);
        double ratio = Strunika.Neural.AudioDomainClassifier.LowBandRatio(samples, sr);
        bool guitarLike = ratio < Strunika.Neural.AudioDomainClassifier.GuitarThreshold;
        Console.WriteLine($"{Path.GetFileName(args[1])}: lowBand={ratio:F4} -> " +
                          (guitarLike ? "GUITAR model" : "MIX model"));
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

static void AnalyzeNeural(string path, string modelPath, double keyPrior = 0.5)
{
    Console.WriteLine($"Analyzing (neural: {Path.GetFileNameWithoutExtension(modelPath)}, keyprior {keyPrior}) {Path.GetFileName(path)}...");
    var (samples, _) = AudioLoader.LoadMono(path, Strunika.Neural.CqtExtractor.SampleRate);
    using var recognizer = new Strunika.Neural.NeuralChordRecognizer(modelPath)
        { KeyPriorStrength = keyPrior };
    var timeline = recognizer.Recognize(samples);
    Console.WriteLine($"key: {recognizer.DetectedKey ?? "не визначено"}");
    foreach (var segment in timeline)
        Console.WriteLine($"  {Format(segment.Start),7} - {Format(segment.End),-7} " +
                          Strunika.Neural.ChordLabels.Pretty(segment.Label));

    static string Format(double seconds) =>
        $"{(int)seconds / 60}:{seconds % 60:00.0}";
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
