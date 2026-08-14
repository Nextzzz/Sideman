namespace Sideman.Core.Analysis;

public sealed record SongAnalysis(
    double DurationSeconds,
    double Bpm,
    double[] BeatTimes,
    IReadOnlyList<ChordSegment> Chords);

/// <summary>One entry point: samples in, full song analysis out.</summary>
public sealed class SongAnalyzer
{
    private readonly OnsetDetector _onsets = new();
    private readonly TempoEstimator _tempo = new();
    private readonly BeatTracker _beats = new();
    private readonly ChordRecognizer _chords;

    public SongAnalyzer(ChordRecognizerOptions? chordOptions = null)
    {
        _chords = new ChordRecognizer(chordOptions);
    }

    public SongAnalysis Analyze(float[] samples, int sampleRate)
    {
        var novelty = _onsets.NoveltyCurve(samples, sampleRate);
        double frameRate = _onsets.FrameRate(sampleRate);

        double bpm = _tempo.Estimate(novelty, frameRate);
        var beatFrames = _beats.Track(novelty, frameRate, bpm);
        var beatTimes = beatFrames
            .Select(f => (f * (double)_onsets.Hop + _onsets.NFft / 2.0) / sampleRate)
            .ToArray();

        var chords = _chords.Recognize(samples, sampleRate);

        return new SongAnalysis(
            DurationSeconds: samples.Length / (double)sampleRate,
            Bpm: bpm,
            BeatTimes: beatTimes,
            Chords: chords);
    }
}
