using YoutubeExplode;
using YoutubeExplode.Videos.Streams;

namespace Sideman.Media;

/// <summary>
/// Downloads the audio track of a YouTube video to a local file that
/// <see cref="AudioLoader"/> can decode (m4a via MediaFoundation).
/// </summary>
public sealed class YoutubeAudioService
{
    private readonly YoutubeClient _client = new();

    public async Task<string> DownloadAudioAsync(
        string url,
        string targetDirectory,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        var video = await _client.Videos.GetAsync(url, ct);
        var manifest = await _client.Videos.Streams.GetManifestAsync(url, ct);

        // Prefer m4a (AAC): Windows decodes it natively, no ffmpeg needed.
        var stream = manifest
            .GetAudioOnlyStreams()
            .Where(s => s.Container == Container.Mp4)
            .OrderByDescending(s => s.Bitrate)
            .FirstOrDefault()
            ?? throw new InvalidOperationException("No m4a audio stream available for this video.");

        Directory.CreateDirectory(targetDirectory);
        string fileName = Sanitize(video.Title) + ".m4a";
        string path = Path.Combine(targetDirectory, fileName);

        await _client.Videos.Streams.DownloadAsync(stream, path, progress, ct);
        return path;
    }

    private static string Sanitize(string title)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var cleaned = new string(title.Select(c => invalid.Contains(c) ? '_' : c).ToArray());
        return cleaned.Length > 80 ? cleaned[..80] : cleaned;
    }
}
