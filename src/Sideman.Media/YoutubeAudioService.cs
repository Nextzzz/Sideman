using System.Diagnostics;
using System.Net.Http;
using Sideman.Core.Diagnostics;
using YoutubeExplode;
using YoutubeExplode.Videos.Streams;

namespace Sideman.Media;

/// <summary>
/// Downloads the audio track of a YouTube video to a local file that
/// <see cref="AudioLoader"/> can decode.
///
/// Two engines: YoutubeExplode (fast, in-process) with a fallback to
/// yt-dlp (an external tool updated weekly — YouTube keeps breaking
/// third-party libraries, and yt-dlp survives those changes best).
/// yt-dlp.exe is fetched automatically on first use.
/// </summary>
public sealed class YoutubeAudioService
{
    private static readonly string YtDlpPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Sideman", "tools", "yt-dlp.exe");

    public async Task<string> DownloadAudioAsync(
        string url,
        string targetDirectory,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(targetDirectory);

        try
        {
            var path = await DownloadWithExplodeAsync(url, targetDirectory, progress, ct);
            FileLog.Info($"YouTube audio via YoutubeExplode: {path}");
            return path;
        }
        catch (Exception ex)
        {
            FileLog.Error($"YoutubeExplode failed for {url}, falling back to yt-dlp", ex);
        }

        var fallback = await DownloadWithYtDlpAsync(url, targetDirectory, ct);
        FileLog.Info($"YouTube audio via yt-dlp: {fallback}");
        return fallback;
    }

    private async Task<string> DownloadWithExplodeAsync(
        string url, string targetDirectory, IProgress<double>? progress, CancellationToken ct)
    {
        var client = new YoutubeClient();
        var video = await client.Videos.GetAsync(url, ct);
        var manifest = await client.Videos.Streams.GetManifestAsync(url, ct);

        // Prefer m4a (AAC): Windows decodes it natively, no ffmpeg needed.
        var stream = manifest
            .GetAudioOnlyStreams()
            .Where(s => s.Container == Container.Mp4)
            .OrderByDescending(s => s.Bitrate)
            .FirstOrDefault()
            ?? throw new InvalidOperationException("No m4a audio stream available for this video.");

        string path = Path.Combine(targetDirectory, Sanitize(video.Title) + ".m4a");
        await client.Videos.Streams.DownloadAsync(stream, path, progress, ct);
        return path;
    }

    private static async Task<string> DownloadWithYtDlpAsync(
        string url, string targetDirectory, CancellationToken ct)
    {
        string exe = await EnsureYtDlpAsync(ct);
        string jsRuntime = await GetJsRuntimeArgsAsync(ct);
        string template = Path.Combine(
            targetDirectory, $"yt_{DateTime.Now:yyyyMMdd_HHmmss}.%(ext)s");

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            // m4a strongly preferred (native Windows decode); --print gives
            // us the final file path on stdout. A JS runtime is required by
            // modern yt-dlp to solve YouTube's stream signatures.
            Arguments = $"-f \"ba[ext=m4a]/ba\" --no-playlist --no-warnings {jsRuntime} " +
                        $"--print after_move:filepath -o \"{template}\" \"{url}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start yt-dlp.");
        string stdout = await process.StandardOutput.ReadToEndAsync(ct);
        string stderr = await process.StandardError.ReadToEndAsync(ct);
        await process.WaitForExitAsync(ct);

        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"yt-dlp exited with code {process.ExitCode}: {Truncate(stderr, 400)}");

        string? path = stdout
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .LastOrDefault(File.Exists);
        return path
            ?? throw new InvalidOperationException("yt-dlp finished but produced no file.");
    }

    private static async Task<string> EnsureYtDlpAsync(CancellationToken ct)
    {
        if (File.Exists(YtDlpPath))
            return YtDlpPath;

        FileLog.Info("Downloading yt-dlp.exe (first use)...");
        Directory.CreateDirectory(Path.GetDirectoryName(YtDlpPath)!);
        using var http = new HttpClient();
        var bytes = await http.GetByteArrayAsync(
            "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe", ct);
        await File.WriteAllBytesAsync(YtDlpPath, bytes, ct);
        FileLog.Info($"yt-dlp.exe saved to {YtDlpPath} ({bytes.Length / 1e6:F1} MB)");
        return YtDlpPath;
    }

    private static string? _jsRuntimeArgs;

    /// <summary>yt-dlp needs a JavaScript runtime to solve YouTube stream
    /// signatures: use Node.js when the machine has it, otherwise download
    /// a portable Deno once.</summary>
    private static async Task<string> GetJsRuntimeArgsAsync(CancellationToken ct)
    {
        if (_jsRuntimeArgs != null)
            return _jsRuntimeArgs;

        if (CanRun("node", "--version"))
        {
            FileLog.Info("yt-dlp JS runtime: system Node.js");
            return _jsRuntimeArgs = "--js-runtimes node";
        }

        string denoPath = Path.Combine(Path.GetDirectoryName(YtDlpPath)!, "deno.exe");
        if (!File.Exists(denoPath))
        {
            FileLog.Info("Downloading portable Deno (JS runtime for yt-dlp)...");
            string zipPath = Path.Combine(Path.GetTempPath(), "sideman_deno.zip");
            using (var http = new HttpClient())
            {
                var bytes = await http.GetByteArrayAsync(
                    "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip", ct);
                await File.WriteAllBytesAsync(zipPath, bytes, ct);
            }
            System.IO.Compression.ZipFile.ExtractToDirectory(
                zipPath, Path.GetDirectoryName(denoPath)!, overwriteFiles: true);
            File.Delete(zipPath);
            FileLog.Info($"Deno saved to {denoPath}");
        }
        return _jsRuntimeArgs = $"--js-runtimes deno:\"{denoPath}\"";
    }

    private static bool CanRun(string fileName, string args)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            });
            return process != null && process.WaitForExit(5000) && process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    private static string Sanitize(string title)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var cleaned = new string(title.Select(c => invalid.Contains(c) ? '_' : c).ToArray());
        return cleaned.Length > 80 ? cleaned[..80] : cleaned;
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max] + "…";
}
