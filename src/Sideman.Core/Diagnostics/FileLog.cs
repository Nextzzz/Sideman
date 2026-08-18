namespace Sideman.Core.Diagnostics;

/// <summary>
/// Dead-simple file logger: one daily file, thread-safe, and it must
/// NEVER crash the app — logging failures are swallowed by design.
/// </summary>
public static class FileLog
{
    private static readonly object Lock = new();

    public static string Directory { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "Sideman", "logs");

    public static string CurrentFile =>
        Path.Combine(Directory, $"sideman-{DateTime.Now:yyyyMMdd}.log");

    public static void Info(string message) => Write("INFO ", message);

    public static void Error(string message, Exception? exception = null) =>
        Write("ERROR", exception == null
            ? message
            : message + Environment.NewLine + exception);

    private static void Write(string level, string message)
    {
        try
        {
            lock (Lock)
            {
                System.IO.Directory.CreateDirectory(Directory);
                File.AppendAllText(
                    CurrentFile,
                    $"{DateTime.Now:HH:mm:ss.fff} [{level}] {message}{Environment.NewLine}");
            }
        }
        catch
        {
            // Logging must never take the app down.
        }
    }
}
