using System.Windows;
using Sideman.Core.Diagnostics;

namespace Sideman.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        FileLog.Info("=== Sideman started ===");

        // Every unhandled exception ends up in the log with a full stack.
        DispatcherUnhandledException += (_, args) =>
        {
            FileLog.Error("Unhandled UI exception", args.Exception);
            MessageBox.Show(
                args.Exception.Message + "\n\nДеталі записано в лог:\n" + FileLog.CurrentFile,
                "Sideman — помилка", MessageBoxButton.OK, MessageBoxImage.Error);
            args.Handled = true;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, args) =>
            FileLog.Error("Unhandled domain exception", args.ExceptionObject as Exception);
        TaskScheduler.UnobservedTaskException += (_, args) =>
        {
            FileLog.Error("Unobserved task exception", args.Exception);
            args.SetObserved();
        };
    }
}
