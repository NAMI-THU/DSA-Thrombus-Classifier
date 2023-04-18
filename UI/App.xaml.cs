using System.IO;
using System.Windows;
using Serilog;
using Serilog.Formatting.Compact;
using Services;
using Services.AiService;

namespace UI;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    public App()
    {
        this.Dispatcher.UnhandledException += OnDispatcherUnhandledException;

        var log = new LoggerConfiguration().WriteTo.Console();

        var logPath = Configuration.LogPath;
        if (Configuration.EnableEvaluationSetup) {
            if (string.IsNullOrEmpty(logPath) || !Directory.Exists(logPath)) {
                log.WriteTo.File(new CompactJsonFormatter(), "evaluation-.json", rollingInterval: RollingInterval.Day);
            }
            else {
                log.WriteTo.File(new CompactJsonFormatter(), Path.Combine(logPath, "evaluation-.json"),
                    rollingInterval: RollingInterval.Day);
            }
        }

        using var l = log.CreateLogger();
        Log.Logger = l;
    }

    private void OnDispatcherUnhandledException(object sender,
        System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
    {
        var errorMessage = $"An unhandled exception occurred: {e.Exception.Message}";
        MessageBox.Show(errorMessage, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        e.Handled = true;
    }
}