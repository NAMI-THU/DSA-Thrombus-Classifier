using System.Windows;

namespace UI.View;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        var vm = (MainWindowViewModel)DataContext;
        FrontalNiftiView.FilePreparedNotificationCommand = vm.FrontalPreparedNotification;
        LateralNiftiView.FilePreparedNotificationCommand = vm.LateralPreparedNotification;
        vm.ChangeFrontalNiftiImageCommand = FrontalNiftiView.ChangeImageCommand;
        vm.ChangeLateralNiftiImageCommand = LateralNiftiView.ChangeImageCommand;

        FrontalNiftiView.ResetFilesOtherCommand = LateralNiftiView.ResetFilesCommand;
        LateralNiftiView.ResetFilesOtherCommand = FrontalNiftiView.ResetFilesCommand;
    }
}