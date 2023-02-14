using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using UI.Util;
using License = UI.Util.License;

namespace UI.View;

public class AboutWindowViewModel : INotifyPropertyChanged
{
    public ObservableCollection<Attribution> AttributionListUI { get; } = new();
    public ObservableCollection<Attribution> AttributionListService { get; } = new();
    public event PropertyChangedEventHandler? PropertyChanged;
    private RelayCommand<string>? _openUrlCommand;
    public ICommand OpenUrlCommand { 
        get {
            return _openUrlCommand ??= new RelayCommand<string>(OpenUrl);
        }  
    }
    
    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    public AboutWindowViewModel()
    {
        LoadData();
    }

    private void LoadData()
    {
        AttributionListUI.Add(new Attribution("Cli Wrap",new License("MIT","https://github.com/Tyrrrz/CliWrap/blob/master/License.txt"), "Oleksii Holub"));
        AttributionListUI.Add(new Attribution("MaterialDesign",new License("MIT","https://github.com/MaterialDesignInXAML/MaterialDesignInXamlToolkit/blob/master/LICENSE"),"James Willock"));
        AttributionListUI.Add(new Attribution("Microsoft.Xaml.Behaviors.WPF",new License("MIT","https://licenses.nuget.org/MIT"),"Microsoft"));
        AttributionListUI.Add(new Attribution("Nifti.NET",new License("MIT","https://github.com/plwp/Nifti.NET/blob/master/LICENSE"),"Patrick Prendergast"));
        AttributionListUI.Add(new Attribution("Serilog",new License("Apache V2","https://github.com/serilog/serilog/blob/dev/LICENSE"),"Serilog Contributors"));
        AttributionListUI.Add(new Attribution("Newtonsoft.Json",new License("MIT","https://github.com/JamesNK/Newtonsoft.Json/blob/master/LICENSE.md"),"James Newton-King"));
    }

    private void OpenUrl(string? url)
    {
        if (url is not null && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
    }
}