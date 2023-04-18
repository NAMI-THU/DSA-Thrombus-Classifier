using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
        get { return _openUrlCommand ??= new RelayCommand<string>(OpenUrl); }
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
        AttributionListUI.Add(new Attribution("SimpleITK",
            new License("Apache V2", "https://github.com/SimpleITK/SimpleITK/blob/master/LICENSE"), "NumFOCUS"));
        AttributionListUI.Add(new Attribution("plastimatch",
            new License("Plastimatch Software License",
                "https://gitlab.com/plastimatch/plastimatch/-/blob/master/src/LICENSE.TXT"),
            "The General Hospital Corporation Inc."));

        AttributionListUI.Add(new Attribution("Cli Wrap",
            new License("MIT", "https://github.com/Tyrrrz/CliWrap/blob/master/License.txt"), "Oleksii Holub"));
        AttributionListUI.Add(new Attribution("MaterialDesign",
            new License("MIT",
                "https://github.com/MaterialDesignInXAML/MaterialDesignInXamlToolkit/blob/master/LICENSE"),
            "James Willock"));
        AttributionListUI.Add(new Attribution("Microsoft.Xaml.Behaviors.WPF",
            new License("MIT", "https://licenses.nuget.org/MIT"), "Microsoft"));
        AttributionListUI.Add(new Attribution("Nifti.NET",
            new License("MIT", "https://github.com/plwp/Nifti.NET/blob/master/LICENSE"), "Patrick Prendergast"));
        AttributionListUI.Add(new Attribution("Serilog",
            new License("Apache V2", "https://github.com/serilog/serilog/blob/dev/LICENSE"), "Serilog Contributors"));
        AttributionListUI.Add(new Attribution("Newtonsoft.Json",
            new License("MIT", "https://github.com/JamesNK/Newtonsoft.Json/blob/master/LICENSE.md"),
            "James Newton-King"));

        AttributionListUI.Add(new Attribution("Flask",
            new License("BSD", "https://github.com/pallets/flask/blob/main/LICENSE.rst"), "Pallets"));
        AttributionListUI.Add(new Attribution("Pytorch",
            new License("BSD", "https://github.com/pytorch/pytorch/blob/master/LICENSE"), "Pytorch developers"));
        AttributionListUI.Add(new Attribution("numpy",
            new License("BSD", "https://github.com/numpy/numpy/blob/main/LICENSE.txt"), "NumPy developers"));
        AttributionListUI.Add(new Attribution("matplotlib",
            new License("Matplotlib License", "https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE"),
            "Matplotlib developers"));
        AttributionListUI.Add(new Attribution("nibabel",
            new License("MIT", "https://github.com/nipy/nibabel/blob/master/COPYING"), "Nibabel developers"));
        AttributionListUI.Add(new Attribution("opencv-python",
            new License("MIT", "https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt"), "Olli-Pekka Heinisuo"));
        AttributionListUI.Add(new Attribution("pandas",
            new License("BSD", "https://github.com/pandas-dev/pandas/blob/main/LICENSE"), "Pandas developers"));
        AttributionListUI.Add(new Attribution("Pillow",
            new License("HPND", "https://github.com/python-pillow/Pillow/blob/main/LICENSE"), "Pillow developers"));
        AttributionListUI.Add(new Attribution("scipy",
            new License("BSD", "https://github.com/scipy/scipy/blob/main/LICENSE.txt"), "SciPy developers"));
        AttributionListUI.Add(new Attribution("scikit-image",
            new License("BSD", "https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt"),
            "scikit-image developers"));
        AttributionListUI.Add(new Attribution("albumentations",
            new License("MIT", "https://github.com/albumentations-team/albumentations/blob/master/LICENSE"),
            "albumentations developers"));
        AttributionListUI.Add(new Attribution("scikit-learn",
            new License("BSD", "https://github.com/scikit-learn/scikit-learn/blob/main/COPYING"),
            "scikit-learn developers"));
        AttributionListUI.Add(new Attribution("timm",
            new License("Apache V2", "https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE"),
            "huggingface"));
    }

    private void OpenUrl(string? url)
    {
        if (url is not null && RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
    }
}