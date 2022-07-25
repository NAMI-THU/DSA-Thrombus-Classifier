using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Services.AiService;
using Services.FileUtils;
using ThromboMapUI.Util;

namespace ThromboMapUI.View;

public class MainWindowViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler PropertyChanged;
    public const int DisplayPathLength = 30;

    protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    

    private RelayCommand<object>? _startClassificationCommand;
    private RelayCommand<object>? _browseFrontalCommand;
    private RelayCommand<object>? _browseLateralCommand;
    private RelayCommand<object>? _convertFrontalCommand;
    private RelayCommand<object>? _convertLateralCommand;
    private string? _fileNameFrontal;
    private string? _fileNameLateral;
    private bool _convertLateralInProgress;
    private bool _convertFrontalInProgress;
    private bool _convertFrontalEnabled;
    private bool _convertLateralEnabled;
    private bool _classificationInProgress;
    private bool _startClassificationEnabled = true; // TODO
    private ImageSource _imageDisplayFrontal = new BitmapImage();
    private ImageSource _imageDisplayLateral = new BitmapImage();
    private string _classificationResultsText = "Not run yet.";

    public ICommand StartClassificationCommand { 
        get {
            return _startClassificationCommand ??= new RelayCommand<object>(p => StartClassificationOnClick(), a => true);
        } 
    }
    
    public ICommand BrowseFrontalCommand { 
        get {
            return _browseFrontalCommand ??= new RelayCommand<object>(p => BrowseFrontalOnClick(), a => true);
        } 
    }
    public ICommand BrowseLateralCommand { 
        get {
            return _browseLateralCommand ??= new RelayCommand<object>(p => BrowseLateralOnClick(), a => true);
        } 
    }

    public ICommand ConvertFrontalCommand
    {
        get {
            return _convertFrontalCommand ??= new RelayCommand<object>(p=>ConvertFrontalOnClick(), a=>_convertFrontalEnabled);
        }
    }
    
    public ICommand ConvertLateralCommand
    {
        get {
            return _convertLateralCommand ??= new RelayCommand<object>(p=>ConvertLateralOnClick(), a=>_convertLateralEnabled);
        }
    }

    public string? FileNameFrontal
    {
        get => _fileNameFrontal;
        private set
        {
            if (_fileNameFrontal == value) return;
            _fileNameFrontal = value;
            OnPropertyChanged(nameof(FileNameFrontal));
        }
    }

    public string? FileNameLateral
    {
        get => _fileNameLateral;
        private set
        {
            if (_fileNameLateral == value) return;
            _fileNameLateral = value;
            OnPropertyChanged(nameof(FileNameLateral));
        }
    }

    public bool ConvertFrontalInProgress
    {
        get => _convertFrontalInProgress;
        private set
        {
            if (_convertFrontalInProgress == value) return;
            _convertFrontalInProgress = value;
            OnPropertyChanged(nameof(ConvertFrontalInProgress));
        }
    }
    
    public bool ConvertLateralInProgress
    {
        get => _convertLateralInProgress;
        private set
        {
            if (_convertLateralInProgress == value) return;
            _convertLateralInProgress = value;
            OnPropertyChanged(nameof(ConvertLateralInProgress));
        }
    }
    
    public bool ConvertFrontalEnabled
    {
        get => _convertFrontalEnabled;
        private set
        {
            if (_convertFrontalEnabled == value) return;
            _convertFrontalEnabled = value;
            OnPropertyChanged(nameof(ConvertFrontalEnabled));
        }
    }
    
    public bool ConvertLateralEnabled
    {
        get => _convertLateralEnabled;
        private set
        {
            if (_convertLateralEnabled == value) return;
            _convertLateralEnabled = value;
            OnPropertyChanged(nameof(ConvertLateralEnabled));
        }
    }

    public bool ClassificationInProgress
    {
        get => _classificationInProgress;
        private set
        {
            if (_classificationInProgress == value) return;
            _classificationInProgress = value;
            OnPropertyChanged(nameof(ClassificationInProgress));
        }
    }

    public bool StartClassificationEnabled
    {
        get => _startClassificationEnabled;
        private set
        {
            if (_startClassificationEnabled == value) return;
            _startClassificationEnabled = value;
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }

    public ImageSource ImageDisplayFrontal
    {
        get => _imageDisplayFrontal;
        private set
        {
            if (_imageDisplayFrontal == value) return;
            _imageDisplayFrontal = value;
            OnPropertyChanged(nameof(ImageDisplayFrontal));
        }
    }
    public ImageSource ImageDisplayLateral
    {
        get => _imageDisplayLateral;
        private set
        {
            if (_imageDisplayLateral == value) return;
            _imageDisplayLateral = value;
            OnPropertyChanged(nameof(ImageDisplayFrontal));
        }
    }

    public string ClassificationResultsText
    {
        get => _classificationResultsText;
        private set
        {
            if (_classificationResultsText == value) return;
            _classificationResultsText = value;
            OnPropertyChanged(nameof(ClassificationResultsText));
        }
    }

    private void BrowseFrontalOnClick()
    {
        var file = OpenFileChooser();
        if (file != null)
        {
            FileNameFrontal = file;
            CheckFileFormats();
        }
    }
    
    private void BrowseLateralOnClick()
    {
        var file = OpenFileChooser();
        if (file != null)
        {
            FileNameLateral = file;
            CheckFileFormats();
        }
    }

    private async void ConvertFrontalOnClick()
    {
        ConvertFrontalEnabled = false;
        ConvertFrontalInProgress = true;
        var newPath = await DicomConverter.Dicom2Nifti(FileNameFrontal);
        ConvertFrontalInProgress = false;
        FileNameFrontal = newPath;
        
        // Load and display image
        // var bitmap = await ImageUtil.LoadNiftiToBitmap(newPath);
        // ImageDisplayFrontal = bitmap;
    }

    private async void ConvertLateralOnClick()
    {
        ConvertLateralEnabled = false;
        ConvertLateralInProgress = true;
        var newPath = await DicomConverter.Dicom2Nifti(FileNameLateral);
        ConvertLateralInProgress = false;
        FileNameLateral = newPath;
        
        // Load and display image
        // var bitmap = await ImageUtil.LoadNiftiToBitmap(newPath);
        // ImageDisplayLateral = bitmap;
    }
    
    private async void StartClassificationOnClick()
    {
        // TODO Check if paths are valid and everything is converted and set
        ClassificationInProgress = true;
        var response = await AiServiceCommunication.ClassifySequence(FileNameFrontal, FileNameLateral);
        ClassificationInProgress = false;
        ClassificationResultsText = $"Frontal: {response.OutputFrontal} | Lateral: {response.OutputLateral}";
    }

    private string? OpenFileChooser()
    {
        var openFileDialog = new OpenFileDialog();
        return openFileDialog.ShowDialog() == true ? openFileDialog.FileName : null;
    }

    private void CheckFileFormats()
    {
        if (FileNameFrontal != null)
        {
            if (FileNameFrontal.EndsWith(".nii"))
            {
                // TODO: Display green icon
                ConvertFrontalEnabled = false;
            }
            else
            {
                // TODO: Do proper check
                ConvertFrontalEnabled = true;
            }
        }

        if (FileNameLateral != null)
        {
            if (FileNameLateral.EndsWith(".nii"))
            {
                ConvertLateralEnabled = false;
            }
            else
            {
                ConvertLateralEnabled = true;
            }
        }

        // TODO: Change to proper status and enable StartClassification
    }
}