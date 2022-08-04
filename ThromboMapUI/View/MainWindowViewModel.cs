using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
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
    public event PropertyChangedEventHandler? PropertyChanged;
    public const int DisplayPathLength = 30;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    

    private RelayCommand<object>? _startClassificationCommand;
    private RelayCommand<object>? _browseFrontalCommand;
    private RelayCommand<object>? _browseLateralCommand;
    private RelayCommand<object>? _convertFrontalCommand;
    private RelayCommand<object>? _convertLateralCommand;
    private RelayCommand<object>? _windowLoadedCommand;
    private string? _fileNameFrontal;
    private string? _fileNameLateral;
    private bool _convertLateralInProgress;
    private bool _convertFrontalInProgress;
    private bool _convertFrontalEnabled;
    private bool _convertLateralEnabled;
    private bool _classificationInProgress;
    private ImageSource _imageDisplayFrontal = new BitmapImage();
    private ImageSource _imageDisplayLateral = new BitmapImage();
    private string _classificationResultsText = "Not run yet.";
    private bool _fileFrontalValid;
    private bool _fileLateralValid;
    private bool _modelsPrepared;

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

    public ICommand WindowLoadedCommand
    {
        get
        {
            return _windowLoadedCommand ??= new RelayCommand<object>(p=>PreloadModels(), a=>true); 
        }
    }

    public string? FileNameFrontal
    {
        get => _fileNameFrontal;
        private set
        {
            if (_fileNameFrontal == value) return;
            _fileNameFrontal = value;
            OnPropertyChanged();
        }
    }

    public string? FileNameLateral
    {
        get => _fileNameLateral;
        private set
        {
            if (_fileNameLateral == value) return;
            _fileNameLateral = value;
            OnPropertyChanged();
        }
    }

    public bool ConvertFrontalInProgress
    {
        get => _convertFrontalInProgress;
        private set
        {
            if (_convertFrontalInProgress == value) return;
            _convertFrontalInProgress = value;
            OnPropertyChanged();
        }
    }
    
    public bool ConvertLateralInProgress
    {
        get => _convertLateralInProgress;
        private set
        {
            if (_convertLateralInProgress == value) return;
            _convertLateralInProgress = value;
            OnPropertyChanged();
        }
    }
    
    public bool ConvertFrontalEnabled
    {
        get => _convertFrontalEnabled;
        private set
        {
            if (_convertFrontalEnabled == value) return;
            _convertFrontalEnabled = value;
            OnPropertyChanged();
        }
    }
    
    public bool ConvertLateralEnabled
    {
        get => _convertLateralEnabled;
        private set
        {
            if (_convertLateralEnabled == value) return;
            _convertLateralEnabled = value;
            OnPropertyChanged();
        }
    }

    public bool ClassificationInProgress
    {
        get => _classificationInProgress;
        private set
        {
            if (_classificationInProgress == value) return;
            _classificationInProgress = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }

    public bool StartClassificationEnabled => FileFrontalValid && FileLateralValid && ModelsPrepared && !ClassificationInProgress;

    public bool FileFrontalValid
    {
        get => _fileFrontalValid;
        private set
        {
            if (value == _fileFrontalValid) return;
            _fileFrontalValid = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }

    public bool FileLateralValid
    {
        get => _fileLateralValid;
        private set
        {
            if (value == _fileLateralValid) return;
            _fileLateralValid = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }

    public bool ModelsPrepared
    {
        get => _modelsPrepared;
        private set
        {
            if (value == _modelsPrepared) return;
            _modelsPrepared = value;
            OnPropertyChanged();
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
            OnPropertyChanged();
        }
    }
    public ImageSource ImageDisplayLateral
    {
        get => _imageDisplayLateral;
        private set
        {
            if (_imageDisplayLateral == value) return;
            _imageDisplayLateral = value;
            OnPropertyChanged();
        }
    }

    public string ClassificationResultsText
    {
        get => _classificationResultsText;
        private set
        {
            if (_classificationResultsText == value) return;
            _classificationResultsText = value;
            OnPropertyChanged();
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
        CheckFileFormats();
        
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
        CheckFileFormats();
        
        // Load and display image
        // var bitmap = await ImageUtil.LoadNiftiToBitmap(newPath);
        // ImageDisplayLateral = bitmap;
    }
    
    private async void StartClassificationOnClick()
    {
        // TODO Check if paths are valid and everything is converted and set, and only then enable the button
        ClassificationInProgress = true;
        var response = await AiServiceCommunication.ClassifySequence(FileNameFrontal, FileNameLateral);
        ClassificationInProgress = false;
        ClassificationResultsText = $"Frontal: {string.Join(", ", response.OutputFrontal)} | Lateral: {string.Join(", ", response.OutputLateral)}";
    }

    private string? OpenFileChooser()
    {
        var openFileDialog = new OpenFileDialog();
        return openFileDialog.ShowDialog() == true ? openFileDialog.FileName : null;
    }

    private void CheckFileFormats()
    {
        if (FileNameFrontal != null && !ConvertFrontalInProgress)
        {
            if (FileNameFrontal.EndsWith(".nii"))
            {
                // TODO: Display green icon
                ConvertFrontalEnabled = false;
                FileFrontalValid = true;
            }
            else
            {
                // TODO: Do proper check
                ConvertFrontalEnabled = true;
                FileFrontalValid = false;
            }
        }
        else
        {
            FileFrontalValid = false;
        }

        if (FileNameLateral != null && !ConvertLateralInProgress)
        {
            if (FileNameLateral.EndsWith(".nii"))
            {
                ConvertLateralEnabled = false;
                FileLateralValid = true;
            }
            else
            {
                ConvertLateralEnabled = true;
                FileLateralValid = false;
            }
        }
        else
        {
            FileLateralValid = false;
        }
    }

    private async void PreloadModels()
    {
        ModelsPrepared = false;
        ClassificationInProgress = true;
        ClassificationResultsText = "Initializing models...";
        await AiServiceCommunication.PreloadModels();
        ClassificationResultsText = "";
        ModelsPrepared = true;
        ClassificationInProgress = false;
    }
}