using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using Serilog;
using Services;
using Services.AiService;
using Services.AiService.Interpreter;
using Services.AiService.Responses;
using ThromboMapUI.Util;

namespace ThromboMapUI.View;

public class MainWindowViewModel : INotifyPropertyChanged
{
    private readonly ResultInterpreter _resultInterpreter = new();
    
    private bool _aiClassificationDone;
    private double _aiClassificationOutcomeCombined;
    private double _aiClassificationThreshold = 0.5;
    private RelayCommand<object>? _browseModelFolderCommand;
    private bool _classificationInProgress;
    private double _classificationProgressPercentage;
    private SolidColorBrush _classificationResultColor = Brushes.White;
    private string _classificationResultFrontal = "/";
    private string _classificationResultLateral = "/";
    private string _classificationResultText = "Not run yet";
    private bool _conversionFrontalDone;
    private bool _conversionLateralDone;
    private string? _fileNameFrontal;
    private string? _fileNameLateral;
    private RelayCommand<string>? _frontalPreparedNotification;
    private RelayCommand<string>? _lateralPreparedNotification;
    private string? _modelSelectionFolder;
    private PackIcon _modelSelectionFolderBadge = new(){Kind = PackIconKind.Alert};
    private bool _modelsPrepared;

    private RelayCommand<object>? _startClassificationCommand;
    private RelayCommand<object>? _windowLoadedCommand;


    public ICommand ChangeFrontalNiftiImageCommand { get; set; }
    public ICommand ChangeLateralNiftiImageCommand { get; set; }
    
    public ICommand BrowseModelFolderCommand{
        get
        {
            return _browseModelFolderCommand ??= new RelayCommand<object>(_ => OnBrowseModelFolderClicked());
        }
    }

    public ICommand FrontalPreparedNotification
    {
        get
        {
            return _frontalPreparedNotification ??= new RelayCommand<string>(s =>
            {
                FileNameFrontal = s;
                ConversionFrontalDone = true;
            });
        }
    }

    public ICommand LateralPreparedNotification
    {
        get
        {
            return _lateralPreparedNotification ??= new RelayCommand<string>(async s =>
            {
                FileNameLateral = s;
                ConversionLateralDone = true;
            });
        }
    }

    public ICommand StartClassificationCommand { 
        get {
            return _startClassificationCommand ??= new RelayCommand<object>(_ => StartClassificationOnClick());
        } 
    }

    public ICommand WindowLoadedCommand
    {
        get
        {
            return _windowLoadedCommand ??= new RelayCommand<object>(_ =>
            {
                LoadInterpreter();
                RestoreUserSettings();
            });
        }
    }

    private bool ConversionFrontalDone
    {
        get => _conversionFrontalDone;
        set
        {
            _conversionFrontalDone = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
            UpdateImages();
        }
    }

    private bool ConversionLateralDone
    {
        get => _conversionLateralDone;
        set
        {
            _conversionLateralDone = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
            UpdateImages();
        }
    }

    private string? FileNameFrontal
    {
        get => _fileNameFrontal;
        set
        {
            _fileNameFrontal = value;
            OnPropertyChanged();
        }
    }

    private string? FileNameLateral
    {
        get => _fileNameLateral;
        set
        {
            _fileNameLateral = value;
            OnPropertyChanged();
        }
    }

    public bool ClassificationInProgress
    {
        get => _classificationInProgress;
        private set
        {
            _classificationInProgress = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }

    public bool StartClassificationEnabled => ConversionFrontalDone && ConversionLateralDone && ModelsPrepared && !ClassificationInProgress;

    private bool ModelsPrepared
    {
        get => _modelsPrepared;
        set
        {
            _modelsPrepared = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }
    public bool AiClassificationDone
    {
        get => _aiClassificationDone;
        set
        {
            _aiClassificationDone = value;
            OnPropertyChanged();
        }
    }

    public double AiClassificationOutcomeCombined
    {
        get => _aiClassificationOutcomeCombined;
        set
        {
            _aiClassificationOutcomeCombined = value;
            
            OnPropertyChanged();
            UpdateClassificationText();
        }
    }

    public double AiClassificationThreshold
    {
        get => _aiClassificationThreshold;
        set
        {
            if (value.Equals(_aiClassificationThreshold)) return;
            value = Math.Round(value, 2);
            _aiClassificationThreshold = value;
            _resultInterpreter.Threshold = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(Threshold_TP));
            OnPropertyChanged(nameof(Threshold_FP));
            OnPropertyChanged(nameof(Threshold_FN));
            OnPropertyChanged(nameof(Threshold_TN));
            
            OnPropertyChanged(nameof(Accuracy));
            OnPropertyChanged(nameof(F1Score));
            OnPropertyChanged(nameof(Precision));
            OnPropertyChanged(nameof(Recall));
            OnPropertyChanged(nameof(MCC));
            
            UpdateClassificationText();
        }
    }

    public string Threshold_TP => _resultInterpreter.TruePositivesPercentage;
    public string Threshold_FP => _resultInterpreter.FalsePositivesPercentage;
    public string Threshold_FN => _resultInterpreter.FalseNegativesPercentage;
    public string Threshold_TN => _resultInterpreter.TrueNegativesPercentage;

    public string Accuracy => _resultInterpreter.AccuracyString;
    public string F1Score => _resultInterpreter.F1ScoreString;
    public string Precision => _resultInterpreter.PrecisionString;
    public string Recall => _resultInterpreter.RecallString;
    public string MCC => _resultInterpreter.MCCString;

    public PackIcon ModelSelectionFolderBadge
    {
        get => _modelSelectionFolderBadge;
        set
        {
            _modelSelectionFolderBadge = value;
            OnPropertyChanged();
        }
    }

    public string ClassificationResultText
    {
        get => _classificationResultText;
        private set
        {
            _classificationResultText = value;
            OnPropertyChanged();
        }
    }

    public SolidColorBrush ClassificationResultColor
    {
        get => _classificationResultColor;
        private set
        {
            _classificationResultColor = value;
            OnPropertyChanged();
        }
    }

    public double ClassificationProgressPercentage
    {
        get => _classificationProgressPercentage;
        set
        {
            _classificationProgressPercentage = value;
            OnPropertyChanged();
        }
    }

    public string ClassificationResultFrontal
    {
        get => _classificationResultFrontal;
        private set
        {
            _classificationResultFrontal = value;
            OnPropertyChanged();
        }
    }

    public string ClassificationResultLateral
    {
        get => _classificationResultLateral;
        private set
        {
            _classificationResultLateral = value;
            OnPropertyChanged();
        }
    }

    public string? ModelSelectionFolder
    {
        get => _modelSelectionFolder;
        private set
        {
            _modelSelectionFolder = value;
            OnPropertyChanged();
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    private void UpdateClassificationText()
    {
        if (_resultInterpreter.HasThrombus(AiClassificationOutcomeCombined))
        {
            ClassificationResultText = "Thrombus detected!";
            ClassificationResultColor = Brushes.DarkRed;
        }
        else
        {
            ClassificationResultText = "No Thrombus detected.";
            ClassificationResultColor = Brushes.DarkGreen;
        }
    }

    private void OnBrowseModelFolderClicked()
    {
        var dialog = new CommonOpenFileDialog
        {
            IsFolderPicker = true
        };
        if (dialog.ShowDialog() == CommonFileDialogResult.Ok)
        {
            var success = UpdateModelPath(dialog.FileName);
            if(!success)
            {
                MessageBox.Show("The structure of the selected directory is invalid. Ensure that the selected folder has a frontal and lateral subdirectory, in which the respective model (or multiple model-folds) is. Additionally, make sure that the filenames of both versions are identical (that means, corresponding model names should have the same name, e.g. frontal/fold1.pt and lateral/fold1.pt or frontal/model.pt and lateral/model.pt", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }

    private bool UpdateModelPath(string path)
    {
        var structureOkay = Validation.CheckModelFolderStructure(path);
        if (structureOkay)
        {
            ModelSelectionFolder = path;
            ModelSelectionFolderBadge = new PackIcon { Kind = PackIconKind.Sync };
            PreloadModels();
            UserPersistence.Default.ModelPath = path;
            return true;
        }

        return false;
    }

    private async void StartClassificationOnClick()
    {
        ClassificationInProgress = true;

        var responses = new List<ClassificationResponse>();
        var folderFrontal = Path.Join(ModelSelectionFolder, "frontal");
        var models = Directory.GetFiles(folderFrontal);
        foreach(var model in models)
        {
            var modelName = Path.GetFileName(model);
            double currentCount = 0;
            var total = models.Length;
            ClassificationProgressPercentage = -1;
            var response = await AiServiceCommunication.ClassifySequence(modelName, modelName, FileNameFrontal, FileNameLateral);
            responses.Add(response);
            currentCount++;
            ClassificationProgressPercentage = currentCount / total;
        }
        
        ClassificationInProgress = false;
        AiClassificationDone = true;
        var averages = ResultInterpreter.CalculateCombinedResult(responses);
        AiClassificationOutcomeCombined = averages.Item1;
        ClassificationResultFrontal = $"{averages.Item2:F2}";
        ClassificationResultLateral = $"{averages.Item3:F2}";

        if (Configuration.EnableEvaluationSetup)
        {
            var result = MessageBox.Show("Was there a thrombus in this sequence? \n(You can disable this evaluation mode in appsettings.json)","Service evaluation", MessageBoxButton.YesNo, MessageBoxImage.Question);
            var truth = result == MessageBoxResult.Yes ? 1 : 0;
            Log.Information("Classification evaluated: {@OutputFrontal}, {@OutputLateral}, {@Truth}. {@ModelPath} with {@ImageFrontal} and {@ImageLateral}", averages.Item2, averages.Item3, truth, ModelSelectionFolder, FileNameFrontal, FileNameLateral);
        }
    }

    private async void PreloadModels()
    {
        if (ModelSelectionFolder == null) return;
        
        ModelsPrepared = false;
        ClassificationInProgress = true;
        ClassificationResultText = "Initializing models...";
        try
        {
            await AiServiceCommunication.PreloadModels(ModelSelectionFolder);
            ModelSelectionFolderBadge = new PackIcon { Kind = PackIconKind.Check };
            ClassificationResultText = "Not run yet";
            ModelsPrepared = true;
        }
        catch (HttpRequestException)
        {
            MessageBox.Show("Server is unavailable. Please make sure it is running and select the folder again.", "Connection refused",
                MessageBoxButton.OK, MessageBoxImage.Error);
            ModelsPrepared = false;
            ModelSelectionFolderBadge = new PackIcon { Kind = PackIconKind.Alert };
        }
        finally
        {
            ClassificationResultText = "Not run yet";
            ClassificationInProgress = false;
        }
    }

    private async void UpdateImages()
    {
        if (!ConversionFrontalDone || !ConversionLateralDone) return;
        var imageResponse = await AiServiceCommunication.LoadImages(FileNameFrontal, FileNameLateral);
        ChangeFrontalNiftiImageCommand.Execute(imageResponse.img1);
        ChangeLateralNiftiImageCommand.Execute(imageResponse.img2);
    }

    private async void LoadInterpreter()
    {
        await _resultInterpreter.LoadData().ConfigureAwait(false);
        AiClassificationThreshold = _resultInterpreter.CalculateBestThreshold();
    }

    private void RestoreUserSettings()
    {
        var path = UserPersistence.Default.ModelPath;
        if (path is not null)
        {
            UpdateModelPath(path);
        }
    }
}