using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using Microsoft.WindowsAPICodePack.Dialogs;
using Services.AiService;
using Services.AiService.Interpreter;
using Services.AiService.Responses;

namespace ThromboMapUI.View;

public class MainWindowViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;
    public const int DisplayPathLength = 30;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    private RelayCommand<object>? _startClassificationCommand;
    private RelayCommand<object>? _windowLoadedCommand;
    private RelayCommand<string>? _frontalPreparedNotification;
    private RelayCommand<string>? _lateralPreparedNotification;
    private RelayCommand<object>? _browseModelFolderCommand;
    private string? _fileNameFrontal;
    private string? _fileNameLateral;
    private bool _classificationInProgress;
    private string _classificationResultsText = "Not run yet.";
    private bool _modelsPrepared;
    private bool _aiClassificationDone;
    private double _aiClassificationOutcomeCombined;
    private double _aiClassificationThreshold = 0.5;
    private string _classificationResultText;
    private SolidColorBrush _classificationResultColor;
    private string _modelSelectionFolder;
    private PackIcon _modelSelectionFolderBadge = new(){Kind = PackIconKind.Alert};
    private double _classificationProgressPercentage;

    // Is that so good? We might fail here in the constructor
    private ResultInterpreter _resultInterpreter = new();


    public ICommand BrowseModelFolderCommand{
        get
        {
            return _browseModelFolderCommand ??= new RelayCommand<object>(s => OnBrowseModelFolderClicked());
        }
    }

    public ICommand FrontalPreparedNotification
    {
        get
        {
            return _frontalPreparedNotification ??= new RelayCommand<string>(s => FileNameFrontal = s);
        }
    }
    public ICommand LateralPreparedNotification
    {
        get
        {
            return _lateralPreparedNotification ??= new RelayCommand<string>(s => FileNameLateral = s);
        }
    }
    
    public ICommand StartClassificationCommand { 
        get {
            return _startClassificationCommand ??= new RelayCommand<object>(p => StartClassificationOnClick(), a => true);
        } 
    }

    public ICommand WindowLoadedCommand
    {
        get
        {
            return _windowLoadedCommand ??= new RelayCommand<object>(_ => {});
            // return _windowLoadedCommand ??= new RelayCommand<object>(p=>PreloadModels(), a=>true); 
        }
    }

    
    private string? FileNameFrontal
    {
        get => _fileNameFrontal;
        set
        {
            if (_fileNameFrontal == value) return;
            _fileNameFrontal = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
        }
    }
    
    private string? FileNameLateral
    {
        get => _fileNameLateral;
        set
        {
            if (_fileNameLateral == value) return;
            _fileNameLateral = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
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

    public bool StartClassificationEnabled => FileNameFrontal != "" && FileNameLateral != "" && ModelsPrepared && !ClassificationInProgress;

    private bool ModelsPrepared
    {
        get => _modelsPrepared;
        set
        {
            if (value == _modelsPrepared) return;
            _modelsPrepared = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(StartClassificationEnabled));
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

    public bool AiClassificationDone
    {
        get => _aiClassificationDone;
        set
        {
            if (value == _aiClassificationDone) return;
            _aiClassificationDone = value;
            OnPropertyChanged();
        }
    }

    public double AiClassificationOutcomeCombined
    {
        get => _aiClassificationOutcomeCombined;
        set
        {
            if (value.Equals(_aiClassificationOutcomeCombined)) return;
            _aiClassificationOutcomeCombined = value;
            
            OnPropertyChanged();
            
            // TODO: Not quite right yet
            if (_resultInterpreter.HasThrombus(value))
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
            OnPropertyChanged(nameof(AiClassificationOutcomeCombined));
            OnPropertyChanged(nameof(Threshold_TP));
            OnPropertyChanged(nameof(Threshold_FP));
            OnPropertyChanged(nameof(Threshold_FN));
            OnPropertyChanged(nameof(Threshold_TN));
            
            OnPropertyChanged(nameof(Accuracy));
            OnPropertyChanged(nameof(F1Score));
            OnPropertyChanged(nameof(Precision));
            OnPropertyChanged(nameof(Recall));
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

    public PackIcon ModelSelectionFolderBadge
    {
        get => _modelSelectionFolderBadge;
        set
        {
            if (Equals(value, _modelSelectionFolderBadge)) return;
            _modelSelectionFolderBadge = value;
            OnPropertyChanged();
        }
    }

    public string ClassificationResultText
    {
        get => _classificationResultText;
        set
        {
            if (value == _classificationResultText) return;
            _classificationResultText = value;
            OnPropertyChanged();
        }
    }

    public SolidColorBrush ClassificationResultColor
    {
        get => _classificationResultColor;
        set
        {
            if (Equals(value, _classificationResultColor)) return;
            _classificationResultColor = value;
            OnPropertyChanged();
        }
    }

    public double ClassificationProgressPercentage
    {
        get => _classificationProgressPercentage;
        set
        {
            if (value.Equals(_classificationProgressPercentage)) return;
            _classificationProgressPercentage = value;
            OnPropertyChanged();
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
            ModelSelectionFolder = dialog.FileName;
            // TODO: Proper validation of folder contents
            ModelSelectionFolderBadge = new() { Kind = PackIconKind.Check };
            PreloadModels();
        }
    }

    public string ModelSelectionFolder
    {
        get => _modelSelectionFolder;
        set
        {
            if (value == _modelSelectionFolder) return;
            _modelSelectionFolder = value;
            OnPropertyChanged();
        }
    }

    private async void StartClassificationOnClick()
    {
        // TODO Check if paths are valid and everything is converted and set, and only then enable the button
        // TODO: Make sure, that the folder structure is like this: /frontal/model1.pt .. /lateral/model1.pt
        // The models must have the same name!
        ClassificationInProgress = true;

        //var modelFolders = Directory.GetDirectories(ModelSelectionFolder);
        var responses = new List<ClassificationResponse>();
        var frontals = Path.Join(ModelSelectionFolder, "frontal");
        var models = Directory.GetFiles(frontals);
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
        //ClassificationResultsText = $"Frontal: {string.Join(", ", response.OutputFrontal)} | Lateral: {string.Join(", ", response.OutputLateral)}";
        AiClassificationOutcomeCombined = CalculateCombinedResult(responses);
    }

    private double CalculateCombinedResult(List<ClassificationResponse> responses)
    {
        // TODO: Average isn't working yet + is wrong
        var avgF = responses.Sum(r => r.OutputFrontal.Sum()) / (responses.Count/2);
        var avgL = responses.Sum(r => r.OutputLateral.Sum()) / (responses.Count / 2);
        return (avgF + avgL) / 2;
    }

    private async void PreloadModels()
    {
        ModelsPrepared = false;
        ClassificationInProgress = true;
        ClassificationResultsText = "Initializing models...";
        await AiServiceCommunication.PreloadModels(ModelSelectionFolder);
        ClassificationResultsText = "";
        ModelsPrepared = true;
        ClassificationInProgress = false;
    }

    private void LoadInterpreter()
    {
        _resultInterpreter.LoadData();
        AiClassificationThreshold = _resultInterpreter.CalculateBestThreshold();
    }

    public MainWindowViewModel()
    {
        // TODO
        LoadInterpreter();
    }
}