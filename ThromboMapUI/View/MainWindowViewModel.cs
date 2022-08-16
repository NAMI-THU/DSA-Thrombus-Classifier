using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using Services.AiService;

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
    private string? _fileNameFrontal;// = @"C:\Users\Timo\Documents\GitHub\DSA_CNN_Demonstrator\Python-SourceCode\images\thrombYes\263-01-aci-l-f.nii";
    private string? _fileNameLateral;// = @"C:\Users\Timo\Documents\GitHub\DSA_CNN_Demonstrator\Python-SourceCode\images\thrombYes\263-01-aci-l-s.nii";
    private bool _classificationInProgress;
    private string _classificationResultsText = "Not run yet.";
    private bool _modelsPrepared;
    private bool _aiClassificationDone;
    private double _aiClassificationOutcomeCombined;
    private double _aiClassificationThreshold = 0.5;
    private string _classificationResultText;
    private SolidColorBrush _classificationResultColor;
    private double _avgModelOutcome;

    public double AvgModelOutcome
    {
        get => _avgModelOutcome;
        set
        {
            if (value.Equals(_avgModelOutcome)) return;
            _avgModelOutcome = value;
            OnPropertyChanged();
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
            return _windowLoadedCommand ??= new RelayCommand<object>(p=>PreloadModels(), a=>true); 
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
            
            if (value >= AiClassificationThreshold)
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
            _aiClassificationThreshold = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(AiClassificationOutcomeCombined));
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

    private async void StartClassificationOnClick()
    {
        // TODO Check if paths are valid and everything is converted and set, and only then enable the button
        ClassificationInProgress = true;
        var response = await AiServiceCommunication.ClassifySequence(FileNameFrontal, FileNameLateral);
        ClassificationInProgress = false;
        AiClassificationDone = true;
        ClassificationResultsText = $"Frontal: {string.Join(", ", response.OutputFrontal)} | Lateral: {string.Join(", ", response.OutputLateral)}";
        AiClassificationOutcomeCombined = CalculateCombinedResult(response.OutputFrontal, response.OutputLateral);
    }

    private double CalculateCombinedResult(float[]? frontals, float[]? laterals)
    {
        var avgF = frontals.Sum() / frontals.Length;
        var avgL = laterals.Sum() / laterals.Length;
        return (avgF + avgL) / 2;
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