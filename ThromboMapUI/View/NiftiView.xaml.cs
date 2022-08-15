using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using Services.FileUtils;
using ThromboMapUI.Util;
using Brush = System.Windows.Media.Brush;
using Brushes = System.Windows.Media.Brushes;

namespace ThromboMapUI.View;

public partial class NiftiView : UserControl, INotifyPropertyChanged
{
    private ImageSource? _imageDisplay;
    private string _fileName = "";
    private bool _convertInProgress;
    private bool _convertEnabled;
    public event PropertyChangedEventHandler? PropertyChanged;
    private RelayCommand<object>? _convertCommand;
    private RelayCommand<object>? _browseCommand;
    private bool _filePrepared;
    private PackIconKind _headerIcon = PackIconKind.Error;
    private Brush _headerColor = Brushes.DarkRed;
    private string _imageName;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    public NiftiView()
    {
        InitializeComponent();
    }

    public ImageSource ImageDisplay
    {
        get => _imageDisplay;
        private set
        {
            if (_imageDisplay == value) return;
            _imageDisplay = value;
            OnPropertyChanged();
        }
    }

    public string FileName
    {
        get => _fileName;
        private set{
            if (_fileName == value) return;
            _fileName = value;

            if (_fileName.EndsWith(".nii"))
            {
                ConvertEnabled = false;
                FilePrepared = true;
                LoadImage();
            }
            else
            {
                ConvertEnabled = true;
            }
            
            OnPropertyChanged();
            OnPropertyChanged(nameof(ConvertEnabled));
            OnPropertyChanged(nameof(FilePrepared));
        }
    }

    public bool ConvertInProgress
    {
        get => _convertInProgress;
        private set{
            if (_convertInProgress == value) return;
            _convertInProgress = value;
            OnPropertyChanged();
        }
    }

    public bool ConvertEnabled
    {
        get => _convertEnabled;
        private set{
            if (_convertEnabled == value) return;
            _convertEnabled = value;
            OnPropertyChanged();
        }
    }

    public bool FilePrepared
    {
        get => _filePrepared;
        private set{
            if (_filePrepared == value) return;
            _filePrepared = value;

            if (_filePrepared)
            {
                HeaderColor = new SolidColorBrush(new PaletteHelper().GetTheme().PrimaryDark.Color);
                HeaderIcon = PackIconKind.Check;
            }
            OnPropertyChanged();
        }
    }

    public PackIconKind HeaderIcon
    {
        get => _headerIcon;
        set
        {
            if (Equals(value, _headerIcon)) return;
            _headerIcon = value;
            OnPropertyChanged();
        }
    }

    public Brush HeaderColor
    {
        get => _headerColor ;
        set
        {
            if (Equals(value, _headerColor))  return;
            _headerColor = value;
            OnPropertyChanged();
        }
    }

    public string ImageName
    {
        get => _imageName;
        set
        {
            if (value == _imageName) return;
            _imageName = value;
            OnPropertyChanged();
        }
    }

    public ICommand ConvertCommand
    {
        get { return _convertCommand ??= new RelayCommand<object>(p => ConvertOnClick(), a => ConvertEnabled); }
    }

    public ICommand BrowseCommand
    {
        get { return _browseCommand ??= new RelayCommand<object>(p=>BrowseOnClick(), a=>true); }
    }

    private void BrowseOnClick()
    {
        var openFileDialog = new OpenFileDialog();
        if(openFileDialog.ShowDialog() == true)
        {
           FileName = openFileDialog.FileName;
        }
    }

    private async void ConvertOnClick()
    {
        ConvertEnabled = false;
        ConvertInProgress = true;
        var newPath = await DicomConverter.Dicom2Nifti(FileName);
        ConvertInProgress = false;
        FileName = newPath;
    }

    private async void LoadImage()
    {
        var bitmap = await ImageUtil.LoadNiftiToImage(FileName).ConfigureAwait(false);
        ImageDisplay = bitmap;
    }
}