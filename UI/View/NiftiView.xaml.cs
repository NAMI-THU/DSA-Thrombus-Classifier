﻿using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using Services.FileUtils;
using UI.Util;
using Brush = System.Windows.Media.Brush;
using Brushes = System.Windows.Media.Brushes;

namespace UI.View;

public partial class NiftiView : UserControl, INotifyPropertyChanged
{
    private RelayCommand<object>? _browseCommand;
    private RelayCommand<object>? _convertCommand;
    private RelayCommand<float[][]>? _changeImageCommand;
    private RelayCommand<object>? _resetFilesCommand;
    private bool _convertEnabled;
    private bool _convertInProgress;
    private string _fileName = "";
    private bool _filePrepared;
    private Brush _headerColor = Brushes.DarkRed;
    private PackIconKind _headerIcon = PackIconKind.Error;
    private ImageSource? _imageDisplay;
    private string _imageName = "";

    public NiftiView()
    {
        InitializeComponent();
    }

    public ICommand? FilePreparedNotificationCommand { get; set; }
    public ICommand? ResetFilesOtherCommand { get; set; }

    public ImageSource? ImageDisplay {
        get => _imageDisplay;
        private set {
            _imageDisplay = value;
            OnPropertyChanged();
        }
    }

    public string FileName {
        get => _fileName;
        private set {
            _fileName = value;

            ConvertEnabled = false;
            FilePrepared = false;

            if (_fileName.EndsWith(".nii")) {
                FilePrepared = true;
            }
            else if (!string.IsNullOrEmpty(_fileName)) {
                ConvertEnabled = true;
            }

            OnPropertyChanged();
            OnPropertyChanged(nameof(ConvertEnabled));
            OnPropertyChanged(nameof(FilePrepared));
        }
    }

    public bool ConvertInProgress {
        get => _convertInProgress;
        private set {
            if (_convertInProgress == value) return;
            _convertInProgress = value;
            OnPropertyChanged();
        }
    }

    public bool ConvertEnabled {
        get => _convertEnabled;
        private set {
            if (_convertEnabled == value) return;
            _convertEnabled = value;
            OnPropertyChanged();
        }
    }

    public bool FilePrepared {
        get => _filePrepared;
        private set {
            _filePrepared = value;

            if (_filePrepared) {
                HeaderColor = new SolidColorBrush(new PaletteHelper().GetTheme().PrimaryDark.Color);
                HeaderIcon = PackIconKind.Check;
                FilePreparedNotificationCommand?.Execute(FileName);
            }
            else {
                HeaderColor = Brushes.DarkRed;
                HeaderIcon = PackIconKind.Error;
                FilePreparedNotificationCommand?.Execute(null);
            }

            OnPropertyChanged();
        }
    }

    public PackIconKind HeaderIcon {
        get => _headerIcon;
        set {
            if (Equals(value, _headerIcon)) return;
            _headerIcon = value;
            OnPropertyChanged();
        }
    }

    public Brush HeaderColor {
        get => _headerColor;
        set {
            if (Equals(value, _headerColor)) return;
            _headerColor = value;
            OnPropertyChanged();
        }
    }

    public string ImageName {
        get => _imageName;
        set {
            if (value == _imageName) return;
            _imageName = value;
            OnPropertyChanged();
        }
    }

    public ICommand ConvertCommand {
        get { return _convertCommand ??= new RelayCommand<object>(_ => ConvertOnClick(), _ => ConvertEnabled); }
    }

    public ICommand BrowseCommand {
        get { return _browseCommand ??= new RelayCommand<object>(_ => BrowseOnClick()); }
    }

    public ICommand ChangeImageCommand {
        get { return _changeImageCommand ??= new RelayCommand<float[][]>(array => ImageInjected(array)); }
    }

    public ICommand ResetFilesCommand {
        get {
            return _resetFilesCommand ??= new RelayCommand<object>(_ => {
                FileName = "";
                ImageDisplay = null;
            });
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    private void BrowseOnClick()
    {
        if (FilePrepared) {
            ResetFilesCommand.Execute(null); // Remove Image
            ResetFilesOtherCommand?.Execute(null);
        }

        var openFileDialog = new OpenFileDialog();
        if (openFileDialog.ShowDialog() == true) {
            FileName = openFileDialog.FileName;
        }
    }

    private async void ConvertOnClick()
    {
        ConvertEnabled = false;
        ConvertInProgress = true;
        try {
            var newPath = await DicomConverter.Dicom2Nifti(FileName);
            FileName = newPath;
        }
        catch (ArgumentException) {
            // Conversion was not possible
            MessageBox.Show("Conversion of the input file failed. Please make sure it is in the correct format.",
                "Conversion failed", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally {
            ConvertInProgress = false;
        }
    }

    private async void ImageInjected(float[][] array)
    {
        ImageDisplay = await ImageUtil.Array2Image(array).ConfigureAwait(false);
    }
}