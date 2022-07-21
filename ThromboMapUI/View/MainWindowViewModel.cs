using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Microsoft.Win32;

namespace ThromboMapUI.View;

public class MainWindowViewModel : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null) {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    

    private RelayCommand<object>? _startClassificationCommand;
    private RelayCommand<object>? _browseFrontalCommand;
    private RelayCommand<object>? _browseLateralCommand;
    private string _fileNameFrontal;

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

    public string FileNameFrontal
    {
        get => _fileNameFrontal;
        private set {
            if (_fileNameFrontal != value)
            {
                _fileNameFrontal = value;
                OnPropertyChanged(nameof(FileNameFrontal));
            } 
        }
    }

    public string FileNameLateral { get; private set; }

    private void StartClassificationOnClick()
    {
        // TODO
    }

    private void BrowseFrontalOnClick()
    {
        var file = OpenFileChooser();
        if (file != null)
        {
            FileNameFrontal = file;
        }
    }
    
    private void BrowseLateralOnClick()
    {
        var file = OpenFileChooser();
        if (file != null)
        {
            FileNameLateral = file;
        }
    }

    private string? OpenFileChooser()
    {
        var openFileDialog = new OpenFileDialog();
        return openFileDialog.ShowDialog() == true ? openFileDialog.FileName : null;
    }
}