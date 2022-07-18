using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Windows.Input;

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

    private void StartClassificationOnClick()
    {
        // TODO
    }

    private void BrowseFrontalOnClick()
    {
        // TODO
    }
    
    private void BrowseLateralOnClick()
    {
        // TODO
    }
}