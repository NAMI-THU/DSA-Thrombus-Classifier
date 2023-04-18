using System.ComponentModel;
using System.Configuration;

namespace UI.Util;

public class UserPersistence : ApplicationSettingsBase
{
    public static UserPersistence Default { get; } = (UserPersistence)Synchronized(new UserPersistence());

    protected override void OnPropertyChanged(object sender, PropertyChangedEventArgs e)
    {
        Save();
        base.OnPropertyChanged(sender, e);
    }

    [UserScopedSetting]
    public string ModelPath {
        get => (string)this[nameof(ModelPath)];
        set => this[nameof(ModelPath)] = value;
    }
}