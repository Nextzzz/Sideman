using System.Windows;
using System.Windows.Controls;
using Strunika.App.ViewModels;

namespace Strunika.App;

public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel = new();

    public MainWindow()
    {
        InitializeComponent();
        DataContext = _viewModel;
    }

    private void Devices_DropDownOpened(object? sender, EventArgs e)
    {
        // Virtual mics (phone-as-mic apps) appear after startup.
        _viewModel.RefreshDevices();
    }

    private void SongList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // Keep the currently sounding chord row visible during playback.
        if (e.AddedItems.Count > 0)
            SongList.ScrollIntoView(e.AddedItems[0]);
    }

    private void SongListB_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (e.AddedItems.Count > 0)
            SongListB.ScrollIntoView(e.AddedItems[0]);
    }

    protected override void OnClosed(EventArgs e)
    {
        _viewModel.Dispose();
        base.OnClosed(e);
    }
}
