using System.Windows;
using System.Windows.Controls;
using Sideman.App.ViewModels;

namespace Sideman.App;

public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel = new();

    public MainWindow()
    {
        InitializeComponent();
        DataContext = _viewModel;
    }

    private void SongList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // Keep the currently sounding chord row visible during playback.
        if (e.AddedItems.Count > 0)
            SongList.ScrollIntoView(e.AddedItems[0]);
    }

    protected override void OnClosed(EventArgs e)
    {
        _viewModel.Dispose();
        base.OnClosed(e);
    }
}
