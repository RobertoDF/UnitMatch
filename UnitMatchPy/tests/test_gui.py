import numpy as np

from UnitMatchPy import GUI as gui


def test_order_good_sites_handles_sixteen_channels():
    channel_pos = np.zeros((1, 16, 3))
    channel_pos[0, :, 1] = np.tile([0, 20], 8)
    channel_pos[0, :, 2] = np.repeat(np.arange(8), 2)
    good_sites = np.arange(15, -1, -1).reshape(-1, 1)

    reordered = gui.order_good_sites(good_sites, channel_pos, 0)

    np.testing.assert_array_equal(
        reordered,
        [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
    )


def test_plot_raw_waveforms_handles_an_odd_channel_count(monkeypatch):
    class Widget:
        @staticmethod
        def winfo_exists():
            return 0

        @staticmethod
        def grid(**kwargs):
            pass

    class Canvas:
        def __init__(self, figure, master):
            del master
            self.widget = Widget()
            self.widget.figure = figure

        @staticmethod
        def draw():
            pass

        def get_tk_widget(self):
            return self.widget

    channel_pos = np.zeros((1, 15, 3))
    channel_pos[0, :, 1] = np.tile([0, 20], 8)[:15]
    channel_pos[0, :, 2] = np.repeat(np.arange(8), 2)[:15]
    monkeypatch.setattr(gui, "raw_waveform_plot", Widget(), raising=False)
    monkeypatch.setattr(gui, "FigureCanvasTkAgg", Canvas)
    monkeypatch.setattr(gui, "root", object(), raising=False)
    monkeypatch.setattr(gui, "clus_info", {"session_id": np.array([0, 0])}, raising=False)
    monkeypatch.setattr(gui, "channel_pos", channel_pos, raising=False)
    monkeypatch.setattr(gui, "max_site", np.array([[0, 0], [0, 0]]), raising=False)
    monkeypatch.setattr(gui, "max_site_mean", np.array([0, 0]), raising=False)
    monkeypatch.setattr(
        gui,
        "waveform",
        np.broadcast_to(
            np.linspace(-1, 1, 5)[None, :, None, None],
            (2, 5, 15, 2),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "nearest_channels",
        lambda *args: np.arange(15),
    )

    gui.plot_raw_waveforms(0, 1, "Avg")

    assert len(gui.raw_waveform_plot.figure.axes) == 16
