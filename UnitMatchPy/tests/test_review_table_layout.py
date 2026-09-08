from unittest.mock import Mock

from UnitMatchPy import GUI as gui


def test_review_tables_share_a_row_and_show_complete_values(monkeypatch):
    root = gui.Tk()
    root.geometry("1000x300")
    monkeypatch.setattr(gui, "root", root, raising=False)
    monkeypatch.setattr(gui, "unit_tables_frame", None)
    monkeypatch.setattr(gui, "frame_table", Mock(winfo_exists=lambda: False), raising=False)
    monkeypatch.setattr(gui, "score_table", Mock(winfo_exists=lambda: False), raising=False)
    data = [
        ["Unit", "207", "428"],
        ["Avg Centroid", "[ 1.  5000.12  270.34]", "[ 1.  5001.23  272.56]"],
        ["Amplitude", "-47.88", "-43.21"],
        ["Spatial Decay", "0.123", "0.456"],
        ["Units Matches", "207\n243", "428"],
        ["Stability", "0.998", "0.987"],
    ]
    scores = [
        ["Score", "207 and 428"],
        ["amp_score", "0.771"],
        ["spatial_decay_score", "0.884"],
        ["centroid_overlord_score", "0.456"],
        ["centroid_dist", "0.678"],
        ["waveform_score", "0.789"],
        ["trajectory_score", "0.901"],
    ]
    try:
        gui.MakeTable(data)
        gui.make_unit_score_table(scores)
        container = gui.unit_tables_frame
        root.update_idletasks()
        for frame, column, values in ((gui.frame_table, 0, data), (gui.score_table, 1, scores)):
            assert frame.master is container
            assert frame.grid_info()["row"] == 0
            assert frame.grid_info()["column"] == column
            assert frame.winfo_rooty() + frame.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            for row_index, row in enumerate(values):
                for column_index, value in enumerate(row):
                    entry, = frame.grid_slaves(row=row_index, column=column_index)
                    expected = " ".join(value.split())
                    assert entry.get() == expected
                    assert str(entry.cget("state")) == "readonly"
                    assert entry.winfo_width() >= frame._value_font.measure(expected) + 4
        assert gui.frame_table.winfo_rootx() + gui.frame_table.winfo_width() < gui.score_table.winfo_rootx()
        assert container.winfo_width() >= container.winfo_reqwidth()
        assert root.grid_columnconfigure(0)["minsize"] >= container.winfo_reqwidth() + 20
        gui.MakeTable(data)
        gui.make_unit_score_table(scores)
        assert gui.unit_tables_frame is container
        assert len(container.winfo_children()) == 2
    finally:
        root.destroy()
