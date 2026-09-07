import numpy as np
import pytest

from UnitMatchPy import GUI as gui


def test_gui_scale_uses_available_screen_space():
    assert gui._get_gui_scale(1920, 1080) == 1.0
    assert gui._get_gui_scale(1366, 768) == 668 / 980
    assert gui._get_gui_scale(800, 600) == gui.GUI_MIN_SCALE


def test_scaled_figure_size_uses_gui_scale(monkeypatch):
    monkeypatch.setattr(gui, "gui_scale", 0.7)

    np.testing.assert_allclose(gui._scaled_figsize(4, 6), (2.8, 4.2))


def _raw_centroids(*unit_xy):
    centroids = np.zeros((3, len(unit_xy), 2), dtype=float)
    for unit_index, (x, y) in enumerate(unit_xy):
        centroids[1, unit_index, :] = x
        centroids[2, unit_index, :] = y
    return centroids


def test_normalize_raw_centroids_copies_and_validates():
    source = _raw_centroids((1, 2), (3, 4))

    normalized = gui._normalize_raw_avg_centroid(source, n_units=2)
    normalized[1, 0, 0] = 99

    assert source[1, 0, 0] == 1
    with np.testing.assert_raises_regex(ValueError, "shape"):
        gui._normalize_raw_avg_centroid(np.zeros((2, 2, 2)), n_units=2)


def test_approved_displacement_deduplicates_and_scopes_matches():
    centroids = _raw_centroids(
        (0, 0),
        (10, 4),
        (2, 2),
        (20, 8),
        (100, 100),
        (200, 200),
    )
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1, 0, 1]),
        "probe_numbers": np.array([0, 0, 0, 0, 1, 1]),
    }
    approved_pairs = [
        [0, 1],
        [1, 0],
        [2, 3],
        [3, 2],
        [4, 5],
        [0, 2],
    ]

    result = gui._approved_displacement(
        0,
        1,
        approved_pairs,
        centroids,
        cluster_info,
    )

    assert result["count"] == 2
    np.testing.assert_allclose([result["dx"], result["dy"]], [14, 5])
    np.testing.assert_allclose(result["magnitude"], np.hypot(14, 5))


def test_approved_displacement_reverses_with_displayed_session_order():
    centroids = _raw_centroids((0, 0), (10, 4))
    cluster_info = {
        "session_id": np.array([0, 1]),
        "probe_numbers": np.array([0, 0]),
    }

    forward = gui._approved_displacement(
        0, 1, [[0, 1], [1, 0]], centroids, cluster_info
    )
    reverse = gui._approved_displacement(
        1, 0, [[0, 1], [1, 0]], centroids, cluster_info
    )

    np.testing.assert_allclose(
        [reverse["dx"], reverse["dy"]],
        [-forward["dx"], -forward["dy"]],
    )


def test_approved_displacement_does_not_use_automatic_candidates(monkeypatch):
    centroids = _raw_centroids((0, 0), (10, 4))
    cluster_info = {
        "session_id": np.array([0, 1]),
        "probe_numbers": np.array([0, 0]),
    }
    monkeypatch.setattr(
        gui,
        "automatic_match_pairs",
        {(0, 1), (1, 0)},
        raising=False,
    )

    result = gui._approved_displacement(
        0,
        1,
        [],
        centroids,
        cluster_info,
    )

    assert result["available"] is False
    assert result["message"] == "No approved cross-session matches"


def test_approved_displacement_reports_mixed_nonfinite_exclusions():
    centroids = _raw_centroids((0, 0), (4, 3), (1, 1), (7, 8))
    centroids[1, 3, 1] = np.nan
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1]),
        "probe_numbers": np.array([0, 0, 0, 0]),
    }

    result = gui._approved_displacement(
        0,
        1,
        [[0, 1], [1, 0], [2, 3], [3, 2]],
        centroids,
        cluster_info,
    )

    assert result["count"] == 1
    assert result["excluded_nonfinite"] == 1
    np.testing.assert_allclose([result["dx"], result["dy"]], [4, 3])


def test_approved_displacement_keeps_zero_vector():
    centroids = _raw_centroids((5, 6), (5, 6))
    cluster_info = {
        "session_id": np.array([0, 1]),
        "probe_numbers": np.array([0, 0]),
    }

    result = gui._approved_displacement(
        0,
        1,
        [[0, 1], [1, 0]],
        centroids,
        cluster_info,
    )

    assert result["available"] is True
    assert result["magnitude"] == 0
    np.testing.assert_array_equal([result["dx"], result["dy"]], [0, 0])


def test_approved_displacement_excludes_third_session():
    centroids = _raw_centroids((0, 0), (4, 3), (100, 100))
    cluster_info = {
        "session_id": np.array([0, 1, 2]),
        "probe_numbers": np.array([0, 0, 0]),
    }

    result = gui._approved_displacement(
        0,
        1,
        [[0, 1], [1, 0], [0, 2], [2, 0]],
        centroids,
        cluster_info,
    )

    assert result["count"] == 1
    np.testing.assert_allclose([result["dx"], result["dy"]], [4, 3])


def test_selected_displacement_reports_unapproved_pair_separately():
    centroids = _raw_centroids((0, 0), (8, -3), (2, 1), (12, 6))
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1]),
        "probe_numbers": np.array([0, 0, 0, 0]),
    }

    selected = gui._selected_displacement(
        2,
        3,
        [[0, 1], [1, 0]],
        centroids,
        cluster_info,
    )
    approved = gui._approved_displacement(
        2,
        3,
        [[0, 1], [1, 0]],
        centroids,
        cluster_info,
    )

    assert selected["approved"] is False
    np.testing.assert_allclose([selected["dx"], selected["dy"]], [10, 5])
    np.testing.assert_allclose([approved["dx"], approved["dy"]], [8, -3])


def test_selected_displacement_reverses_with_pair_order():
    centroids = _raw_centroids((2, 3), (10, -1))
    cluster_info = {
        "session_id": np.array([0, 1]),
        "probe_numbers": np.array([0, 0]),
    }

    forward = gui._selected_displacement(
        0, 1, [], centroids, cluster_info
    )
    reverse = gui._selected_displacement(
        1, 0, [], centroids, cluster_info
    )

    np.testing.assert_allclose(
        [reverse["dx"], reverse["dy"]],
        [-forward["dx"], -forward["dy"]],
    )


def test_curated_accepted_pairs_combines_automatic_manual_and_rejections():
    accepted = gui._curated_accepted_pairs(
        {(0, 1), (1, 0), (2, 3)},
        [[4, 5], [5, 4]],
        [[1, 0]],
    )

    assert {frozenset(pair) for pair in accepted} == {
        frozenset((2, 3)),
        frozenset((4, 5)),
    }


def test_automatic_displacement_anomalies_flag_only_opposite_direction():
    centroids = _raw_centroids(
        (0, 0),
        (10, 0),
        (0, 0),
        (10, 0),
        (10, 0),
        (5, 0),
    )
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1, 0, 1]),
        "probe_numbers": np.zeros(6, dtype=int),
    }
    automatic_pairs = {
        (0, 1),
        (1, 0),
        (2, 3),
        (3, 2),
        (4, 5),
        (5, 4),
    }

    result = gui._automatic_displacement_anomalies(
        automatic_pairs,
        centroids,
        cluster_info,
    )

    assert result["flagged_pairs"] == {(4, 5), (5, 4)}
    assert result["group_info"][(0, 1, 0)]["valid_count"] == 3
    assert result["group_info"][(0, 1, 0)]["flagged_count"] == 1


def test_automatic_displacement_anomalies_include_ninety_degrees():
    centroids = _raw_centroids(
        (0, 0),
        (10, -10),
        (0, 0),
        (10, 0),
        (0, 0),
        (0, 10),
    )
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1, 0, 1]),
        "probe_numbers": np.zeros(6, dtype=int),
    }

    result = gui._automatic_displacement_anomalies(
        {(0, 1), (2, 3), (4, 5)},
        centroids,
        cluster_info,
    )

    assert (4, 5) in result["flagged_pairs"]
    assert result["group_info"][(0, 1, 0)]["flagged_count"] == 1


@pytest.mark.parametrize(("angle", "flagged"), [(59.9, False), (60, False), (60.1, True)])
def test_automatic_displacement_anomalies_sixty_degree_boundary(angle, flagged):
    x, y = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    centroids = _raw_centroids((0, 0), (x, y), (0, 0), (x, -y), (0, 0), (5, 0))
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1, 0, 1]),
        "probe_numbers": np.zeros(6, dtype=int),
    }
    result = gui._automatic_displacement_anomalies(
        {(0, 1), (2, 3), (4, 5)}, centroids, cluster_info
    )
    assert ((0, 1) in result["flagged_pairs"]) == flagged
    assert ((1, 0) in result["flagged_pairs"]) == flagged


def test_automatic_displacement_anomalies_are_session_and_probe_scoped():
    centroids = _raw_centroids(
        (0, 0),
        (10, 0),
        (0, 0),
        (20, 0),
        (0, 0),
        (10, 0),
        (0, 0),
        (-20, 0),
    )
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1, 0, 2, 0, 1]),
        "probe_numbers": np.array([0, 0, 0, 0, 0, 0, 1, 1]),
    }

    result = gui._automatic_displacement_anomalies(
        {
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
        },
        centroids,
        cluster_info,
    )

    assert result["group_info"][(0, 1, 0)]["mean_defined"] is True
    assert result["group_info"][(0, 2, 0)]["mean_defined"] is True
    assert result["group_info"][(0, 1, 1)]["mean_defined"] is True
    assert result["flagged_pairs"] == set()


def test_automatic_displacement_anomalies_report_undefined_vectors():
    centroids = _raw_centroids((1, 1), (1, 1), (0, 0), (4, 3))
    centroids[1, 3, :] = np.nan
    cluster_info = {
        "session_id": np.array([0, 1, 0, 1]),
        "probe_numbers": np.zeros(4, dtype=int),
    }

    result = gui._automatic_displacement_anomalies(
        {(0, 1), (1, 0), (2, 3), (3, 2)},
        centroids,
        cluster_info,
    )

    group = result["group_info"][(0, 1, 0)]
    assert group["mean_defined"] is False
    assert group["excluded_zero"] == 1
    assert group["excluded_nonfinite"] == 1
    assert result["flagged_pairs"] == set()


def test_unusual_filter_restricts_only_unit_a_options(monkeypatch):
    monkeypatch.setattr(
        gui,
        "review_matches",
        np.array([[0, 3], [0, 4], [1, 5], [2, 4]]),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "session_switch",
        np.array([0, 3, 6]),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "output_avg",
        np.array(
            [
                [0, 0, 0, 0.9, 0.8, 0.1],
                [0, 0, 0, 0.1, 0.2, 0.95],
                [0, 0, 0, 0.1, 0.7, 0.2],
            ]
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "unusual_displacement_pairs",
        {(0, 4), (4, 0), (1, 5), (5, 1)},
        raising=False,
    )

    assert gui.get_ranked_unit_a_options(1, 2, unusual_only=True) == [
        [0, 4],
        [1, 5],
    ]
    assert gui.get_ranked_unit_b_options(0, 2) == [3, 4, 5]
    assert gui.get_ranked_unit_b_options(1, 2) == [5, 4, 3]
    assert gui.get_ranked_unit_a_options(1, 2, unusual_only=False) == [
        [0, 3],
        [1, 5],
        [2, 4],
    ]
    assert gui.get_ranked_unit_a_options(1, 2, unusual_only=True)


def test_unusual_filter_handles_no_flagged_pairs(monkeypatch):
    monkeypatch.setattr(
        gui,
        "review_matches",
        np.array([[0, 1]]),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "session_switch",
        np.array([0, 1, 2]),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "output_avg",
        np.ones((2, 2)),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "unusual_displacement_pairs",
        set(),
        raising=False,
    )

    assert gui.get_ranked_unit_a_options(1, 2, unusual_only=True) == []
    assert gui.get_ranked_unit_b_options(0, 2) == [1]
    assert gui.get_ranked_unit_a_options(1, 2, unusual_only=False) == [[0, 1]]


def test_compute_event_psth_returns_trial_averaged_firing_rate():
    centers, firing_rate, event_count = gui._compute_event_psth(
        [9.95, 10.05, 19.95, 20.05],
        [10.0, 20.0],
        before_s=0.1,
        after_s=0.1,
        bin_size_s=0.1,
    )

    np.testing.assert_allclose(centers, [-0.05, 0.05])
    np.testing.assert_allclose(firing_rate, [10.0, 10.0])
    assert event_count == 2


def test_compute_event_psth_handles_no_events():
    centers, firing_rate, event_count = gui._compute_event_psth(
        [1.0, 2.0],
        [],
        before_s=0.1,
        after_s=0.1,
        bin_size_s=0.1,
    )

    np.testing.assert_allclose(centers, [-0.05, 0.05])
    np.testing.assert_array_equal(firing_rate, [0.0, 0.0])
    assert event_count == 0


def test_normalize_event_data_sorts_and_filters_event_times():
    get_spike_times = lambda unit_index: np.array([unit_index])
    event_data = gui._normalize_event_data(
        {
            "event_times_by_session": [
                {"Stimulus": [2.0, np.nan, 1.0]},
                {},
            ],
            "get_spike_times": get_spike_times,
            "session_names": ["first", "second"],
        },
        n_sessions=2,
    )

    np.testing.assert_array_equal(
        event_data["event_times_by_session"][0]["Stimulus"],
        [1.0, 2.0],
    )
    assert event_data["get_spike_times"] is get_spike_times


def test_get_event_psth_caches_spike_times_and_histograms(monkeypatch):
    calls = []

    def get_spike_times(unit_index):
        calls.append(unit_index)
        return np.array([0.95, 1.05])

    monkeypatch.setattr(
        gui,
        "clus_info",
        {"session_indices": np.array([0])},
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "event_data",
        {
            "event_times_by_session": [{"Stimulus": np.array([1.0])}],
            "get_spike_times": get_spike_times,
        },
    )
    monkeypatch.setattr(gui, "event_spike_times_cache", {})
    monkeypatch.setattr(gui, "event_psth_cache", {})

    first = gui._get_event_psth(0, "Stimulus", 0.1, 0.1, 0.1)
    second = gui._get_event_psth(0, "Stimulus", 0.1, 0.1, 0.1)

    assert first is second
    assert calls == [0]


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


def test_pair_review_category_uses_bidirectional_better_alternatives_and_ties():
    session_ids = np.array([0, 0, 1, 0, 1, 0, 1])
    scores = np.zeros((7, 7))
    scores[0, 2] = 0.8
    scores[1, 2] = 0.9
    scores[3, 4] = 0.7
    scores[5, 6] = 0.7
    scores[0, 4] = 0.7

    assert gui._pair_review_category(
        0,
        2,
        [(0, 2), (2, 1)],
        scores,
        session_ids,
    ) == "accepted"
    assert gui._pair_review_category(
        0,
        2,
        [(2, 1)],
        scores,
        session_ids,
    ) == "better_alternative"
    assert gui._pair_review_category(
        3,
        4,
        [(5, 6)],
        scores,
        session_ids,
    ) == "no_alternative"
    assert gui._pair_review_category(
        3,
        4,
        [(4, 0)],
        scores,
        session_ids,
    ) == "better_alternative"


def test_color_unit_a_options_shows_all_three_states(monkeypatch):
    calls = []

    class Entry:
        @staticmethod
        def set_review_status(unit, category):
            calls.append((unit, category))

    scores = np.zeros((5, 5))
    scores[0, 2] = 0.9
    scores[1, 2] = 0.8
    monkeypatch.setattr(gui, "entry_a", Entry(), raising=False)
    monkeypatch.setattr(
        gui,
        "option_a",
        [[0, 2], [1, 2], [3, 4]],
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "automatic_match_pairs",
        {(0, 2), (2, 0)},
        raising=False,
    )
    monkeypatch.setattr(gui, "is_match", [], raising=False)
    monkeypatch.setattr(gui, "not_match", [], raising=False)
    monkeypatch.setattr(gui, "output_avg", scores, raising=False)
    monkeypatch.setattr(
        gui,
        "clus_info",
        {"session_id": np.array([0, 0, 1, 0, 1])},
        raising=False,
    )

    gui.color_unit_a_options()

    assert calls == [
        (0, "accepted"),
        (1, "better_alternative"),
        (3, "no_alternative"),
    ]
    gui.not_match[:] = [[0, 2], [2, 0]]
    calls.clear()
    gui.color_unit_a_options()
    assert calls == [(0, "no_alternative"), (1, "no_alternative"), (3, "no_alternative")]


def test_unit_a_table_selection_colors_and_filter_refresh():
    root = gui.Tk()
    root.withdraw()
    try:
        table = gui.UnitATable(root, ["0 (1) 2", "1 (1) 2", "3 (0) 4"])
        events = []
        table.bind("<<UnitASelected>>", lambda event: events.append(table.get()))
        table.set(0)
        root.update()
        assert events == []
        table.table.selection_set("1")
        root.update()
        assert events == ["1"]
        assert table.get() == "1"
        assert table._metric_pairs == [("0", 0, 2), ("1", 1, 2), ("3", 3, 4)]
        for unit, category in ((0, "accepted"), (1, "better_alternative"), (3, "no_alternative")):
            table.set_review_status(unit, category)
            assert table.table.item(str(unit), "tags") == (category,)
        assert table.table.set("1", "status") == "Better match exists"
        assert str(table.table.tag_configure("accepted", "foreground")) == gui.APPROVED_MATCH_COLOR
        assert gui.ttk.Style(root).map("UnitA.Treeview", "foreground") == []
        table.set_options([])
        table.set("")
        root.update()
        assert table.get() == ""
        assert table.table.get_children() == ()
        table.set_options(["7 (2) 9"])
        table.set(7)
        root.update()
        assert table.get() == "7"
        assert table._metric_pairs == [("7", 7, 9)]
        assert events == ["1"]
    finally:
        root.destroy()


def test_unit_a_diagnostics_use_each_rows_partner(monkeypatch):
    monkeypatch.setattr(gui, "raw_avg_centroid", _raw_centroids((0, 0), (1, 1), (3, 4), (4, 1)))
    monkeypatch.setattr(gui, "clus_info", {
        "session_id": np.array([0, 0, 1, 1]), "probe_numbers": np.zeros(4, dtype=int),
    }, raising=False)
    monkeypatch.setattr(gui, "unusual_displacement_group_info", {
        (0, 1, 0): {"mean_defined": True, "mean_vector": np.array([1., 0.])},
    })
    monkeypatch.setattr(gui, "event_data", None)
    root = gui.Tk()
    root.withdraw()
    try:
        table = gui.UnitATable(root, ["0 (1) 2", "1 (1) 3"])
        table.set(0)
        assert table.table.set("0", "distance") == "5.0"
        assert table.table.set("0", "angle") == "53.1"
        assert table.table.set("1", "distance") == "3.0"
        assert table.table.set("1", "angle") == "0.0"
        assert table.table.set("1", "event_r") == "n/a"
    finally:
        root.destroy()


def test_reopening_review_focuses_existing_window_without_reset(monkeypatch):
    from unittest.mock import Mock

    existing = Mock()
    existing.winfo_exists.return_value = True
    monkeypatch.setattr(gui, "root", existing, raising=False)
    monkeypatch.setattr(gui, "is_match", [[0, 1]])
    monkeypatch.setattr(gui, "not_match", [[2, 3]])
    monkeypatch.setattr(gui, "matches_GUI", ["existing"], raising=False)
    assert gui.run_GUI() == ([[0, 1]], [[2, 3]], ["existing"])
    existing.deiconify.assert_called_once()
    existing.lift.assert_called_once()


def test_histogram_limits_preserve_peaks_and_expand_dominant_spike():
    from matplotlib.figure import Figure

    normal, spiked, empty = Figure().subplots(1, 3)
    gui._set_histogram_y_limits(normal, (np.array([1., 2., 3.]), None))
    gui._set_histogram_y_limits(
        spiked, (np.r_[95., np.full(99, 0.05)], None),
        (np.array([0., 1., 5., 10.]), None),
    )
    gui._set_histogram_y_limits(empty, (np.array([0., np.nan]), None))
    assert normal.get_yscale() == "linear"
    np.testing.assert_allclose(normal.get_ylim(), (0, 3.24))
    assert spiked.get_yscale() == "symlog"
    assert spiked.get_ylim()[1] > 95
    assert spiked.get_ylabel() == "Density (symlog)"
    assert empty.get_ylim() == (0, 1)


def test_candidate_displacement_angle_and_rays_use_filter_reference(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(gui, "raw_avg_centroid", _raw_centroids((0, 0), (3, 4)))
    monkeypatch.setattr(gui, "clus_info", {
        "session_id": np.array([0, 1]), "probe_numbers": np.array([0, 0]),
    }, raising=False)
    monkeypatch.setattr(gui, "unusual_displacement_group_info", {
        (0, 1, 0): {"mean_defined": True, "mean_vector": np.array([1., 0.])},
    })
    monkeypatch.setattr(gui, "automatic_match_pairs", {(0, 1), (1, 0)})
    monkeypatch.setattr(gui, "is_match", [])
    monkeypatch.setattr(gui, "not_match", [])
    np.testing.assert_allclose(gui._pair_displacement_metrics(0, 1), (5, 53.1301023542))
    np.testing.assert_allclose(gui._pair_displacement_metrics(1, 0), (5, 53.1301023542))
    for a, b, sign in ((0, 1, 1), (1, 0, -1)):
        axis = gui._add_raw_displacement_overlay(Figure(), a, b)
        rays = [line for line in axis.lines if str(line.get_gid()).startswith("displacement_threshold_")]
        assert len(rays) == 2
        for line in rays:
            vector = np.array([line.get_xdata()[-1], line.get_ydata()[-1]])
            np.testing.assert_allclose(sign * vector[0] / np.linalg.norm(vector), 0.5)
            assert line.get_linestyle() == "--"
    gui.raw_avg_centroid[:, 1, :] = gui.raw_avg_centroid[:, 0, :]
    assert gui._pair_displacement_metrics(0, 1) == (0., None)


def test_event_response_correlation_averages_shared_profiles_and_caches():
    settings = (1., 2., 0.01)
    data = {
        "event_times_by_session": [
            {"cue": [1, 2], "reward": [1, 2], "only_a": [1, 2]},
            {"cue": [1, 2], "reward": [1, 2]},
        ],
        "get_spike_times": lambda unit: np.array([]),
    }
    profiles = {
        (0, "cue", *settings): (None, np.array([0., 1., 0., 2.]), 2),
        (1, "cue", *settings): (None, np.array([0., 2., 0., 4.]), 2),
        (0, "reward", *settings): (None, np.array([0., 1., 2., 3.]), 2),
        (1, "reward", *settings): (None, np.array([3., 2., 1., 0.]), 2),
    }
    context = (data, np.array([0, 1]), {0: [], 1: []}, profiles, {})
    np.testing.assert_allclose(gui._event_response_correlation(0, 1, settings, context), 0, atol=1e-12)
    np.testing.assert_allclose(gui._event_response_correlation(1, 0, settings, context), 0, atol=1e-12)
    assert len(context[-1]) == 1
    context[-1].clear()
    profiles[(1, "reward", *settings)] = (None, np.ones(4), 2)
    np.testing.assert_allclose(gui._event_response_correlation(0, 1, settings, context), 1)
    context[-1].clear()
    data["event_times_by_session"][1]["cue"] = [1]
    assert gui._event_response_correlation(0, 1, settings, context) is None
    assert gui._event_response_correlation(0, 1, (0.5, 1., 0.05), context) is None


def test_event_correlation_worker_cancels_and_reports_data_errors():
    from queue import Queue
    from threading import Event

    results, cancelled = Queue(), Event()
    cancelled.set()
    gui._compute_candidate_event_correlations([(1, 0, 1)], (1., 2., .01), None, cancelled, results)
    assert results.empty()
    cancelled.clear()

    def missing_spikes(unit):
        raise OSError("Spike data unavailable")

    context = (
        {"event_times_by_session": [{"cue": [1, 2]}, {"cue": [1, 2]}],
         "get_spike_times": missing_spikes},
        np.array([0, 1]), {}, {}, {},
    )
    gui._compute_candidate_event_correlations([(1, 0, 1)], (1., 2., .01), context, cancelled, results)
    assert results.get_nowait() == (1, None, "Spike data unavailable")


def test_histogram_legend_has_visible_matching_line_handles(monkeypatch):
    class Widget:
        pass

    class Canvas:
        def __init__(self, figure, master):
            del master
            self.figure = figure
            self.widget = Widget()
            self.widget.figure = figure

        @staticmethod
        def draw():
            pass

        def get_tk_widget(self):
            return self.widget

    monkeypatch.setattr(gui, "FigureCanvasTkAgg", Canvas)
    monkeypatch.setattr(gui, "root", object(), raising=False)

    gui.create_hist_legend()

    figure = gui.hist_legend_plot.figure
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "All scores",
        "Expected matches",
        "Current pair",
    ]
    handles = legend.legend_handles
    assert [handle.get_color() for handle in handles] == [
        gui.ALL_SCORES_COLOR,
        gui.APPROVED_MATCH_COLOR,
        "white",
    ]
    assert handles[-1].get_linestyle() == "--"


@pytest.mark.parametrize(
    ("channel_count", "waveform_values"),
    [
        (1, (1, 3, 8, 4, 2)),
        (5, (-9, -4, -2, -1, -0.5)),
        (10, (-1, 0, 1, 2, 9)),
        (15, (-1, -0.5, 0, 0.5, 1)),
    ],
)
def test_plot_raw_waveforms_stay_below_displacement_panel(
    monkeypatch,
    channel_count,
    waveform_values,
):
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

        @staticmethod
        def draw_idle():
            pass

        def get_tk_widget(self):
            return self.widget

    channel_pos = np.zeros((1, channel_count, 3))
    row_count = int(np.ceil(channel_count / 2))
    channel_pos[0, :, 1] = np.tile([0, 20], row_count)[:channel_count]
    channel_pos[0, :, 2] = np.repeat(
        np.arange(row_count),
        2,
    )[:channel_count]
    monkeypatch.setattr(gui, "raw_waveform_plot", Widget(), raising=False)
    monkeypatch.setattr(gui, "FigureCanvasTkAgg", Canvas)
    monkeypatch.setattr(gui, "root", object(), raising=False)
    monkeypatch.setattr(
        gui,
        "clus_info",
        {
            "session_id": np.array([0, 0]),
            "probe_numbers": np.array([0, 0]),
        },
        raising=False,
    )
    monkeypatch.setattr(gui, "is_match", [], raising=False)
    monkeypatch.setattr(gui, "raw_avg_centroid", None, raising=False)
    monkeypatch.setattr(gui, "channel_pos", channel_pos, raising=False)
    monkeypatch.setattr(gui, "max_site", np.array([[0, 0], [0, 0]]), raising=False)
    monkeypatch.setattr(gui, "max_site_mean", np.array([0, 0]), raising=False)
    monkeypatch.setattr(
        gui,
        "waveform",
        np.broadcast_to(
            np.asarray(waveform_values)[None, :, None, None],
            (2, 5, channel_count, 2),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "nearest_channels",
        lambda *args: np.arange(channel_count),
    )

    gui.plot_raw_waveforms(0, 1, "Avg")

    assert len(gui.raw_waveform_plot.figure.axes) == channel_count + 2
    main_bounds = gui.raw_waveform_plot.figure.axes[0].get_position().bounds
    np.testing.assert_allclose(main_bounds, gui.RAW_WAVEFORM_AXES_BOUNDS)
    assert all(
        axis.get_position().y1 <= 0.600001
        for axis in gui.raw_waveform_plot.figure.axes[1:1 + channel_count]
    )
    displacement_bounds = gui.raw_displacement_axis.get_position(original=True).bounds
    np.testing.assert_allclose(
        displacement_bounds,
        gui.RAW_DISPLACEMENT_AXES_BOUNDS,
    )
    assert displacement_bounds[2] * displacement_bounds[3] > 3 * (
        0.27 * 0.17
    )


def test_displacement_overlay_uses_shared_physical_scale(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(
        gui,
        "clus_info",
        {
            "session_id": np.array([0, 1, 0, 1]),
            "probe_numbers": np.array([0, 0, 0, 0]),
        },
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "raw_avg_centroid",
        _raw_centroids((0, 0), (4, 3), (1, 1), (11, 1)),
        raising=False,
    )
    monkeypatch.setattr(gui, "is_match", [[0, 1], [1, 0]], raising=False)

    axis = gui._add_raw_displacement_overlay(Figure(), 2, 3)

    np.testing.assert_allclose(axis.get_xlim(), [-12.5, 12.5])
    np.testing.assert_allclose(axis.get_ylim(), [-12.5, 12.5])
    assert axis.get_aspect() == 1.0
    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_text == [
        "Mean displacement",
        "Selected unit displacement",
    ]
    status = next(
        text.get_text()
        for text in axis.texts
        if text.get_gid() == "displacement_status"
    )
    assert status == (
        "Mean: 5.0 um (n=1)\n"
        "Selected: 10.0 um (not approved)"
    )


def test_displacement_overlay_uses_automatic_acceptance_for_mean(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(
        gui,
        "clus_info",
        {
            "session_id": np.array([0, 1]),
            "probe_numbers": np.array([0, 0]),
        },
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "raw_avg_centroid",
        _raw_centroids((0, 0), (4, 3)),
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "automatic_match_pairs",
        {(0, 1), (1, 0)},
        raising=False,
    )
    monkeypatch.setattr(gui, "is_match", [], raising=False)
    monkeypatch.setattr(gui, "not_match", [], raising=False)

    axis = gui._add_raw_displacement_overlay(Figure(), 0, 1)

    status = next(
        text.get_text()
        for text in axis.texts
        if text.get_gid() == "displacement_status"
    )
    assert "Mean: 5.0 um (n=1)" in status
    assert "Selected: 5.0 um (approved)" in status


def test_displacement_overlay_reports_nonfinite_exclusions(monkeypatch):
    from matplotlib.figure import Figure

    centroids = _raw_centroids((0, 0), (4, 3), (1, 1), (7, 8))
    centroids[2, 3, 0] = np.nan
    monkeypatch.setattr(
        gui,
        "clus_info",
        {
            "session_id": np.array([0, 1, 0, 1]),
            "probe_numbers": np.array([0, 0, 0, 0]),
        },
        raising=False,
    )
    monkeypatch.setattr(gui, "raw_avg_centroid", centroids, raising=False)
    monkeypatch.setattr(
        gui,
        "is_match",
        [[0, 1], [1, 0], [2, 3], [3, 2]],
        raising=False,
    )

    axis = gui._add_raw_displacement_overlay(Figure(), 0, 1)

    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_text == [
        "Mean displacement",
        "Selected unit displacement",
    ]
    status = next(
        text.get_text()
        for text in axis.texts
        if text.get_gid() == "displacement_status"
    )
    assert "Mean: 5.0 um (n=1); excluded=1" in status


def test_identical_displacement_arrows_remain_distinct(monkeypatch):
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from matplotlib.text import Annotation

    monkeypatch.setattr(
        gui,
        "clus_info",
        {
            "session_id": np.array([0, 1]),
            "probe_numbers": np.array([0, 0]),
        },
        raising=False,
    )
    monkeypatch.setattr(
        gui,
        "raw_avg_centroid",
        _raw_centroids((0, 0), (4, 3)),
        raising=False,
    )
    monkeypatch.setattr(gui, "is_match", [[0, 1], [1, 0]], raising=False)

    axis = gui._add_raw_displacement_overlay(Figure(), 0, 1)

    arrows = [
        text.arrow_patch
        for text in axis.texts
        if isinstance(text, Annotation) and text.arrow_patch is not None
    ]
    assert len(arrows) == 2
    assert [arrow.get_edgecolor() for arrow in arrows] == [
        to_rgba(gui.APPROVED_MATCH_COLOR),
        to_rgba("#42A5F5"),
    ]
    assert arrows[0].get_linewidth() > arrows[1].get_linewidth()
    assert arrows[0].get_linestyle() in {"-", "solid"}
    assert arrows[1].get_linestyle() in {"--", "dashed"}


def test_manual_decisions_refresh_displacement_overlay(monkeypatch):
    class Entry:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class CV:
        @staticmethod
        def get():
            return 1

    redraws = []
    monkeypatch.setattr(gui, "entry_a", Entry("2"), raising=False)
    monkeypatch.setattr(gui, "entry_b", Entry("3"), raising=False)
    monkeypatch.setattr(gui, "CV_tkinter", CV(), raising=False)
    monkeypatch.setattr(gui, "is_match", [], raising=False)
    monkeypatch.setattr(gui, "not_match", [], raising=False)
    monkeypatch.setattr(gui, "add_probability_label", lambda *args: None)
    monkeypatch.setattr(gui, "color_unit_a_options", lambda: None)
    monkeypatch.setattr(
        gui,
        "_refresh_raw_displacement_overlay",
        lambda: redraws.append(True),
    )

    gui.set_match()
    gui.set_not_match()

    assert redraws == [True, True]
