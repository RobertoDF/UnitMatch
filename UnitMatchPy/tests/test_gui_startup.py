import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch
import warnings

import numpy as np

from UnitMatchPy import GUI as gui


def gui_inputs():
    rng = np.random.default_rng(17)
    n_units, n_time = 8, 10
    output = rng.random((n_units, n_units))
    np.fill_diagonal(output, 0.99)
    output[0, 1] = 0.8
    output[1, 0] = 0.7
    sessions = np.repeat([0, 1], n_units // 2)
    return dict(
        output=output,
        match_threshold_in=0.8,
        scores_to_include={"amplitude": rng.random(output.shape)},
        total_score=rng.random(output.shape),
        amplitude_in=rng.random((n_units, 2)),
        spatial_decay_in=rng.random((n_units, 2)),
        avg_centroid_in=rng.random((3, n_units, 2)),
        avg_waveform_in=rng.random((n_time, n_units, 2)),
        avg_waveform_per_tp_in=rng.random((3, n_units, n_time, 2)),
        wave_idx_in=np.ones((n_units, n_time, 2), dtype=bool),
        max_site_in=np.zeros((n_units, 2), dtype=int),
        max_site_mean_in=np.zeros(n_units, dtype=int),
        waveform_in=rng.random((n_units, n_time, 16, 2)),
        within_session_in=sessions[:, None] == sessions[None, :],
        channel_pos_in=np.zeros((2, 16, 3)),
        clus_info_in={
            "session_id": sessions,
            "session_switch": np.array([0, 4, 8]),
            "original_ids": np.arange(n_units),
            "probe_numbers": np.zeros(n_units, dtype=int),
        },
        param_in={"n_sessions": 2},
    )


class GuiStartupTests(unittest.TestCase):
    def setUp(self):
        self.inputs = gui_inputs()
        gui.process_info_for_GUI(**self.inputs)

    def test_review_colors_use_average_probability_not_directional_score(self):
        output = self.inputs["output"]
        output[0, 4], output[4, 0] = 0.894, 0.325
        output[1, 4], output[4, 1] = 0.662, 0.828
        self.inputs["total_score"][0, 4] = 0.867
        self.inputs["total_score"][1, 4] = 0.802
        gui.process_info_for_GUI(**self.inputs, automatic_matches_in=[[1, 4]])
        with patch.object(gui, "is_match", []), patch.object(gui, "not_match", []):
            self.assertEqual(gui._unit_a_review_category(0, 4), "better_alternative")
            self.assertEqual(gui._unit_a_review_category(4, 0), "better_alternative")

    def test_unit_b_table_uses_automatic_threshold_rule_and_preserves_selection(self):
        output = self.inputs["output"]
        output[0, 4:8] = [0.9, 0.9, 0.8, np.nan]
        output[4:8, 0] = [0.1, 0.85, 0.1, 0.9]
        root = gui.Tk()
        root.withdraw()
        try:
            for mode, eligible in (("or", {4, 5, 7}), ("and", {5})):
                with self.subTest(mode=mode):
                    gui.process_info_for_GUI(**self.inputs, match_mode=mode)
                    with patch.object(gui, "entry_a", Mock(get=lambda: "0"), create=True):
                        table = gui.UnitBTable(root, [4, 5, 6, 7])
                        events = []
                        table.bind("<<UnitBSelected>>", lambda event: events.append(table.get()))
                        table.current(0)
                        root.update()
                        self.assertEqual(events, [])
                        for unit in (4, 5, 6, 7):
                            category = "above_threshold" if unit in eligible else "below_threshold"
                            self.assertEqual(table.table.item(str(unit), "tags"), (category,))
                        self.assertEqual(table.table.set("4", "cv12"), "0.900")
                        self.assertEqual(table.table.set("4", "cv21"), "0.100")
                        table.table.selection_set("5")
                        root.update()
                        self.assertEqual(events, ["5"])
                        self.assertEqual(table.get(), "5")
                        table.set_options([])
                        table.set("")
                        root.update()
                        self.assertEqual(table.get(), "")
                        table.destroy()
        finally:
            root.destroy()

    def assert_histograms_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        for first_group, second_group in zip(first[1:], second[1:]):
            for first_hist, second_hist in zip(first_group, second_group):
                for first_array, second_array in zip(first_hist, second_hist):
                    np.testing.assert_array_equal(first_array, second_array)

    def test_preparation_preserves_pairs_scores_and_order(self):
        output = self.inputs["output"]
        output[2, 5] = np.nan
        output[5, 2] = np.inf
        for mode in ("and", "or"):
            for approved in (None, [[0, 4], [1, 5]]):
                with self.subTest(mode=mode, approved=approved):
                    gui.process_info_for_GUI(
                        **self.inputs, match_mode=mode, automatic_matches_in=approved
                    )
                    avg = (output + output.T) / 2
                    np.testing.assert_array_equal(gui.output_avg, avg)
                    for index, matrix in enumerate((output, output.T)):
                        np.testing.assert_array_equal(
                            gui.matches_GUI[index], np.argwhere(matrix > 0.8)
                        )
                    legacy_avg_pairs = np.unique(
                        np.concatenate((
                            np.argwhere(avg > 0.8), np.argwhere(avg.T > 0.8)
                        )), axis=0
                    )
                    np.testing.assert_array_equal(gui.matches_avg, legacy_avg_pairs)
                    threshold = max(0, 0.8 - 0.1)
                    review_pairs = np.argwhere(
                        (output >= threshold) | (output.T >= threshold)
                    )
                    order = np.argsort(
                        -avg[review_pairs[:, 0], review_pairs[:, 1]], kind="stable"
                    )
                    np.testing.assert_array_equal(gui.review_matches, review_pairs[order])
                    mask = (
                        (output > 0.8) & (output.T > 0.8)
                        if mode == "and"
                        else (output > 0.8) | (output.T > 0.8)
                    )
                    self.assertEqual(
                        gui.automatic_candidate_pairs, set(map(tuple, np.argwhere(mask)))
                    )
                    expected = np.argwhere(mask) if approved is None else approved
                    self.assertEqual(
                        gui.automatic_match_pairs,
                        {tuple(pair) for pair in expected}
                        | {tuple(reversed(pair)) for pair in expected},
                    )
                    for name, values in self.inputs["scores_to_include"].items():
                        np.testing.assert_array_equal(
                            gui.scores_to_include_avg[name], (values + values.T) / 2
                        )

    def test_histograms_preserve_bins_density_and_reciprocal_distribution(self):
        for dtype in (np.float32, np.float64):
            values = self.inputs["scores_to_include"]["amplitude"].astype(dtype)
            values[0, :3] = [0, 0.5, 1]
            for mask in (
                self.inputs["output"] > 0.8,
                np.zeros(values.shape, dtype=bool),
                np.ones(values.shape, dtype=bool),
            ):
                with self.subTest(dtype=dtype, selected=mask.sum()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        expected = (
                            ["amplitude"], [np.histogram(values, bins=100, density=True)],
                            [np.histogram(values[mask], bins=100, density=True)],
                        )
                        actual = gui.get_score_histograms(
                            {"amplitude": values}, mask.astype(float)
                        )
                        reciprocal = gui.get_score_histograms(
                            {"amplitude": values.T}, mask.T
                        )
                    self.assert_histograms_equal(actual, expected)
                    self.assert_histograms_equal(reciprocal, expected)

    def test_histograms_are_lazy_and_reused_across_reciprocal_cv(self):
        self.assertEqual(gui.score_histogram_cache, {})
        with patch.object(
            gui, "get_score_histograms", wraps=gui.get_score_histograms
        ) as compute:
            average = gui._get_score_histograms_for_cv(-1)
            self.assertIs(gui._get_score_histograms_for_cv(-1), average)
            self.assertEqual(compute.call_count, 1)
            reciprocal = gui._get_score_histograms_for_cv(1)
            self.assertIs(gui._get_score_histograms_for_cv(0), reciprocal)
            self.assertEqual(compute.call_count, 2)
        self.assert_histograms_equal(
            reciprocal,
            gui.get_score_histograms(
                gui.scores_to_include_GUI[1], gui.output_GUI[1] > gui.match_threshold
            ),
        )

    def test_new_preparation_invalidates_histograms_and_event_caches(self):
        old = gui._get_score_histograms_for_cv(-1)
        gui.event_psth_cache["old"] = object()
        gui.event_spike_times_cache["old"] = object()
        self.inputs["scores_to_include"]["amplitude"] += 2
        gui.process_info_for_GUI(**self.inputs)
        self.assertEqual(gui.score_histogram_cache, {})
        self.assertEqual(gui.event_psth_cache, {})
        self.assertEqual(gui.event_spike_times_cache, {})
        new = gui._get_score_histograms_for_cv(-1)
        self.assertIsNot(old, new)
        np.testing.assert_allclose(new[1][0][1], old[1][0][1] + 2)

    def test_empty_average_matches_remain_two_column_array(self):
        self.inputs["output"].fill(0)
        gui.process_info_for_GUI(**self.inputs)
        self.assertEqual(gui.matches_avg.shape, (0, 2))

    def test_unit_table_uses_equivalent_row_local_multiplication(self):
        class RowOnly(np.ndarray):
            def __mul__(self, other):
                if self.ndim != 1:
                    raise AssertionError("Must select a row before multiplying")
                return super().__mul__(other)

        mask = self.inputs["within_session_in"]
        with patch.object(gui, "within_session", mask.view(RowOnly)):
            for cv, matrix in (
                ("Avg", gui.output_avg), ([0, 1], gui.output_GUI[0]),
                ([1, 0], gui.output_GUI[1]),
            ):
                table = gui.get_table_data(0, 5, cv)
                for column, unit in enumerate((0, 5), start=1):
                    expected = str(
                        np.argwhere((mask * matrix)[unit] > gui.match_threshold)
                    ).replace("[", "").replace("]", "")
                    self.assertEqual(table[4][column], expected)

    def test_hidden_histograms_do_no_work_and_visible_cv_uses_own_scores(self):
        with ExitStack() as stack:
            for name in (
                "MakeTable", "make_unit_score_table", "plot_avg_waveforms",
                "plot_trajectories", "plot_raw_waveforms", "add_probability_label",
                "add_original_ID", "plot_acgs", "color_unit_a_options",
                "install_navigation_bindtags",
            ):
                stack.enter_context(patch.object(gui, name))
            stack.enter_context(patch.object(gui, "root", Mock(), create=True))
            stack.enter_context(patch.object(gui, "event_view_refresh", None))
            for name, value in (
                ("entry_a", "0"), ("entry_b", "5"),
                ("toggle_raw_val", False), ("toggle_acg_val", False),
            ):
                stack.enter_context(patch.object(
                    gui, name, Mock(get=Mock(return_value=value)), create=True
                ))
            toggle = stack.enter_context(patch.object(
                gui, "toggle_UM_score_val", Mock(), create=True
            ))
            stack.enter_context(patch.object(gui, "hist_plot", Mock(), create=True))
            cv_variable = stack.enter_context(patch.object(
                gui, "CV_tkinter", Mock(), create=True
            ))
            plot = stack.enter_context(patch.object(gui, "plot_histograms"))
            for cv_value in (0, 1, 2):
                cv_variable.get.return_value = cv_value
                toggle.get.return_value = True
                gui.update(None)
            self.assertEqual(gui.score_histogram_cache, {})
            plot.assert_not_called()
            toggle.get.return_value = False
            for cv_value in (0, 1, 2):
                cv_variable.get.return_value = cv_value
                gui.update(None)
                expected = (
                    gui.scores_to_include_avg if cv_value == 0
                    else gui.scores_to_include_GUI[cv_value - 1]
                )
                self.assertIs(plot.call_args.args[3], expected)
                self.assertEqual(plot.call_args.args[4:], (0, 5))


if __name__ == "__main__":
    unittest.main()
