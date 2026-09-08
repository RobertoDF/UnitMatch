import time
import unittest
import ast
from pathlib import Path
from concurrent.futures import Future
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from UnitMatchPy import GUI as gui
from test_gui_startup import gui_inputs


def result(score=72.0):
    return SimpleNamespace(
        score=score, distance=0.81, residual_um=1.2,
        reference_count=23, reason="" if score is not None else "Insufficient reference",
    )


class DisplacementGuiTests(unittest.TestCase):
    def setUp(self):
        module_state = patch.dict(gui.__dict__)
        module_state.start()
        self.addCleanup(module_state.stop)
        gui.score_histogram_cache = {}
        gui.displacement_metric_executor = None
        self.inputs = gui_inputs()
        gui.process_info_for_GUI(**self.inputs)

    def tearDown(self):
        if gui.displacement_metric_executor is not None:
            gui.displacement_metric_executor.shutdown(wait=True, cancel_futures=True)
            gui.displacement_metric_executor = None

    def test_worker_cancellation_does_not_publish_stale_results(self):
        cancel = Event()
        results = Queue()
        model = Mock()
        model.score.side_effect = lambda *args: (cancel.set(), result())[1]
        gui._compute_displacement_consistency([("0", 0, 4)], model, cancel, results)
        self.assertTrue(results.empty())
        model.score.reset_mock()
        gui._compute_displacement_consistency([("0", 0, 4)], model, cancel, results)
        model.score.assert_not_called()

    def test_tk_variables_and_images_use_explicit_window_ownership(self):
        tree = ast.parse(Path(gui.__file__).read_text(encoding="utf-8"))
        constructors = {"IntVar", "DoubleVar", "BooleanVar", "StringVar", "PhotoImage"}
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in constructors
        ]
        self.assertGreaterEqual(len(calls), 10)
        for call in calls:
            with self.subTest(constructor=call.func.id, line=call.lineno):
                self.assertIn("master", {keyword.arg for keyword in call.keywords})

    def test_consistency_filter_excludes_missing_and_uses_strict_cutoff(self):
        model = Mock()
        model.score.side_effect = [
            SimpleNamespace(score=19.9), SimpleNamespace(score=20),
            SimpleNamespace(score=None), SimpleNamespace(score=0),
        ]
        selected, unavailable = gui._compute_consistency_filter(
            [(0, 4), (1, 5), (2, 6), (3, 7)], model, 20, Event(),
        )
        self.assertEqual(selected, {(0, 4), (3, 7)})
        self.assertEqual(unavailable, 1)
        cancel = Event()
        cancel.set()
        self.assertIsNone(gui._compute_consistency_filter([(0, 4)], model, 20, cancel))

    def test_queued_navigation_is_coalesced(self):
        root = Mock()
        root.after_idle.side_effect = ["first", "second"]
        with patch.object(gui, "root", root, create=True):
            gui.pending_redraw = None
            gui.update(None)
            gui.update(None)
            root.after_cancel.assert_called_once_with("first")
            self.assertEqual(gui.pending_redraw, "second")

    def test_filter_refresh_after_decision_uses_current_reference(self):
        with (
            patch.object(gui, "_unusual_filter_enabled", return_value=True),
            patch.object(gui, "update_unusual_displacement_filter") as refresh,
        ):
            gui._refresh_displacement_consistency()
            refresh.assert_called_once_with()

    def test_async_filter_waits_then_applies_results(self):
        future = Future()
        root = Mock()
        with (
            patch.object(gui, "root", root, create=True),
            patch.object(gui, "consistency_filter_future", future),
            patch.object(gui, "_apply_displacement_filter") as apply,
        ):
            gui._poll_consistency_filter()
            apply.assert_not_called()
            root.after.assert_called_once_with(50, gui._poll_consistency_filter)
            future.set_result(({(0, 4)}, 2))
            gui._poll_consistency_filter()
            self.assertEqual(gui.consistency_filter_pairs, {(0, 4)})
            self.assertEqual(gui.consistency_filter_unavailable, 2)
            apply.assert_called_once_with()

    def test_enabling_filter_schedules_scoring_off_the_tk_thread(self):
        root, executor, model, status = Mock(), Mock(), Mock(), Mock()
        future = Future()
        executor.submit.return_value = future
        with (
            patch.object(gui, "root", root, create=True),
            patch.object(gui, "displacement_metric_executor", executor),
            patch.object(gui, "consistency_threshold_var", Mock(get=lambda: 20), create=True),
            patch.object(gui, "session_entry_a", Mock(get=lambda: "1"), create=True),
            patch.object(gui, "session_entry_b", Mock(get=lambda: "2"), create=True),
            patch.object(gui, "unusual_displacement_status_label", status, create=True),
            patch.object(gui, "_unusual_filter_enabled", return_value=True),
            patch.object(gui, "_get_displacement_context", return_value=(model, "", "probe")),
            patch.object(gui, "get_ranked_unit_a_options", return_value=[[0, 4]]),
            patch.object(gui, "_set_pair_controls") as controls,
        ):
            gui.update_unusual_displacement_filter()
            model.score.assert_not_called()
            self.assertIs(gui.consistency_filter_future, future)
            self.assertEqual(executor.submit.call_args.args[:4], (
                gui._compute_consistency_filter, [[0, 4]], model, 20,
            ))
            controls.assert_called_once_with(False)
            root.after.assert_called_once_with(50, gui._poll_consistency_filter)
            gui._cancel_consistency_filter()
            self.assertTrue(future.cancelled())

    def test_histogram_markers_update_without_rebuilding_canvas(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axis = fig.subplots()
        axis.set_xlim(0, 1)
        marker = axis.axvline(0.2, animated=True)
        panel = gui._HistogramPanel(canvas, [marker], ("fixed",))
        canvas.draw()
        original_pixels = np.asarray(canvas.buffer_rgba()).copy()
        with patch.object(canvas, "draw", wraps=canvas.draw) as draw:
            panel.update([0.7])
            draw.assert_not_called()
        np.testing.assert_array_equal(marker.get_xdata(), [0.7, 0.7])
        self.assertFalse(np.array_equal(original_pixels, np.asarray(canvas.buffer_rgba())))
        panel._on_resize(None)
        self.assertIsNone(panel.background)
        canvas.draw()
        self.assertIsNotNone(panel.background)

    def test_raw_plot_reuses_channel_axes_for_another_partner(self):
        line, canvas, figure = Mock(), Mock(), Mock()
        key = (0, "Avg", id(gui.waveform), id(gui.channel_pos), gui.gui_scale)
        with (
            patch.object(gui, "raw_waveform_view_key", key),
            patch.object(gui, "raw_waveform_pair_lines", [(0, line)]),
            patch.object(gui, "raw_waveform_plot", Mock(), create=True),
            patch.object(gui, "raw_waveform_canvas", canvas),
            patch.object(gui, "raw_waveform_figure", figure),
            patch.object(gui, "raw_displacement_axis", None),
            patch.object(gui, "_widget_exists", return_value=True),
            patch.object(gui, "_add_raw_displacement_overlay") as overlay,
            patch.object(gui, "Figure") as create_figure,
        ):
            gui.plot_raw_waveforms(0, 5, "Avg")
            create_figure.assert_not_called()
            np.testing.assert_allclose(
                line.set_ydata.call_args.args[0], gui.waveform[5, :, 0].mean(axis=-1),
            )
            overlay.assert_called_once_with(figure, 0, 5)
            canvas.draw_idle.assert_called_once_with()

    def test_refresh_invalidates_and_reschedules_both_tables(self):
        tables = [
            Mock(spec=gui._DiagnosticUnitTable),
            Mock(spec=gui._DiagnosticUnitTable),
        ]
        with (
            patch.object(gui, "entry_a", tables[0], create=True),
            patch.object(gui, "entry_b", tables[1], create=True),
            patch.object(gui, "_widget_exists", return_value=True),
        ):
            gui.displacement_context_key = "old reference"
            gui._refresh_displacement_consistency()
            self.assertIsNone(gui.displacement_context_key)
            for table in tables:
                table.refresh_displacement_metrics.assert_called_once_with()

    def test_reference_snapshot_updates_when_decisions_change(self):
        self.inputs["raw_avg_centroid_in"] = np.ones((3, 8, 2))
        gui.process_info_for_GUI(**self.inputs, automatic_matches_in=[[0, 4], [1, 5]])
        with patch.object(gui, "is_match", [[2, 6]]), patch.object(gui, "not_match", [[1, 5]]):
            with patch("UnitMatchPy.displacement_consistency.DisplacementConsistency") as cls:
                first = gui._get_displacement_context()
                self.assertEqual(cls.call_args.args[3], ((0, 4), (2, 6)))
                self.assertIn("no shank metadata", first[2])
                self.assertIs(gui._get_displacement_context(), first)
                self.assertEqual(cls.call_count, 1)
                gui.not_match.append([2, 6])
                gui._get_displacement_context()
                self.assertEqual(cls.call_count, 2)
                self.assertEqual(cls.call_args.args[3], ((0, 4),))

    def test_shank_metadata_and_minimum_are_forwarded(self):
        self.inputs["raw_avg_centroid_in"] = np.ones((3, 8, 2))
        shanks = np.zeros(8, dtype=int)
        self.inputs["clus_info_in"]["shank_ids"] = shanks
        gui.process_info_for_GUI(**self.inputs, displacement_min_references_in=25)
        with patch("UnitMatchPy.displacement_consistency.DisplacementConsistency") as cls:
            context = gui._get_displacement_context()
            self.assertEqual(cls.call_args.kwargs["min_references"], 25)
            self.assertIs(cls.call_args.kwargs["shank_ids"], shanks)
            self.assertEqual(context[2], "same probe and shank")

    def test_preparation_invalidates_model_and_validates_minimum(self):
        gui.displacement_context_key = "previous"
        gui.process_info_for_GUI(**self.inputs)
        self.assertIsNone(gui.displacement_context_key)
        for minimum in (True, 2, 2.5, np.nan):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                gui.process_info_for_GUI(**self.inputs, displacement_min_references_in=minimum)

    def test_match_and_nonmatch_refresh_both_tables_without_replacing_lists(self):
        accepted, rejected = [], []
        with (
            patch.object(gui, "is_match", accepted),
            patch.object(gui, "not_match", rejected),
            patch.object(gui, "entry_a", Mock(get=lambda: "0"), create=True),
            patch.object(gui, "entry_b", Mock(get=lambda: "4"), create=True),
            patch.object(gui, "CV_tkinter", Mock(get=lambda: 0), create=True),
            patch.object(gui, "add_probability_label"),
            patch.object(gui, "_refresh_raw_displacement_overlay"),
            patch.object(gui, "color_unit_a_options"),
            patch.object(gui, "_refresh_displacement_consistency") as refresh,
        ):
            gui.set_match()
            self.assertIn([0, 4], accepted)
            gui.set_not_match()
            self.assertEqual(accepted, [])
            self.assertIn([0, 4], rejected)
            self.assertIs(gui.is_match, accepted)
            self.assertIs(gui.not_match, rejected)
            self.assertEqual(refresh.call_count, 2)

    def test_decisions_preserved_by_default(self):
        self.assertTrue(gui.run_GUI.__kwdefaults__["preserve_decisions"])

    def test_replacement_excludes_competitors_but_keeps_other_sessions(self):
        sessions = np.array([0, 0, 1, 1, 2, 2])
        automatic = {(0, 2), (2, 0), (1, 3), (0, 4), (3, 5)}
        accepted, rejected = [[0, 2], [2, 0]], []
        with (
            patch.object(gui, "clus_info", {"session_id": sessions}),
            patch.object(gui, "automatic_match_pairs", automatic),
            patch.object(gui, "is_match", accepted),
            patch.object(gui, "not_match", rejected),
        ):
            gui._accept_pair(3, 0)
            self.assertEqual(
                set(gui._curated_accepted_pairs(automatic, accepted, rejected)),
                {(0, 3), (0, 4), (3, 5)},
            )
            self.assertEqual(set(map(tuple, rejected)), {(0, 2), (2, 0), (1, 3), (3, 1)})
            self.assertIs(gui.is_match, accepted)
            self.assertIs(gui.not_match, rejected)
            self.assertEqual(automatic, {(0, 2), (2, 0), (1, 3), (0, 4), (3, 5)})
            gui._accept_pair(0, 2)
            self.assertIn((0, 2), gui._curated_accepted_pairs(automatic, accepted, rejected))
            self.assertIn([0, 3], rejected)

    def test_tables_compute_spatial_metrics_without_event_data(self):
        root = gui.Tk()
        root.withdraw()
        model = Mock()
        model.score.return_value = result()
        try:
            with (
                patch.object(gui, "_get_displacement_context", return_value=(model, "", "same probe")),
                patch.object(gui, "entry_a", Mock(get=lambda: "0"), create=True),
            ):
                tables = [
                    gui.UnitATable(root, ["0 (1) 4", "1 (1) 5"]),
                    gui.UnitBTable(root, [4, 5]),
                ]
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        root.update()
                        if all(
                            table.table.set(table.table.get_children()[0], "displacement") == "72.0"
                            for table in tables
                        ):
                            break
                        time.sleep(0.01)
                    for table in tables:
                        row = table.table.get_children()[0]
                        self.assertEqual(table.table.set(row, "event_r"), "n/a")
                        self.assertEqual(table.table.set(row, "displacement"), "72.0")
                        self.assertIn("Residual: 1.20 um", table._displacement_details[row])
                        self.assertIn("23 independent accepted", table._displacement_details[row])
                    scored_pairs = {call.args for call in model.score.call_args_list}
                    self.assertEqual(scored_pairs, {(0, 4), (1, 5), (0, 5)})
                    previous_cancel = tables[0]._displacement_cancel
                    tables[0].set_options(["1 (1) 4"])
                    self.assertTrue(previous_cancel.is_set())
                    self.assertEqual(tables[0]._displacement_details, {})
                finally:
                    for table in tables:
                        table.destroy()
                    root.update()
        finally:
            root.destroy()

    def test_unavailable_and_worker_errors_are_not_fake_scores(self):
        root = gui.Tk()
        root.withdraw()
        try:
            with patch.object(gui, "_get_displacement_context", return_value=(None, "No raw centroids", "")):
                table = gui.UnitATable(root, ["0 (1) 4"])
                root.update()
                self.assertEqual(table.table.set("0", "displacement"), "n/a")
                self.assertEqual(table._displacement_details["0"], "No raw centroids")
                table._displacement_future = Future()
                table._displacement_future.set_exception(RuntimeError("worker failed"))
                table._displacement_results = Queue()
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    table._poll_displacement_metrics()
                table.destroy()
        finally:
            root.destroy()

    def test_final_result_is_drained_and_unavailable_reason_is_visible(self):
        root = gui.Tk()
        root.withdraw()
        try:
            table = gui.UnitATable(root, ["0 (1) 4"])
            table._cancel_displacement_metrics()
            table._displacement_scope = "same probe and shank"
            table._displacement_results = Queue()
            table._displacement_results.put(("0", result(score=None)))
            table._displacement_future = Future()
            table._displacement_future.set_result(None)
            table._poll_displacement_metrics()
            self.assertEqual(table.table.set("0", "displacement"), "n/a")
            self.assertIn("Insufficient reference", table._displacement_details["0"])
            self.assertIsNone(table._displacement_after)
            table.destroy()
        finally:
            root.destroy()
