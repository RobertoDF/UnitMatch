import unittest
from unittest.mock import Mock, patch

import numpy as np
from test_gui_startup import gui_inputs

from UnitMatchPy import GUI as gui


class SessionSummaryTests(unittest.TestCase):
    def test_probe_counts_use_unique_units_and_ignore_other_session_links(self):
        accepted = gui._curated_accepted_pairs(
            [(0, 4), (4, 0), (0, 5), (2, 6), (3, 7), (0, 8)],
            [], [(3, 7)],
        )
        groups = dict(gui._session_group_summaries(
            accepted, [0, 4, 8, 10], 1, 2,
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        ))
        self.assertEqual(list(groups), ["Probe 0", "Probe 1"])
        self.assertEqual(groups["Probe 0"]["accepted_pairs"], 2)
        self.assertEqual(groups["Probe 0"]["matched_a"], 1)
        self.assertEqual(groups["Probe 0"]["matched_b"], 2)
        self.assertEqual(groups["Probe 0"]["percent_a"], 50)
        self.assertEqual(groups["Probe 0"]["percent_b"], 100)
        self.assertEqual(groups["Probe 1"]["accepted_pairs"], 1)
        overall = gui._session_pair_summary(accepted, [0, 4, 8, 10], 1, 2)
        for field in ("accepted_pairs", "units_a", "units_b", "matched_a", "matched_b"):
            self.assertEqual(sum(group[field] for group in groups.values()), overall[field])
        reverse = dict(gui._session_group_summaries(
            accepted, [0, 4, 8, 10], 2, 1,
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        ))
        self.assertEqual(reverse["Probe 0"]["matched_a"], 2)
        self.assertEqual(reverse["Probe 0"]["matched_b"], 1)

    def test_shank_groups_preserve_unmatched_units_and_missing_metadata(self):
        groups = dict(gui._session_group_summaries(
            [(0, 4), (1, 5), (2, 6)], [0, 4, 8], 1, 2,
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, None, 0, 1, 0, None, 0, 1],
        ))
        self.assertEqual(list(groups), [
            "Probe 0 / shank 0", "Probe 0 / shank ?",
            "Probe 1 / shank 0", "Probe 1 / shank 1",
        ])
        self.assertEqual(groups["Probe 0 / shank ?"]["accepted_pairs"], 1)
        self.assertEqual(groups["Probe 1 / shank 1"]["units_a"], 1)
        self.assertEqual(groups["Probe 1 / shank 1"]["units_b"], 1)
        self.assertEqual(groups["Probe 1 / shank 1"]["matched_a"], 0)

    def test_probe_present_in_one_session_has_zero_other_session_total(self):
        groups = dict(gui._session_group_summaries(
            [(0, 4)], [0, 4, 8], 1, 2, [0, 0, 1, 1, 0, 0, 2, 2],
        ))
        self.assertEqual(groups["Probe 1"]["units_b"], 0)
        self.assertIsNone(groups["Probe 1"]["percent_b"])
        self.assertEqual(groups["Probe 2"]["units_a"], 0)
        self.assertIsNone(groups["Probe 2"]["percent_a"])

    def test_group_metadata_length_is_validated(self):
        with self.assertRaisesRegex(ValueError, "probe metadata"):
            gui._session_group_summaries([], [0, 2, 4], 1, 2, [0, 0])
        with self.assertRaisesRegex(ValueError, "shank metadata"):
            gui._session_group_summaries([], [0, 2, 4], 1, 2, [0] * 4, [0])

    def test_live_group_widget_refreshes_decisions_and_scrolls_extra_groups(self):
        module_state = patch.dict(gui.__dict__)
        module_state.start()
        self.addCleanup(module_state.stop)
        root = gui.Tk()
        self.addCleanup(root.destroy)
        gui.root = root
        gui.session_entry_a = Mock(get=lambda: "1")
        gui.session_entry_b = Mock(get=lambda: "2")
        gui.session_switch = np.array([0, 4, 8])
        gui.clus_info = {
            "probe_numbers": [0, 0, 1, 1, 0, 0, 1, 1],
            "shank_ids": [0, 1, 0, 1, 0, 1, 0, 1],
        }
        gui.automatic_match_pairs = {(0, 4), (1, 5), (2, 6), (3, 7)}
        gui.is_match = []
        gui.not_match = []
        gui.option_a = []
        parent = gui.ttk.Frame(root)
        parent.pack()
        gui._create_session_summary(parent).grid(row=0, column=1)
        root.update_idletasks()
        tree = gui.session_group_summary
        self.assertEqual(len(tree.get_children()), 4)
        self.assertEqual(int(tree.cget("height")), 2)
        self.assertEqual(tree.item("group-0", "values"), (
            "Probe 0 / shank 0", "1", "1/1 (100.0%)", "1/1 (100.0%)",
        ))
        height = tree.winfo_height()
        gui.not_match.extend([[0, 4], [4, 0]])
        gui.color_unit_a_options()
        self.assertEqual(tree.item("group-0", "values"), (
            "Probe 0 / shank 0", "0", "0/1 (0.0%)", "0/1 (0.0%)",
        ))
        gui.is_match.extend([[0, 5], [5, 0]])
        gui.color_unit_a_options()
        self.assertEqual(tree.item("group-4", "values"), (
            "Across groups", "1", "See overall", "See overall",
        ))
        tree.yview_moveto(1)
        root.update_idletasks()
        self.assertGreater(tree.yview()[0], 0)
        self.assertEqual(tree.winfo_height(), height)
        self.assertIn("Accepted pairs (S1 / S2): 4", gui.session_summary_label.cget("text"))

    def test_gui_reference_minimum_defaults_to_five_and_allows_override(self):
        with patch.dict(gui.__dict__):
            gui.process_info_for_GUI(**gui_inputs())
            self.assertEqual(gui.displacement_min_references, 5)
            gui.process_info_for_GUI(**gui_inputs(), displacement_min_references_in=9)
            self.assertEqual(gui.displacement_min_references, 9)

    def test_reciprocal_pairs_are_counted_once_and_other_sessions_are_excluded(self):
        summary = gui._session_pair_summary(
            [(0, 3), (3, 0), (0, 3), (1, 4), (2, 7), (0, 1), (0, 0)],
            [0, 3, 7, 9], 1, 2,
        )
        self.assertEqual(summary["accepted_pairs"], 2)
        self.assertEqual((summary["units_a"], summary["units_b"]), (3, 4))
        self.assertEqual((summary["matched_a"], summary["matched_b"]), (2, 2))
        self.assertAlmostEqual(summary["percent_a"], 200 / 3)
        self.assertEqual(summary["percent_b"], 50)

    def test_percentages_count_unique_units_not_pairs(self):
        summary = gui._session_pair_summary(
            [(0, 3), (0, 4), (0, 5), (0, 6)], [0, 3, 7], 1, 2,
        )
        self.assertEqual(summary["accepted_pairs"], 4)
        self.assertEqual(summary["matched_a"], 1)
        self.assertAlmostEqual(summary["percent_a"], 100 / 3)
        self.assertEqual(summary["percent_b"], 100)

    def test_swapping_sessions_swaps_denominators_and_percentages(self):
        forward = gui._session_pair_summary([(0, 3), (1, 4)], [0, 3, 7], 1, 2)
        reverse = gui._session_pair_summary([(0, 3), (1, 4)], [0, 3, 7], 2, 1)
        self.assertEqual(forward["accepted_pairs"], reverse["accepted_pairs"])
        self.assertEqual(forward["units_a"], reverse["units_b"])
        self.assertEqual(forward["percent_a"], reverse["percent_b"])
        self.assertEqual(forward["percent_b"], reverse["percent_a"])

    def test_empty_matches_and_empty_session(self):
        summary = gui._session_pair_summary([], [0, 0, 3], 1, 2)
        self.assertEqual(summary["accepted_pairs"], 0)
        self.assertIsNone(summary["percent_a"])
        self.assertEqual(summary["percent_b"], 0)

    def test_invalid_or_identical_sessions_raise(self):
        for first, second in ((0, 1), (1, 3), (1, 1)):
            with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                gui._session_pair_summary([], [0, 3, 7], first, second)

    def test_label_combines_auto_manual_and_rejections_independent_of_display_filter(self):
        module_state = patch.dict(gui.__dict__)
        module_state.start()
        self.addCleanup(module_state.stop)
        gui.session_summary_label = Mock()
        gui.session_summary_label.cget.return_value = ""
        gui.session_entry_a = Mock(get=lambda: "1")
        gui.session_entry_b = Mock(get=lambda: "2")
        gui.session_switch = np.array([0, 3, 7, 9])
        gui.automatic_match_pairs = {(0, 3), (3, 0), (1, 4), (2, 7)}
        gui.is_match = [[1, 5], [5, 1]]
        gui.not_match = [[1, 4], [4, 1]]
        gui.option_a = []
        gui.review_matches = np.empty((0, 2), dtype=int)
        with patch.object(gui, "_widget_exists", return_value=True):
            gui._refresh_session_summary()
            text = gui.session_summary_label.configure.call_args.kwargs["text"]
            self.assertEqual(text, (
                "Accepted pairs (S1 / S2): 2\n"
                "S1: 3 units; 2 matched (66.7%)\n"
                "S2: 4 units; 2 matched (50.0%)"
            ))
            gui.not_match.append([0, 3])
            gui._refresh_session_summary()
            self.assertIn(
                "Accepted pairs (S1 / S2): 1",
                gui.session_summary_label.configure.call_args.kwargs["text"],
            )
            gui.session_entry_a.get = lambda: "3"
            gui._refresh_session_summary()
            self.assertIn(
                "S3: 2 units; 0 matched (0.0%)",
                gui.session_summary_label.configure.call_args.kwargs["text"],
            )
            gui.session_entry_b.get = lambda: "3"
            gui._refresh_session_summary()
            gui.session_summary_label.configure.assert_called_with(
                text="Summary: select two different sessions"
            )
            gui.session_entry_a.get = lambda: "1"
            gui.session_entry_b.get = lambda: "2"
            gui.session_switch = np.array([0, 0, 3])
            gui._refresh_session_summary(accepted_pairs=[])
            self.assertIn(
                "S1: 0 units; 0 matched (n/a)",
                gui.session_summary_label.configure.call_args.kwargs["text"],
            )

    def test_review_color_refresh_updates_summary_with_the_same_accepted_set(self):
        with (
            patch.object(gui, "automatic_match_pairs", {(0, 3)}),
            patch.object(gui, "is_match", []),
            patch.object(gui, "not_match", []),
            patch.object(gui, "option_a", [], create=True),
            patch.object(gui, "_refresh_session_summary") as refresh,
        ):
            gui.color_unit_a_options()
            self.assertEqual(set(refresh.call_args.args[0]), {(0, 3)})
