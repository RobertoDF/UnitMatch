import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np

from UnitMatchPy.spikeinterface_merging import _load_unit_diagnostics
from UnitMatchPy.spikeinterface_review import prepare_unitmatch_diagnostic_data


class UnitMatchDiagnosticDataTests(unittest.TestCase):
    def setUp(self):
        self.waveforms = np.arange(24, dtype=float).reshape(2, 4, 3)
        self.waveform_extension = SimpleNamespace(
            params={"ms_before": 1.0},
            get_waveforms_one_unit=Mock(return_value=self.waveforms),
        )
        self.random_spikes = SimpleNamespace(
            get_selected_indices_in_spike_train=Mock(
                return_value=np.array([0, 2])
            )
        )
        self.sorting = SimpleNamespace(
            unit_ids=np.array([42]),
            get_unit_spike_train=Mock(return_value=np.array([0, 50, 100])),
        )
        self.analyzer = SimpleNamespace(
            sorting=self.sorting,
            sparsity=None,
            sampling_frequency=100.0,
            return_in_uV=True,
            channel_ids=np.array(["c0", "c1", "c2"]),
            get_num_segments=Mock(return_value=1),
            get_num_channels=Mock(return_value=3),
            get_channel_locations=Mock(
                return_value=np.array([[0, 0], [10, 0], [20, 0]])
            ),
            has_extension=Mock(
                side_effect=lambda name: name in {"waveforms", "random_spikes"}
            ),
            get_extension=Mock(
                side_effect={
                    "waveforms": self.waveform_extension,
                    "random_spikes": self.random_spikes,
                }.__getitem__
            ),
            compute=Mock(side_effect=AssertionError("review must not compute")),
        )
        self.rows = [
            {
                "session_index": 1,
                "probe_n": 3,
                "unit_id": 42,
                "export_unit_id": 7,
            }
        ]
        self.provider = prepare_unitmatch_diagnostic_data(
            [{}, {3: self.analyzer}],
            self.rows,
        )

    def test_callback_reuses_original_diagnostic_loader_without_computing(self):
        actual = self.provider["get_diagnostics"](0)
        expected = _load_unit_diagnostics(self.analyzer, 42)

        np.testing.assert_array_equal(actual.waveforms, expected.waveforms)
        np.testing.assert_array_equal(
            actual.spike_amplitudes_by_channel,
            expected.spike_amplitudes_by_channel,
        )
        np.testing.assert_array_equal(actual.spike_times_s, [0.0, 1.0])
        self.assertEqual(actual.amplitude_units, "uV")
        self.analyzer.compute.assert_not_called()

    def test_row_metadata_is_snapshotted_and_analyzer_identity_is_explicit(self):
        self.rows[0]["unit_id"] = 99

        diagnostics = self.provider["get_diagnostics"](0)

        self.assertEqual(diagnostics.peak_channel_index, 2)
        self.waveform_extension.get_waveforms_one_unit.assert_called_with(
            unit_id=42
        )

    def test_missing_extensions_surface_without_computation(self):
        self.analyzer.has_extension.side_effect = lambda name: name == "waveforms"

        with self.assertRaisesRegex(ValueError, "review does not compute"):
            self.provider["get_diagnostics"](0)

        self.analyzer.compute.assert_not_called()

    def test_multiple_segments_are_rejected_without_concatenation(self):
        self.analyzer.get_num_segments.return_value = 2

        with self.assertRaisesRegex(ValueError, "never concatenated"):
            self.provider["get_diagnostics"](0)

    def test_unscaled_waveforms_are_labeled_adc_units(self):
        self.analyzer.return_in_uV = False

        diagnostics = self.provider["get_diagnostics"](0)

        self.assertEqual(diagnostics.amplitude_units, "ADC units")

    def test_legacy_extension_loader_and_scaled_parameter_are_supported(self):
        del self.analyzer.return_in_uV
        del self.analyzer.get_extension
        self.waveform_extension._params = {"return_scaled": True}
        self.analyzer.load_extension = Mock(
            side_effect={
                "waveforms": self.waveform_extension,
                "random_spikes": self.random_spikes,
            }.__getitem__
        )

        diagnostics = self.provider["get_diagnostics"](0)

        self.assertEqual(diagnostics.amplitude_units, "uV")
        self.assertEqual(
            self.analyzer.load_extension.call_args_list,
            [
                call("waveforms"),
                call("random_spikes"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
