import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from UnitMatchPy.spikeinterface_amplitudes import prepare_unitmatch_amplitude_data


class UnitMatchAmplitudeDataTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([-10., -20., -11., -21., -22.])
        self.extension = Mock()
        self.extension.get_data.return_value = self.values
        self.sorting = Mock()
        self.sorting.unit_ids = np.array([7, 42])
        self.sorting.get_num_segments.return_value = 2
        self.sorting.get_spike_vector_to_indices.return_value = {
            0: {7: np.array([0, 2]), 42: np.array([1])},
            1: {7: np.array([], dtype=int), 42: np.array([3, 4])},
        }
        self.times = {
            (7, 0): np.array([0.1, 0.2]),
            (42, 0): np.array([0.1]),
            (7, 1): np.array([]),
            (42, 1): np.array([3.0, 4.0]),
        }
        self.sorting.get_unit_spike_train.side_effect = (
            lambda unit_id, *, segment_index, return_times:
            self.times[(unit_id, segment_index)]
        )
        self.analyzer = SimpleNamespace(
            sorting=self.sorting,
            return_in_uV=True,
            has_extension=Mock(return_value=True),
            get_extension=Mock(return_value=self.extension),
            compute=Mock(side_effect=AssertionError("Review must not compute")),
        )
        self.rows = [
            {"session_index": 1, "probe_n": 3, "unit_id": 42, "export_unit_id": 0},
            {"session_index": 1, "probe_n": 3, "unit_id": 7, "export_unit_id": 1},
        ]
        self.loader = prepare_unitmatch_amplitude_data(
            [{}, {3: self.analyzer}], self.rows
        )["get_amplitudes"]

    def test_exact_analyzer_ids_signed_amplitudes_and_segment_alignment(self):
        result = self.loader(0)
        self.assertEqual(result["amplitude_units"], "uV")
        np.testing.assert_array_equal(result["segments"][0]["amplitudes"], [-20])
        np.testing.assert_array_equal(result["segments"][1]["amplitudes"], [-21, -22])
        np.testing.assert_array_equal(result["segments"][1]["times_s"], [3, 4])
        self.extension.get_data.assert_called_once_with(outputs="numpy")
        self.analyzer.compute.assert_not_called()
        np.testing.assert_array_equal(self.values, [-10, -20, -11, -21, -22])

    def test_empty_segment_and_numpy_row_index(self):
        result = self.loader(np.int64(1))
        self.assertEqual(result["segments"][1]["amplitudes"].size, 0)
        self.assertEqual(result["segments"][1]["times_s"].size, 0)

    def test_unscaled_amplitudes_are_not_labeled_microvolts(self):
        self.analyzer.return_in_uV = False
        self.assertEqual(self.loader(0)["amplitude_units"], "ADC units")

    def test_transitional_sorting_analyzer_uses_return_scaled(self):
        del self.analyzer.return_in_uV
        self.analyzer.return_scaled = True
        self.assertEqual(self.loader(0)["amplitude_units"], "uV")
        self.analyzer.return_scaled = False
        self.assertEqual(self.loader(0)["amplitude_units"], "ADC units")

    def test_older_spikeinterface_vector_and_scaled_flag_are_supported(self):
        class LegacyExtension:
            def __init__(self, segments):
                self.segments = segments
                self.outputs = []
                self._params = {"return_scaled": True}

            def get_data(self, outputs="concatenated"):
                self.outputs.append(outputs)
                return self.segments

        spike_vector = np.array(
            [(0, 0), (1, 0), (0, 0), (1, 1), (1, 1)],
            dtype=[("unit_index", "i8"), ("segment_index", "i8")],
        )
        extension = LegacyExtension([
            self.values[:3],
            self.values[3:],
        ])
        sorting = SimpleNamespace(
            unit_ids=np.array([7, 42]),
            get_num_segments=lambda: 2,
            to_spike_vector=lambda: spike_vector,
            get_unit_spike_train=self.sorting.get_unit_spike_train,
        )
        analyzer = SimpleNamespace(
            sorting=sorting,
            return_scaled=False,
            recording=SimpleNamespace(has_scaled_traces=lambda: True),
            has_recording=lambda: True,
            has_extension=Mock(return_value=True),
            load_extension=Mock(return_value=extension),
        )
        loader = prepare_unitmatch_amplitude_data(
            [{}, {3: analyzer}], self.rows
        )["get_amplitudes"]

        result = loader(0)

        self.assertEqual(result["amplitude_units"], "uV")
        np.testing.assert_array_equal(result["segments"][0]["amplitudes"], [-20])
        np.testing.assert_array_equal(result["segments"][1]["amplitudes"], [-21, -22])
        self.assertEqual(extension.outputs, ["concatenated"])
        analyzer.load_extension.assert_called_once_with("spike_amplitudes")

        analyzer.recording.has_scaled_traces = lambda: False
        self.assertEqual(loader(0)["amplitude_units"], "ADC units")

    def test_generic_legacy_extension_wrapper_does_not_receive_copy(self):
        class ForwardingExtension:
            def __init__(self, values):
                self.values = values
                self.calls = []

            def get_data(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                if "copy" in kwargs:
                    raise TypeError("copy is unsupported")
                if kwargs.get("outputs") != "numpy":
                    raise ValueError("expected numpy output")
                return self.values

        extension = ForwardingExtension(self.values)
        self.analyzer.get_extension.return_value = extension

        result = self.loader(0)

        np.testing.assert_array_equal(result["segments"][1]["amplitudes"], [-21, -22])
        self.assertEqual(extension.calls, [((), {"outputs": "numpy"})])

    def test_modern_extension_uses_zero_copy_when_implementation_supports_it(self):
        class ModernExtension:
            def __init__(self, values):
                self.values = values
                self.calls = []

            def _get_data(self, outputs="numpy", copy=True):
                return self.values

            def get_data(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return self._get_data(*args, **kwargs)

        extension = ModernExtension(self.values)
        self.analyzer.get_extension.return_value = extension

        self.loader(0)

        self.assertEqual(
            extension.calls,
            [((), {"outputs": "numpy", "copy": False})],
        )

    def test_mapping_snapshot_is_not_changed_by_caller_edits(self):
        self.rows[0]["unit_id"] = 7
        np.testing.assert_array_equal(self.loader(0)["segments"][0]["amplitudes"], [-20])

    def test_invalid_rows_are_rejected(self):
        for row in (-1, True, np.bool_(False), 0.5, "0"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.loader(row)
        with self.assertRaises(IndexError):
            self.loader(2)

    def test_missing_analyzer_and_unit_are_explicit(self):
        loader = prepare_unitmatch_amplitude_data([{}], self.rows)["get_amplitudes"]
        with self.assertRaisesRegex(KeyError, "No analyzer"):
            loader(0)
        self.sorting.unit_ids = np.array([7])
        with self.assertRaisesRegex(KeyError, "unit 42"):
            self.loader(0)

    def test_missing_extension_does_not_compute(self):
        self.analyzer.has_extension.return_value = False
        with self.assertRaisesRegex(ValueError, "does not compute"):
            self.loader(0)
        self.analyzer.get_extension.assert_not_called()
        self.analyzer.compute.assert_not_called()

    def test_spike_count_mismatch_is_rejected(self):
        self.times[(42, 0)] = np.array([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.loader(0)

    def test_invalid_indices_are_rejected(self):
        for indices in (np.array([-1]), np.array([5]), np.array([1.5])):
            with self.subTest(indices=indices):
                self.sorting.get_spike_vector_to_indices.return_value[0][42] = indices
                with self.assertRaisesRegex(ValueError, "Invalid amplitude indices"):
                    self.loader(0)

    def test_nonfinite_data_is_not_silently_dropped(self):
        self.values[1] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.loader(0)


if __name__ == "__main__":
    unittest.main()
