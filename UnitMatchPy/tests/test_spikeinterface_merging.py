import numpy as np
import pytest
from UnitMatchPy.spikeinterface_merging import SpikeInterfaceSessionMerger


def test_session_merger_uses_conservative_spatial_limit():
    assert SpikeInterfaceSessionMerger.MAX_DISTANCE_UM == 50


class _Analyzer:
    unit_ids = np.array([10, 11, 12])
    sampling_frequency = 30_000
    recording = "recording"
    sparsity = None

    def __init__(self, unit_ids=None):
        if unit_ids is not None:
            self.unit_ids = np.asarray(unit_ids)
        self.sorting = _Sorting(self.unit_ids)

    @staticmethod
    def has_extension(name):
        return name in {"random_spikes", "waveforms"}

    @staticmethod
    def get_num_segments():
        return 1

    @staticmethod
    def get_num_samples():
        return 90_000

    @staticmethod
    def get_channel_locations():
        return np.array([[0, 0], [0, 20]])

    @staticmethod
    def get_num_channels():
        return 2

    def get_extension(self, name):
        if name == "waveforms":
            return _Waveforms()
        if name == "random_spikes":
            return _RandomSpikes()
        raise AssertionError(name)

    def merge_units(self, merge_unit_groups, censor_ms, merging_mode):
        self.merge_kwargs = {
            "merge_unit_groups": merge_unit_groups,
            "censor_ms": censor_ms,
            "merging_mode": merging_mode,
        }
        return "merged"


class _Sorting:
    def __init__(self, unit_ids):
        self.unit_ids = np.asarray(unit_ids)

    def select_units(self, unit_ids):
        return tuple(unit_ids)

    @staticmethod
    def get_unit_spike_train(unit_id):
        del unit_id
        return np.array([0, 30_000, 60_000])


class _Waveforms:
    def __init__(self):
        self.params = {"ms_before": 1.0, "ms_after": 1.0}

    @staticmethod
    def get_waveforms_one_unit(unit_id):
        del unit_id
        return np.array(
            [
                [[0.0, 0.0], [-2.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-4.0, -2.0], [0.0, 0.0]],
            ]
        )


class _RandomSpikes:
    @staticmethod
    def get_selected_indices_in_spike_train(unit_id, segment_index):
        del unit_id, segment_index
        return np.array([0, 2])


def test_apply_requires_all_proposals_reviewed():
    merger = SpikeInterfaceSessionMerger(_Analyzer())
    merger.merge_groups = [[10, 11]]
    merger.decisions = {(10, 11): None}

    with pytest.raises(RuntimeError, match="Approve or reject every"):
        merger.apply_merges()


def test_apply_soft_merges_only_approved_groups():
    analyzer = _Analyzer()
    merger = SpikeInterfaceSessionMerger(analyzer, censored_period_ms=0.75)
    merger.merge_groups = [[10, 11], [12, 13]]
    merger.decisions = {(10, 11): True, (12, 13): False}

    result = merger.apply_merges()

    assert result == "merged"
    assert analyzer.merge_kwargs == {
        "merge_unit_groups": [[10, 11]],
        "censor_ms": 0.75,
        "merging_mode": "soft",
    }


def test_apply_forwards_hard_merging_mode():
    analyzer = _Analyzer()
    merger = SpikeInterfaceSessionMerger(analyzer, merging_mode="hard")
    merger.decisions = {(10, 11): True}

    merger.apply_merges()

    assert analyzer.merge_kwargs["merging_mode"] == "hard"


def test_rejects_unknown_merging_mode():
    with pytest.raises(ValueError, match="either 'soft' or 'hard'"):
        SpikeInterfaceSessionMerger(_Analyzer(), merging_mode="automatic")


def test_unit_diagnostics_include_waveform_and_peak_location():
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    (
        times_ms,
        waveform,
        peak_location,
        channel_locations,
        spike_times_s,
        spike_amplitudes,
    ) = (
        merger._get_unit_diagnostics(10)
    )

    np.testing.assert_allclose(times_ms, [-1.0, -0.96666667, -0.93333333])
    np.testing.assert_allclose(waveform, [0.0, -3.0, 0.0])
    np.testing.assert_array_equal(peak_location, [0, 0])
    np.testing.assert_array_equal(channel_locations, [[0, 0], [0, 20]])
    np.testing.assert_array_equal(spike_times_s, [0.0, 2.0])
    np.testing.assert_array_equal(spike_amplitudes, [-2.0, -4.0])


def test_group_figure_renders_waveforms_and_probe_locations():
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    figure = merger._make_group_figure([10, 11])

    assert len(figure.axes) == 3
    assert figure.axes[0].get_title() == "Mean waveform on peak channel"
    assert figure.axes[1].get_title() == "Peak location on probe"
    assert figure.axes[2].get_title() == "Spike amplitudes over time"


def test_save_requires_applied_analyzer(tmp_path):
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    with pytest.raises(RuntimeError, match=r"apply_merges\(\)"):
        merger.save(tmp_path / "merged")


def test_apply_and_save_runs_both_steps(monkeypatch, tmp_path):
    merger = SpikeInterfaceSessionMerger(_Analyzer())
    calls = []

    monkeypatch.setattr(
        merger,
        "apply_merges",
        lambda: calls.append("apply"),
    )

    def save(folder, format, overwrite):
        calls.append((folder, format, overwrite))
        return "saved"

    monkeypatch.setattr(merger, "save", save)
    output_path = tmp_path / "merged"

    result = merger.apply_and_save(output_path, overwrite=True)

    assert result == "saved"
    assert calls == [
        "apply",
        (output_path, "binary_folder", True),
    ]


def test_multiple_analyzers_require_final_metric_unit_ids():
    with pytest.raises(ValueError, match="metrics.index"):
        SpikeInterfaceSessionMerger([_Analyzer([1, 2]), _Analyzer([3])])


def test_final_metric_ids_select_complementary_analyzer_units():
    original = _Analyzer([1, 2, 4])
    replacements = _Analyzer([3])

    merger = SpikeInterfaceSessionMerger(
        [original, replacements],
        unit_ids=[3, 4],
    )

    assert merger.unit_ids.tolist() == [3, 4]
    assert merger._unit_sources[3] is replacements
    assert merger._unit_sources[4] is original


def test_replaced_source_units_are_not_selected(monkeypatch):
    original = _Analyzer([1, 2, 4])
    replacements = _Analyzer([3])
    aggregate_calls = []

    def aggregate_units(sortings, renamed_unit_ids):
        aggregate_calls.append((sortings, renamed_unit_ids.tolist()))
        return "combined-sorting"

    class _CombinedAnalyzer:
        def merge_units(self, merge_unit_groups, censor_ms, merging_mode):
            return {
                "merge_unit_groups": merge_unit_groups,
                "censor_ms": censor_ms,
                "merging_mode": merging_mode,
            }

    def create_sorting_analyzer(**kwargs):
        assert kwargs["sorting"] == "combined-sorting"
        assert kwargs["recording"] == "recording"
        return _CombinedAnalyzer()

    import sys
    import types

    spikeinterface = types.SimpleNamespace(
        SortingAnalyzer=object,
        aggregate_units=aggregate_units,
        create_sorting_analyzer=create_sorting_analyzer,
    )
    monkeypatch.setitem(sys.modules, "spikeinterface", spikeinterface)

    merger = SpikeInterfaceSessionMerger(
        [original, replacements],
        unit_ids=[3, 4],
    )
    merger.decisions = {(3, 4): True}

    result = merger.apply_merges()

    assert aggregate_calls == [([(3,), (4,)], [3, 4])]
    assert result["merge_unit_groups"] == [[3, 4]]
