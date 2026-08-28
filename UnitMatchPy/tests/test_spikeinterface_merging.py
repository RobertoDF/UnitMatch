import numpy as np
import pytest
from UnitMatchPy.spikeinterface_merging import SpikeInterfaceSessionMerger


class _Analyzer:
    unit_ids = np.array([10, 11, 12])
    sampling_frequency = 30_000
    recording = "recording"

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

    def merge_units(self, **kwargs):
        self.merge_kwargs = kwargs
        return "merged"


class _Sorting:
    def __init__(self, unit_ids):
        self.unit_ids = np.asarray(unit_ids)

    def select_units(self, unit_ids):
        return tuple(unit_ids)


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
        "censored_period_ms": 0.75,
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


def test_save_requires_applied_analyzer(tmp_path):
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    with pytest.raises(RuntimeError, match=r"apply_merges\(\)"):
        merger.save(tmp_path / "merged")


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
        def merge_units(self, **kwargs):
            return kwargs

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
