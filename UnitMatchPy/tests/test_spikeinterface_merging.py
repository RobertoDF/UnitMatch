import numpy as np
import pytest

from UnitMatchPy.spikeinterface_merging import SpikeInterfaceSessionMerger


class _Analyzer:
    unit_ids = np.array([10, 11, 12])

    @staticmethod
    def has_extension(name):
        return name in {"random_spikes", "waveforms"}

    @staticmethod
    def get_num_segments():
        return 1

    def merge_units(self, **kwargs):
        self.merge_kwargs = kwargs
        return "merged"


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


def test_save_requires_applied_analyzer(tmp_path):
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    with pytest.raises(RuntimeError, match=r"apply_merges\(\)"):
        merger.save(tmp_path / "merged")
