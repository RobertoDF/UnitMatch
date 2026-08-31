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

    def merge_units(
        self,
        merge_unit_groups,
        censor_ms,
        merging_mode,
        return_new_unit_ids,
        **job_kwargs,
    ):
        new_unit_ids = [
            int(np.max(self.unit_ids)) + index + 1
            for index in range(len(merge_unit_groups))
        ]
        self.merge_kwargs = {
            "merge_unit_groups": merge_unit_groups,
            "censor_ms": censor_ms,
            "merging_mode": merging_mode,
            "return_new_unit_ids": return_new_unit_ids,
            **job_kwargs,
        }
        merged_source_ids = {
            unit_id for group in merge_unit_groups for unit_id in group
        }
        remaining_ids = [
            unit_id
            for unit_id in self.unit_ids
            if unit_id not in merged_source_ids
        ]
        return _Analyzer(remaining_ids + new_unit_ids), new_unit_ids


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

    result = merger.apply_merges(job_kwargs={"n_jobs": 2})

    assert result.unit_ids.tolist() == [12, 13]
    assert merger.merged_unit_ids == [13]
    assert analyzer.merge_kwargs == {
        "merge_unit_groups": [[10, 11]],
        "censor_ms": 0.75,
        "merging_mode": "soft",
        "return_new_unit_ids": True,
        "n_jobs": 2,
    }


def test_apply_with_no_approved_groups_returns_original_analyzer():
    analyzer = _Analyzer()
    merger = SpikeInterfaceSessionMerger(analyzer)
    merger.decisions = {(10, 11): False}

    result = merger.apply_merges()

    assert result is analyzer
    assert merger.merged_analyzer is analyzer
    assert merger.merged_unit_ids == []


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

    diagnostics = merger._get_unit_diagnostics(10)

    np.testing.assert_allclose(
        diagnostics.times_ms, [-1.0, -0.96666667, -0.93333333]
    )
    np.testing.assert_allclose(
        diagnostics.waveform_on_channel(0), [0.0, -3.0, 0.0]
    )
    assert diagnostics.peak_channel_index == 0
    np.testing.assert_array_equal(diagnostics.peak_location, [0, 0])
    np.testing.assert_array_equal(
        diagnostics.channel_locations, [[0, 0], [0, 20]]
    )
    np.testing.assert_array_equal(diagnostics.spike_times_s, [0.0, 2.0])
    np.testing.assert_array_equal(diagnostics.spike_amplitudes, [-2.0, -4.0])
    np.testing.assert_array_equal(
        diagnostics.all_spike_times_s, [0.0, 1.0, 2.0]
    )


def test_group_figure_renders_waveforms_and_probe_locations():
    merger = SpikeInterfaceSessionMerger(_Analyzer())

    figure = merger._make_group_figure([10, 11])

    assert len(figure.axes) == 4
    assert figure.axes[0].get_title() == "Mean waveforms on both peak channels"
    assert figure.axes[1].get_title() == "Peak location on probe"
    assert figure.axes[2].get_title() == "Spike amplitudes over time"
    assert figure.axes[3].get_title() == "Spike rate over time"


class _SparseWaveforms(_Waveforms):
    @staticmethod
    def get_waveforms_one_unit(unit_id):
        peak_values = {
            10: [-1.0, -10.0, -4.0],
            11: [-2.0, -5.0, -12.0],
        }[unit_id]
        waveform = np.zeros((2, 3, 3))
        waveform[:, 1, :] = peak_values
        return waveform


class _SparseMapping:
    unit_id_to_channel_indices = {
        10: np.array([0, 2, 3]),
        11: np.array([1, 2, 3]),
    }


class _SparseAnalyzer(_Analyzer):
    sparsity = _SparseMapping()

    def __init__(self):
        super().__init__([10, 11])

    @staticmethod
    def get_num_channels():
        return 4

    @staticmethod
    def get_channel_locations():
        return np.array([[0, 0], [10, 0], [20, 0], [30, 0]])

    def get_extension(self, name):
        if name == "waveforms":
            return _SparseWaveforms()
        return super().get_extension(name)


def test_group_figure_maps_both_sparse_peak_channels():
    merger = SpikeInterfaceSessionMerger(_SparseAnalyzer())

    figure = merger._make_group_figure([10, 11])

    waveform_lines = figure.axes[0].lines[:4]
    assert [line.get_label() for line in waveform_lines] == [
        "Unit 10 on unit 10 peak channel",
        "Unit 10 on unit 11 peak channel",
        "Unit 11 on unit 10 peak channel",
        "Unit 11 on unit 11 peak channel",
    ]
    assert [line.get_color() for line in waveform_lines] == [
        "tab:blue",
        "tab:blue",
        "tab:orange",
        "tab:orange",
    ]
    assert [line.get_linestyle() for line in waveform_lines] == [
        "-",
        "--",
        "--",
        "-",
    ]
    np.testing.assert_array_equal(
        [line.get_ydata()[1] for line in waveform_lines],
        [-10.0, -4.0, -5.0, -12.0],
    )


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
        lambda job_kwargs: calls.append(("apply", job_kwargs)),
    )

    def save(folder, format, overwrite):
        calls.append((folder, format, overwrite))
        return "saved"

    monkeypatch.setattr(merger, "save", save)
    output_path = tmp_path / "merged"

    result = merger.apply_and_save(
        output_path,
        overwrite=True,
        job_kwargs={"n_jobs": 2},
    )

    assert result == "saved"
    assert calls == [
        ("apply", {"n_jobs": 2}),
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


def test_apply_rejects_multiple_analyzers():
    original = _Analyzer([1, 2, 4])
    replacements = _Analyzer([3])
    merger = SpikeInterfaceSessionMerger(
        [original, replacements],
        unit_ids=[3, 4],
    )
    merger.decisions = {(3, 4): True}

    with pytest.raises(RuntimeError, match="same single analyzer"):
        merger.apply_merges()
