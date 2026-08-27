import json
import sys

import numpy as np
import pandas as pd
import pytest

from UnitMatchPy import save_utils
from UnitMatchPy import utils


def _param():
    return {"spike_width": 4, "peak_loc": 2, "waveidx": np.arange(4)}


def _write_waveform(session_path, unit_id, value=0, shape=(4, 3, 2)):
    waveform = np.full(shape, value, dtype=float)
    np.save(session_path / f"Unit{unit_id}_RawSpikes.npy", waveform)
    return waveform


def test_load_waveforms_uses_explicit_non_contiguous_ids(tmp_path):
    sessions = [tmp_path / "Session0", tmp_path / "Session1"]
    for session in sessions:
        session.mkdir()
        (session / "waveform_params.json").write_text("{}", encoding="utf-8")

    expected = [
        _write_waveform(sessions[0], 11, 11),
        _write_waveform(sessions[0], 3, 3),
        _write_waveform(sessions[1], 42, 42),
    ]

    result = utils.load_waveforms(
        sessions,
        [np.array([11, 3], dtype=np.int32), np.array([42], dtype=np.int64)],
        _param(),
        n_units_per_session_all=[8, 5],
    )
    waveform, session_id, session_switch, within_session, unit_ids, param = result

    np.testing.assert_array_equal(waveform, np.stack(expected))
    np.testing.assert_array_equal(session_id, [0, 0, 1])
    np.testing.assert_array_equal(session_switch, [0, 2, 3])
    np.testing.assert_array_equal(within_session, [[0, 0, 1], [0, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(unit_ids[0], [11, 3])
    np.testing.assert_array_equal(unit_ids[1], [42])
    np.testing.assert_array_equal(param["n_units_per_session"], [8, 5])
    assert param["n_units"] == 3
    assert param["n_sessions"] == 2
    assert param["n_channels"] == 3


def test_load_waveforms_preserves_string_unit_ids(tmp_path):
    session = tmp_path / "Session0"
    session.mkdir()
    _write_waveform(session, "unit-7", 7)

    waveform, _, _, _, unit_ids, _ = utils.load_waveforms(
        [session], [["unit-7"]], _param()
    )

    assert unit_ids[0].tolist() == ["unit-7"]
    assert waveform[0, 0, 0, 0] == 7


def test_load_waveforms_normalizes_json_waveform_params(tmp_path):
    session = tmp_path / "Session0"
    session.mkdir()
    _write_waveform(session, 1)
    param = {"spike_width": 4.0, "peak_loc": 2.0, "waveidx": [0, 1, 2, 3]}

    *_, param = utils.load_waveforms([session], [[1]], param)

    assert isinstance(param["spike_width"], int)
    assert isinstance(param["peak_loc"], int)
    np.testing.assert_array_equal(param["waveidx"], np.arange(4))
    assert isinstance(param["waveidx"], np.ndarray)


@pytest.mark.parametrize(
    ("filename", "labels", "expected"),
    [
        ("cluster_group.tsv", ["good", "mua", "good"], [10, 30]),
        ("cluster_bc_unitType.tsv", ["GOOD", "NOISE", "NON-SOMA GOOD"], [10, 30]),
    ],
)
def test_legacy_label_wrapper_delegates(tmp_path, filename, labels, expected):
    session = tmp_path / "Session0"
    session.mkdir()
    for unit_id in [10, 20, 30]:
        _write_waveform(session, unit_id, unit_id)
    label_path = session / filename
    pd.DataFrame({"cluster_id": [10, 20, 30], "label": labels}).to_csv(
        label_path, sep="\t", index=False
    )

    waveform, _, _, _, unit_ids, param = utils.load_good_waveforms(
        [session], [label_path], _param()
    )

    np.testing.assert_array_equal(unit_ids[0], expected)
    np.testing.assert_array_equal(waveform[:, 0, 0, 0], expected)
    np.testing.assert_array_equal(param["n_units_per_session"], [3])


def test_legacy_wrapper_can_load_all_labeled_units(tmp_path):
    session = tmp_path / "Session0"
    session.mkdir()
    for unit_id in [7, 21]:
        _write_waveform(session, unit_id, unit_id)
    label_path = session / "cluster_group.tsv"
    pd.DataFrame({"cluster_id": [7, 21], "label": ["mua", "noise"]}).to_csv(
        label_path, sep="\t", index=False
    )

    waveform, _, _, _, unit_ids, _ = utils.load_good_waveforms(
        [session], [label_path], _param(), good_units_only=False
    )

    np.testing.assert_array_equal(unit_ids[0], [7, 21])
    np.testing.assert_array_equal(waveform[:, 0, 0, 0], [7, 21])


def test_load_waveforms_reports_missing_unit_with_context(tmp_path):
    session = tmp_path / "Session0"
    session.mkdir()

    with pytest.raises(ValueError, match=r"session 0, unit 91"):
        utils.load_waveforms([session], [[91]], _param())


def test_load_waveforms_rejects_inconsistent_shapes(tmp_path):
    session = tmp_path / "Session0"
    session.mkdir()
    _write_waveform(session, 1)
    _write_waveform(session, 2, shape=(5, 3, 2))

    with pytest.raises(ValueError, match=r"session 0, unit 2.*expected"):
        utils.load_waveforms([session], [[1, 2]], _param())


class _FakeWaveforms:
    params = {"ms_before": 1.0, "ms_after": 1.0}

    def __init__(self):
        self._waveforms = np.arange(12, dtype=float).reshape(2, 3, 2)

    def get_data(self):
        return self._waveforms

    def get_waveforms_one_unit(self, unit_id):
        return self._waveforms


class _FakeRandomSpikes:
    @staticmethod
    def get_selected_indices_in_spike_train(unit_id, segment_index):
        return np.array([0, 1])


class _FakeSorting:
    @staticmethod
    def get_unit_spike_train(unit_id):
        return np.array([10, 90])


class _FakeSparsity:
    unit_id_to_channel_indices = {37: np.array([0, 1])}


class _FakeAnalyzer:
    unit_ids = np.array([37])
    sorting = _FakeSorting()
    sparsity = _FakeSparsity()
    sampling_frequency = 1_000

    def __init__(self):
        self.extensions = {
            "random_spikes": _FakeRandomSpikes(),
            "waveforms": _FakeWaveforms(),
        }

    def has_extension(self, name):
        return name in self.extensions

    def get_extension(self, name):
        return self.extensions[name]

    @staticmethod
    def get_num_samples():
        return 100

    @staticmethod
    def get_num_channels():
        return 2

    @staticmethod
    def get_channel_locations():
        return np.array([[0, 0], [10, 20]])


def test_sorting_analyzer_export_needs_no_bombcell_or_quality_extensions(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "spikeinterface", None)
    save_utils.make_UnitMatch_folder_from_sorting_analyzers(
        [_FakeAnalyzer()], tmp_path
    )

    session = tmp_path / "Session0"
    exported = np.load(session / "Unit37_RawSpikes.npy")
    assert exported.shape == (3, 2, 2)
    assert not (session / "bombcell_labels.tsv").exists()
    np.testing.assert_array_equal(
        np.load(session / "channel_locations.npy"),
        [[0, 0, 0], [10, 20, 0]],
    )
    with open(session / "waveform_params.json", encoding="utf-8") as stream:
        assert json.load(stream)["spike_width"] == 2

    with pytest.raises(FileExistsError, match=r"overwrite=True"):
        save_utils.make_UnitMatch_folder_from_sorting_analyzers(
            [_FakeAnalyzer()], tmp_path
        )

    save_utils.make_UnitMatch_folder_from_sorting_analyzers(
        [_FakeAnalyzer()], tmp_path, overwrite=True
    )
    assert (session / "Unit37_RawSpikes.npy").exists()
