from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

si = pytest.importorskip("spikeinterface.full")

from UnitMatchPy.default_params import get_default_param
from UnitMatchPy.save_utils import make_UnitMatch_folder_from_spikeinterface


def _make_recording_and_sorting():
    recording = si.NumpyRecording(
        [np.zeros((1_000, 4), dtype=np.float32)],
        sampling_frequency=30_000,
    )
    recording.set_channel_locations(
        np.array([[0, 0], [0, 20], [0, 40], [0, 60]])
    )
    sorting = si.NumpySorting.from_unit_dict(
        {0: np.array([100, 300, 700]), 1: np.array([200, 500, 800])},
        sampling_frequency=30_000,
    )
    sorting.register_recording(recording)
    return recording, sorting


def test_sorting_input_uses_unitmatch_channel_radius(monkeypatch, tmp_path):
    recording, sorting = _make_recording_and_sorting()
    converted_analyzer = Mock(spec=si.SortingAnalyzer)
    converted_analyzer.has_extension.side_effect = RuntimeError("converted")
    create_analyzer = Mock(return_value=converted_analyzer)
    monkeypatch.setattr(si, "create_sorting_analyzer", create_analyzer)

    with pytest.raises(RuntimeError, match="converted"):
        make_UnitMatch_folder_from_spikeinterface([sorting], tmp_path)

    create_analyzer.assert_called_once_with(
        sorting=sorting,
        recording=recording,
        sparse=True,
        method="radius",
        radius_um=get_default_param()["channel_radius"],
    )


def test_missing_extensions_are_computed(monkeypatch, tmp_path):
    recording, sorting = _make_recording_and_sorting()
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    compute = Mock(side_effect=RuntimeError("computed"))
    monkeypatch.setattr(analyzer, "compute", compute)

    with pytest.raises(RuntimeError, match="computed"):
        make_UnitMatch_folder_from_spikeinterface([analyzer], tmp_path)

    compute.assert_called_once_with(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "template_metrics",
            "quality_metrics",
        ]
    )


def test_dense_analyzer_exports_all_channels(monkeypatch, tmp_path):
    recording, sorting = _make_recording_and_sorting()
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer.compute("random_spikes", max_spikes_per_unit=3)
    analyzer.compute("waveforms", ms_before=0.1, ms_after=0.1)
    analyzer.compute("templates")
    analyzer.compute("template_metrics")
    analyzer.compute("quality_metrics")

    labels = pd.DataFrame(
        {"bombcell_label": ["good", "good"]},
        index=sorting.unit_ids,
    )
    monkeypatch.setattr(si, "bombcell_label_units", lambda _: labels)

    make_UnitMatch_folder_from_spikeinterface([analyzer], tmp_path)

    session_dir = tmp_path / "Session0"
    waveform = np.load(session_dir / "Unit0_RawSpikes.npy")
    assert waveform.shape[1] == recording.get_num_channels()
    assert (session_dir / "bombcell_labels.tsv").is_file()
    assert (session_dir / "channel_locations.npy").is_file()
    assert (session_dir / "waveform_params.json").is_file()
