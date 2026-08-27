import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from UnitMatchPy.save_utils import make_UnitMatch_folder_from_sorting_analyzers

UPSTREAM_URL = "https://github.com/EnnyvanBeest/UnitMatch.git"


def _make_analyzer(seed):
    import spikeinterface.full as si

    recording, sorting = si.generate_ground_truth_recording(
        durations=[5.0],
        sampling_frequency=30_000.0,
        num_channels=16,
        num_units=8,
        seed=seed,
    )
    analyzer = si.create_sorting_analyzer(
        sorting=sorting, recording=recording, format="memory", sparse=True
    )
    analyzer.compute(
        "random_spikes", method="uniform", max_spikes_per_unit=200, seed=seed
    )
    analyzer.compute("waveforms", ms_before=1.0, ms_after=1.0)
    return analyzer


def _run_worker(source_root, dataset_dir, output):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root / "UnitMatchPy")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("upstream_parity_worker.py")),
            str(dataset_dir),
            str(output),
        ],
        check=True,
        env=env,
    )
    return np.load(output)


@pytest.mark.skipif(
    os.environ.get("UNITMATCH_RUN_UPSTREAM_PARITY") != "1",
    reason="set UNITMATCH_RUN_UPSTREAM_PARITY=1 to clone and compare upstream main",
)
def test_synthetic_spikeinterface_pipeline_matches_upstream_main(tmp_path):
    analyzers = [_make_analyzer(410), _make_analyzer(411)]
    dataset_dir = tmp_path / "unitmatch-data"
    make_UnitMatch_folder_from_sorting_analyzers(analyzers, dataset_dir)

    upstream = tmp_path / "upstream"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "main", UPSTREAM_URL, str(upstream)],
        check=True,
    )
    branch = _run_worker(
        Path(__file__).parents[2], dataset_dir, tmp_path / "branch.npz"
    )
    baseline = _run_worker(upstream, dataset_dir, tmp_path / "upstream.npz")

    np.testing.assert_array_equal(
        branch["candidate_pairs"], baseline["candidate_pairs"]
    )
    for output in ("amplitude", "avg_centroid", "total_score", "probability"):
        np.testing.assert_allclose(
            branch[output], baseline[output], rtol=1e-7, atol=1e-9, equal_nan=True
        )
