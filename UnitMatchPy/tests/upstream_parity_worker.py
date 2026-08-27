import argparse
import json
from pathlib import Path

import numpy as np

from UnitMatchPy import bayes_functions as bf
from UnitMatchPy import default_params, overlord, utils


def run(dataset_dir):
    wave_paths = sorted(dataset_dir.glob("Session*"))
    channel_pos = [np.load(path / "channel_locations.npy") for path in wave_paths]
    with (wave_paths[0] / "waveform_params.json").open(encoding="utf-8") as stream:
        waveform_params = json.load(stream)

    param = default_params.get_default_param()
    param.update(waveform_params)
    param["waveidx"] = np.asarray(param["waveidx"], dtype=int)
    param = utils.get_probe_geometry(channel_pos[0], param)

    unit_ids = []
    session_waveforms = []
    for path in wave_paths:
        ids = sorted(
            int(file.stem.removeprefix("Unit").removesuffix("_RawSpikes"))
            for file in path.glob("Unit*_RawSpikes.npy")
        )
        unit_ids.append(np.asarray(ids))
        session_waveforms.append(
            np.stack([np.load(path / f"Unit{unit_id}_RawSpikes.npy") for unit_id in ids])
        )

    waveform = np.concatenate(session_waveforms)
    counts = np.asarray([len(ids) for ids in unit_ids])
    param["n_units"], session_id, session_switch, param["n_sessions"] = (
        utils.get_session_data(counts)
    )
    param["n_units_per_session"] = counts
    param["n_channels"] = waveform.shape[2]
    within_session = utils.get_within_session(session_id, param)
    clus_info = {
        "good_units": unit_ids,
        "session_switch": session_switch,
        "session_id": session_id,
        "original_ids": np.concatenate(unit_ids),
    }

    properties = overlord.extract_parameters(waveform, channel_pos, clus_info, param)
    total_score, candidate_pairs, scores, predictors = overlord.extract_metric_scores(
        properties, session_switch, within_session, param, niter=2
    )
    prior_match = 1 - param["n_expected_matches"] / param["n_units"] ** 2
    priors = np.asarray((prior_match, 1 - prior_match))
    labels = candidate_pairs.astype(int)
    conditions = np.unique(labels)
    kernels = bf.get_parameter_kernels(
        scores, labels, conditions, param, add_one=1
    )
    probability = bf.apply_naive_bayes(
        kernels, priors, predictors, param, conditions
    )[:, 1].reshape(param["n_units"], param["n_units"])

    return {
        "amplitude": properties["amplitude"],
        "avg_centroid": properties["avg_centroid"],
        "candidate_pairs": candidate_pairs,
        "probability": probability,
        "total_score": total_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    np.savez(arguments.output, **run(arguments.dataset_dir))
