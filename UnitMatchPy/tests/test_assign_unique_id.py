import numpy as np
import pytest

from UnitMatchPy.assign_unique_id import _filter_pairs_by_isi


def _params():
    return {
        "remove_over_merges": True,
        "isi_viol_refrac_ms": 1.5,
        "isi_min_fraction_refractory_violations": 0.01,
        "isi_viol_ratio_thrs": 1.5,
    }


def test_isi_filter_uses_direct_spike_times():
    pairs = np.array([[0, 1], [0, 2]])
    clus_info = {
        "session_id": np.array([0, 0, 1]),
        "original_ids": np.array([10, 11, 20]),
        "spike_times": [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.1005, 0.2005, 0.3005]),
            np.array([0.0, 0.1, 0.2]),
        ],
    }

    excluded = _filter_pairs_by_isi(pairs, clus_info, _params())

    np.testing.assert_array_equal(excluded, [True, False])


def test_isi_filter_reports_missing_spike_source():
    clus_info = {
        "session_id": np.array([0, 0]),
        "original_ids": np.array([10, 11]),
    }

    with pytest.warns(RuntimeWarning, match="no spike times are available"):
        excluded = _filter_pairs_by_isi(
            np.array([[0, 1]]), clus_info, _params()
        )

    np.testing.assert_array_equal(excluded, [False])


def test_isi_filter_validates_direct_spike_times():
    clus_info = {
        "session_id": np.array([0, 0]),
        "original_ids": np.array([10, 11]),
        "spike_times": [np.array([0.0, 0.1])],
    }

    with pytest.raises(ValueError, match="one array per UnitMatch unit"):
        _filter_pairs_by_isi(np.array([[0, 1]]), clus_info, _params())
