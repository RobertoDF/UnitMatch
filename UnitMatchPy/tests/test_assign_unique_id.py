import numpy as np
import pytest
from UnitMatchPy.assign_unique_id import (
    _filter_pairs_by_isi,
    curate_match_pairs,
    get_within_session_merge_groups,
)


def _cross_session_same_probe_mask():
    """Four units: two sessions x two probes, ordered (s0p0, s0p1, s1p0, s1p1)."""
    sessions = np.array([0, 0, 1, 1])
    probes = np.array([0, 1, 0, 1])
    return (sessions[:, None] != sessions[None, :]) & (probes[:, None] == probes[None, :])


def test_rejecting_an_ineligible_pair_is_a_no_op():
    mask = _cross_session_same_probe_mask()

    curated = curate_match_pairs(
        [[0, 2]], [], [[1, 2]], valid_pairs=mask,
    )

    np.testing.assert_array_equal(curated, [[0, 2]])


def test_rejections_still_override_accepted_pairs():
    mask = _cross_session_same_probe_mask()

    curated = curate_match_pairs(
        [[0, 2]], [[1, 3]], [[2, 0]], valid_pairs=mask,
    )

    np.testing.assert_array_equal(curated, [[1, 3]])


@pytest.mark.parametrize("field", ["automatic_matches", "is_match"])
def test_accepting_an_ineligible_pair_is_still_refused(field):
    mask = _cross_session_same_probe_mask()
    decisions = {"automatic_matches": [], "is_match": [], "not_match": []}
    decisions[field] = [[1, 2]]

    with pytest.raises(ValueError, match=f"{field} contains pair"):
        curate_match_pairs(
            decisions["automatic_matches"],
            decisions["is_match"],
            decisions["not_match"],
            valid_pairs=mask,
        )


def test_ineligible_rejections_are_still_validated_as_pairs():
    mask = _cross_session_same_probe_mask()

    with pytest.raises(ValueError, match="not_match cannot contain self-pairs"):
        curate_match_pairs([], [], [[1, 1]], valid_pairs=mask)


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


def test_merge_groups_use_or_logic_and_isi_safety():
    probabilities = np.array(
        [
            [1.0, 0.90, 0.95, 0.10],
            [0.80, 1.0, 0.85, 0.10],
            [0.40, 0.70, 1.0, 0.95],
            [0.10, 0.10, 0.90, 1.0],
        ]
    )
    clus_info = {
        "session_id": np.array([0, 0, 0, 0]),
        "original_ids": np.array([10, 11, 12, 13]),
        "spike_times": [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.05, 0.15, 0.25]),
            np.array([0.0, 0.1, 0.2]),
            np.array([0.1005, 0.2005, 0.3005]),
        ],
    }

    groups = get_within_session_merge_groups(
        probabilities, _params(), clus_info, match_threshold=0.5
    )

    assert groups == [[10, 11]]


def test_merge_groups_accept_one_direction_in_or_mode():
    probabilities = np.array([[1.0, 0.9], [0.1, 1.0]])
    clus_info = {
        "session_id": np.array([0, 0]),
        "original_ids": np.array([10, 11]),
        "spike_times": [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.05, 0.15, 0.25]),
        ],
    }

    groups = get_within_session_merge_groups(
        probabilities,
        _params(),
        clus_info,
        match_threshold=0.5,
        match_mode="or",
    )

    assert groups == [[10, 11]]


def test_merge_groups_require_spike_source():
    clus_info = {
        "session_id": np.array([0, 0]),
        "original_ids": np.array([10, 11]),
    }

    with pytest.raises(ValueError, match="ISI safety check"):
        get_within_session_merge_groups(
            np.array([[1.0, 0.9], [0.8, 1.0]]), _params(), clus_info
        )
