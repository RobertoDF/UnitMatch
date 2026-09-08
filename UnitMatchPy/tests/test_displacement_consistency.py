from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import median_abs_deviation

from UnitMatchPy.displacement_consistency import (
    DisplacementConsistency,
    DisplacementConsistencyResult,
)


def _data(displacements, candidate=(10.0, -5.0), noise=0.0):
    """Independent pairs followed by an unaccepted candidate pair."""
    vectors = np.vstack([displacements, candidate])
    n_pairs = len(vectors)
    rng = np.random.default_rng(723)
    starts = rng.normal(0, 100, size=(n_pairs, 2))
    means = np.concatenate([starts, starts + vectors])
    errors = rng.normal(0, noise, size=means.shape)
    raw = np.zeros((3, 2 * n_pairs, 2))
    raw[1:3, :, 0] = (means + errors).T
    raw[1:3, :, 1] = (means - errors).T
    sessions = np.repeat([0, 1], n_pairs)
    probes = np.zeros(2 * n_pairs, dtype=int)
    pairs = np.column_stack([np.arange(n_pairs - 1), np.arange(n_pairs - 1) + n_pairs])
    return raw, sessions, probes, pairs, (n_pairs - 1, 2 * n_pairs - 1)


def _cloud(n=80, scales=(2.0, 2.0), center=(10.0, -5.0), seed=42):
    return np.random.default_rng(seed).normal(size=(n, 2)) * scales + center


def _model(data, **kwargs):
    return DisplacementConsistency(*data[:4], **kwargs)


def test_consistent_candidate_and_outlier():
    data = _data(_cloud())
    result = _model(data).score(*data[4])
    assert isinstance(result, DisplacementConsistencyResult)
    assert result.reason == ""
    assert 90 < result.score <= 100
    assert result.distance >= 0
    assert result.residual_um >= 0
    assert result.reference_count == 80
    assert result.score == pytest.approx(100 * np.exp(-result.distance**2 / 2))

    outlier = _data(_cloud(), candidate=(30, 20))
    assert _model(outlier).score(*outlier[4]).score < 0.01


def test_robust_to_minority_outlier_references():
    clean = _cloud(100)
    contaminated = np.vstack([clean, _cloud(20, center=(100, 100), seed=80)])
    clean_data = _data(clean)
    outlier_data = _data(contaminated)
    clean_result = _model(clean_data).score(*clean_data[4])
    result = _model(outlier_data).score(*outlier_data[4])
    assert result.score > 90
    assert abs(result.score - clean_result.score) < 10
    assert result.residual_um < 1


def test_anisotropy_penalizes_transverse_not_longitudinal_residual():
    cloud = _cloud(200, scales=(10, 1))
    longitudinal = _data(cloud, candidate=(15, -5))
    transverse = _data(cloud, candidate=(10, 0))
    a = _model(longitudinal).score(*longitudinal[4])
    b = _model(transverse).score(*transverse[4])
    assert a.score > 70
    assert b.score < 1
    assert a.distance < b.distance


def test_zero_displacement_is_valid():
    data = _data(_cloud(center=(0, 0)), candidate=(0, 0))
    assert _model(data).score(*data[4]).score > 90


def test_reverse_orientation_and_global_translation_are_invariant():
    data = _data(_cloud(), candidate=(13, -7), noise=0.5)
    result = _model(data).score(*data[4])
    assert _model(data).score(*reversed(data[4])) == result
    reversed_pairs = list(data)
    reversed_pairs[3] = data[3][:, ::-1]
    assert _model(reversed_pairs).score(*data[4]) == result
    translated = list(data)
    translated[0] = data[0].copy()
    translated[0][1:3] += np.array([50000, -75000])[:, None, None]
    shifted_result = _model(translated).score(*data[4])
    assert shifted_result.score == pytest.approx(result.score, abs=1e-8)
    assert shifted_result.distance == pytest.approx(result.distance, abs=1e-8)
    assert shifted_result.residual_um == pytest.approx(result.residual_um, abs=1e-8)


def test_candidate_itself_and_every_competing_endpoint_are_excluded():
    data = list(_data(_cloud(30)))
    a, b = data[4]
    data[3] = np.vstack([data[3], [a, b], [a, data[3][0, 1]], [data[3][1, 0], b]])
    result = _model(data).score(a, b)
    # The two original references now have ambiguously assigned endpoints.
    assert result.reference_count == 28
    assert result.score is not None


def test_accepted_candidate_is_not_its_own_reference():
    data = list(_data(_cloud(20)))
    data[3] = np.vstack([data[3], data[4]])
    assert _model(data).score(*data[4]).reference_count == 20
    result = _model(data).score(*data[3][0])
    assert result.reference_count == 20
    assert result.reason == ""


def test_canonical_duplicate_pairs_do_not_inflate_count_or_change_fit():
    data = _data(_cloud(25))
    duplicates = list(data)
    duplicates[3] = np.vstack([data[3], data[3], data[3][:, ::-1]])
    assert _model(duplicates).score(*data[4]) == _model(data).score(*data[4])


def test_conflicting_reference_assignments_are_all_removed():
    data = list(_data(_cloud(22)))
    data[3] = np.vstack([data[3], [data[3][0, 0], data[3][1, 1]]])
    result = _model(data).score(*data[4])
    assert result.reference_count == 20
    assert result.score is not None


@pytest.mark.parametrize("count,available", [(4, False), (5, True), (6, True)])
def test_minimum_boundary_after_excluding_accepted_candidate(count, available):
    data = list(_data(_cloud(count), noise=0.5))
    data[3] = np.vstack([data[3], data[4]])
    result = _model(data).score(*data[4])
    assert result.reference_count == count
    assert (result.score is not None) == available
    assert bool(result.reason) != available


def test_configurable_minimum():
    data = _data(_cloud(19))
    assert _model(data, min_references=19).score(*data[4]).score is not None
    result = _model(data, min_references=20).score(*data[4])
    assert result.score is None
    assert "19 < 20" in result.reason


@pytest.mark.parametrize("isolation", ["probe", "session", "shank"])
def test_reference_scope_isolation(isolation):
    data = list(_data(_cloud(40)))
    shanks = np.zeros(len(data[1]), dtype=int)
    excluded = data[3][20:].ravel()
    if isolation == "probe":
        data[2][excluded] = 1
    elif isolation == "session":
        data[1][data[3][20:, 1]] = 2
    else:
        shanks[excluded] = 1
    result = _model(data, shank_ids=shanks).score(*data[4])
    assert result.reference_count == 20
    assert result.score is not None
    if isolation == "shank":
        assert _model(data).score(*data[4]).reference_count == 40


def test_cross_probe_and_cross_shank_references_are_not_pooled():
    data = list(_data(_cloud(22)))
    data[2][data[3][0, 1]] = 1
    shanks = np.zeros(len(data[1]), dtype=int)
    shanks[data[3][1, 1]] = 1
    assert _model(data, shank_ids=shanks).score(*data[4]).reference_count == 20


@pytest.mark.parametrize("kind,reason", [
    ("same_session", "two sessions"),
    ("different_probe", "different probes"),
    ("different_shank", "different shanks"),
    ("missing_session", "Missing session"),
    ("missing_probe", "Missing probe"),
    ("missing_shank", "Missing shank"),
    ("nonfinite", "finite halves"),
])
def test_valid_unsuitable_candidates_return_reason(kind, reason):
    data = list(_data(_cloud(30)))
    data[1] = data[1].astype(float)
    data[2] = data[2].astype(float)
    shanks = np.zeros(len(data[1]))
    a, b = data[4]
    if kind == "same_session":
        data[1][b] = data[1][a]
    elif kind == "different_probe":
        data[2][b] = 1
    elif kind == "different_shank":
        shanks[b] = 1
    elif kind == "missing_session":
        data[1][b] = np.nan
    elif kind == "missing_probe":
        data[2][b] = np.nan
    elif kind == "missing_shank":
        shanks[b] = np.nan
    else:
        data[0][1, a, 1] = np.nan
    result = _model(data, shank_ids=shanks).score(a, b)
    assert result.score is result.distance is result.residual_um is None
    assert reason in result.reason


def test_finite_filtering_requires_both_halves_but_not_unused_z():
    data = list(_data(_cloud(23)))
    data[0][1, data[3][0, 0], 0] = np.nan
    data[0][2, data[3][1, 1], 1] = np.inf
    data[0][1, data[3][2, 0], :] = [np.inf, -np.inf]
    data[0][0] = np.nan
    result = _model(data).score(*data[4])
    assert result.reference_count == 20
    assert result.score is not None


def test_nonfinite_reference_displacements_from_overflow_are_filtered():
    data = list(_data(_cloud(21)))
    a, b = data[3][0]
    data[0][1:3, a] = -1e308
    data[0][1:3, b] = 1e308
    result = _model(data).score(*data[4])
    assert result.reference_count == 20
    assert result.score is not None


def test_nonfinite_conflict_does_not_rescue_an_ambiguous_reference():
    data = list(_data(_cloud(22)))
    data[3] = np.vstack([data[3], [data[3][0, 0], data[3][1, 1]]])
    data[0][1, data[3][1, 1], 0] = np.nan
    result = _model(data).score(*data[4])
    assert result.reference_count == 20
    assert result.score is not None


@pytest.mark.parametrize("cloud", [
    np.tile([10.0, -5.0], (40, 1)),
    np.column_stack([np.linspace(0, 20, 40), np.full(40, -5.0)]),
    _cloud(40, scales=(2, 1e-8)),
])
@pytest.mark.parametrize("noise", [0.0, 0.8])
def test_singular_and_near_singular_clouds_require_physical_uncertainty(cloud, noise):
    data = _data(cloud, noise=noise)
    result = _model(data).score(*data[4])
    if noise:
        assert result.reason == ""
        assert result.score > 70
    else:
        assert result.score is None
        assert "covariance" in result.reason


def test_perfectly_consistent_zero_displacement_with_noise():
    data = _data(np.zeros((40, 2)), candidate=(0, 0), noise=1.0)
    result = _model(data).score(*data[4])
    assert result.score == pytest.approx(100)
    assert result.distance == pytest.approx(0, abs=1e-12)
    assert result.residual_um == pytest.approx(0, abs=1e-12)


def test_floor_is_split_half_mad_variance_divided_by_four():
    data = _data(np.zeros((40, 2)), candidate=(1, 2), noise=1)
    model = _model(data)
    result = model.score(*data[4])
    a, b = data[3].T
    halves = data[0][1:3].transpose(1, 0, 2)
    differences = (halves[b, :, 0] - halves[a, :, 0]) - (
        halves[b, :, 1] - halves[a, :, 1]
    )
    noise_std = median_abs_deviation(differences, axis=0, scale="normal") / 2
    expected_distance = np.linalg.norm(np.array([1, 2]) / noise_std)
    assert result.distance == pytest.approx(expected_distance)
    assert result.residual_um == pytest.approx(np.sqrt(5))


def test_one_axis_noise_cannot_invert_a_zero_scatter_cloud():
    data = list(_data(np.zeros((40, 2)), candidate=(0, 0), noise=1))
    data[0][2, :, 1] = data[0][2, :, 0]
    # Reimpose identical Y displacements without changing X repeatability.
    half = len(data[1]) // 2
    data[0][2, half:] = data[0][2, :half]
    result = _model(data).score(*data[4])
    assert result.score is None
    assert "covariance" in result.reason


def test_dominant_exact_point_population_handles_mcd_zero_scatter():
    cloud = np.vstack([np.zeros((35, 2)), _cloud(5, center=(20, 20))])
    data = _data(cloud, candidate=(0, 0), noise=1)
    result = _model(data).score(*data[4])
    assert result.score == pytest.approx(100)


def test_fit_cache_reuses_identical_endpoint_excluded_references():
    data = _data(_cloud(30))
    model = _model(data)
    with patch.object(model, "_fit_references", wraps=model._fit_references) as fit:
        result = model.score(*data[4])
        assert model.score(*data[4]) == result
        assert model.score(*reversed(data[4])) == result
        assert fit.call_count == 1
        model.score(*data[3][0])
        assert fit.call_count == 2
    assert len(_model(data)._fit_cache) == 0


def test_constructor_does_not_fit():
    with patch.object(DisplacementConsistency, "_fit_references") as fit:
        model = _model(_data(_cloud(30)))
        fit.assert_not_called()
        assert len(model._fit_cache) == 0


def test_unavailable_fits_are_cached_and_cache_is_bounded():
    data = _data(np.zeros((25, 2)), candidate=(0, 0))
    model = _model(data)
    model._CACHE_SIZE = 2
    with patch.object(model, "_fit_references", wraps=model._fit_references) as fit:
        model.score(*data[4])
        model.score(*data[4])
        assert fit.call_count == 1
        for pair in data[3][:3]:
            model.score(*pair)
        assert fit.call_count == 4
    assert len(model._fit_cache) == 2


def test_different_candidates_with_identical_references_share_a_fit():
    data = list(_data(_cloud(31)))
    second_candidate = tuple(data[3][-1])
    data[3] = data[3][:-1]
    model = _model(data)
    with patch.object(model, "_fit_references", wraps=model._fit_references) as fit:
        model.score(*data[4])
        model.score(*second_candidate)
        assert fit.call_count == 1


def test_snapshot_is_independent_and_inputs_and_results_are_not_mutated():
    data = _data(_cloud(30), noise=1)
    shanks = np.zeros(len(data[1]), dtype=int)
    originals = [value.copy() for value in data[:4]]
    model = _model(data, shank_ids=shanks)
    expected = model.score(*data[4])
    for value, original in zip(data[:4], originals):
        np.testing.assert_array_equal(value, original)
    with pytest.raises(FrozenInstanceError):
        expected.score = 0
    for value in data[:4]:
        value[...] = 0
    shanks[:] = 99
    model._fit_cache.clear()
    assert model.score(*data[4]) == expected


def test_read_only_input_arrays_are_supported_without_copy_back():
    data = _data(_cloud(30), noise=1)
    for value in data[:4]:
        value.setflags(write=False)
    assert _model(data).score(*data[4]).score is not None
    assert all(not value.flags.writeable for value in data[:4])


def test_finite_extreme_candidate_has_zero_score_without_distance_overflow():
    data = _data(_cloud(30), candidate=(1e200, 1e200))
    result = _model(data).score(*data[4])
    assert result.score == 0
    assert np.isfinite(result.distance)
    assert np.isfinite(result.residual_um)


def test_unrepresentable_candidate_residual_returns_na():
    data = list(_data(_cloud(30)))
    a, b = data[4]
    data[0][1:3, a] = -1e308
    data[0][1:3, b] = 1e308
    result = _model(data).score(*data[4])
    assert result.score is None
    assert "numerical range" in result.reason


@pytest.mark.parametrize("index", [-1, 1000, 1.5, True, "1", np.nan])
def test_invalid_candidate_indices_raise_clear_errors(index):
    data = _data(_cloud(20))
    with pytest.raises((ValueError, IndexError), match="unit index|unit indices"):
        _model(data).score(index, data[4][1])


@pytest.mark.parametrize("minimum", [0, 1, 2, -1, 1.5, True])
def test_invalid_minimum_rejected(minimum):
    with pytest.raises(ValueError, match="min_references"):
        _model(_data(_cloud(20)), min_references=minimum)


def test_invalid_array_shapes_rejected():
    data = _data(_cloud(20))
    with pytest.raises(ValueError, match="shape"):
        DisplacementConsistency(data[0][:2], *data[1:4])
    with pytest.raises(ValueError, match="session_ids"):
        DisplacementConsistency(data[0], data[1][:-1], *data[2:4])
    with pytest.raises(ValueError, match="probe_ids"):
        DisplacementConsistency(data[0], data[1], data[2][:-1], data[3])
    with pytest.raises(ValueError, match="shank_ids"):
        _model(data, shank_ids=[0])


def test_string_scope_identifiers_and_empty_references():
    data = list(_data(_cloud(20)))
    data[1] = np.where(data[1] == 0, "later lexical session", "earlier lexical session")
    data[2] = np.repeat("probe-A", len(data[2]))
    assert _model(data).score(*data[4]).score is not None
    data[3] = []
    result = _model(data).score(*data[4])
    assert result.reference_count == 0
    assert "Insufficient" in result.reason
