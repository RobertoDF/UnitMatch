import os
import warnings
from uuid import uuid4

import numpy as np
import pandas as pd


def check_is_in(test_array, parent_array):
    """
    Checks to see if the test_array is contained within the parent_array

    Parameters
    ----------
    test_array : ndarray
        sub array
    parent_array : ndarray
        parent array which may contain the test_array

    Returns
    -------
    bool
        True if the test_array is within the parent_array
    """
    is_in = (test_array[:, None] == parent_array).all(-1).any(-1)
    return is_in


def _filter_pairs_by_isi(pairs, clus_info, param):
    """
    Returns a boolean mask (len == len(pairs)) where True means the pair should
    be excluded because merging the two same-session units would produce too many
    ISI refractory-period violations.

    Only same-session pairs are evaluated; all cross-session pairs keep False.
    Spike times can be supplied directly in ``clus_info["spike_times"]`` as one
    seconds-based array per UnitMatch unit. Kilosort files remain the fallback.
    """
    isi_exclude = np.zeros(len(pairs), dtype=bool)

    if not param.get("remove_over_merges", True):
        return isi_exclude

    session_ids = np.asarray(clus_info["session_id"])
    same_session = session_ids[pairs[:, 0]] == session_ids[pairs[:, 1]]
    if not np.any(same_session):
        return isi_exclude

    refrac_ms = param.get("isi_viol_refrac_ms", 1.5)
    min_frac = param.get("isi_min_fraction_refractory_violations", 0.01)
    ratio_thrs = param.get("isi_viol_ratio_thrs", 1.5)

    original_ids = clus_info["original_ids"]
    spike_times = clus_info.get("spike_times")
    if spike_times is not None and len(spike_times) != len(session_ids):
        raise ValueError(
            "clus_info['spike_times'] must contain one array per UnitMatch unit."
        )

    ks_dirs = param.get("KS_dirs")
    if spike_times is None and ks_dirs is None:
        warnings.warn(
            "ISI over-merge checking was requested but no spike times are available. "
            "Provide clus_info['spike_times'] or param['KS_dirs'].",
            RuntimeWarning,
            stacklevel=2,
        )
        return isi_exclude

    spike_cache = {}  # sess_id -> (spike_times_sec, spike_clusters) or None

    for pid, pair in enumerate(pairs):
        uid_a, uid_b = int(pair[0]), int(pair[1])
        sess_a = int(session_ids[uid_a])
        sess_b = int(session_ids[uid_b])

        if sess_a != sess_b:
            continue

        if spike_times is not None:
            st_a = np.sort(np.asarray(spike_times[uid_a], dtype=float))
            st_b = np.sort(np.asarray(spike_times[uid_b], dtype=float))
        elif sess_a not in spike_cache:
            try:
                ks_dir = ks_dirs[sess_a]
                st_path = os.path.join(ks_dir, "spike_times.npy")
                sc_path = os.path.join(ks_dir, "spike_clusters.npy")
                if os.path.exists(st_path) and os.path.exists(sc_path):
                    st = np.load(st_path).flatten().astype(np.float64)
                    sc = np.load(sc_path).flatten()
                    sample_rate = 30000.0
                    params_path = os.path.join(ks_dir, "params.py")
                    if os.path.exists(params_path):
                        with open(params_path) as f:
                            for line in f:
                                if "sample_rate" in line and "=" in line:
                                    sample_rate = float(line.split("=")[1].strip())
                                    break
                    spike_cache[sess_a] = (st / sample_rate, sc)
                else:
                    spike_cache[sess_a] = None
            except (OSError, ValueError, EOFError):
                spike_cache[sess_a] = None

        if spike_times is None:
            if spike_cache[sess_a] is None:
                continue

            st_sec, sc = spike_cache[sess_a]
            mask_a = sc == original_ids[uid_a]
            mask_b = sc == original_ids[uid_b]
            if not np.any(mask_a) or not np.any(mask_b):
                continue
            st_a = np.sort(st_sec[mask_a])
            st_b = np.sort(st_sec[mask_b])

        st_merged = np.sort(np.concatenate([st_a, st_b]))

        diffs_a = np.diff(st_a) * 1000
        diffs_b = np.diff(st_b) * 1000
        diffs_merged = np.diff(st_merged) * 1000

        if len(diffs_a) == 0 or len(diffs_b) == 0 or len(diffs_merged) == 0:
            continue

        frac_a = np.sum(diffs_a < refrac_ms) / len(diffs_a)
        frac_b = np.sum(diffs_b < refrac_ms) / len(diffs_b)
        frac_merged = np.sum(diffs_merged < refrac_ms) / len(diffs_merged)

        if frac_merged > min_frac:
            denom = 2 * max(frac_a, frac_b)
            violation_ratio = frac_merged / denom if denom > 0 else np.inf
            if violation_ratio > ratio_thrs:
                isi_exclude[pid] = True

    n_excluded = int(np.sum(isi_exclude))
    if n_excluded:
        print(
            f"ISI check: excluding {n_excluded} same-session pair(s) due to refractory violations."
        )
    return isi_exclude


def get_within_session_merge_groups(
    output_prob_array,
    param,
    clus_info,
    match_threshold=None,
    match_mode="or",
):
    """Return disjoint, ISI-safe unit pairs for session-local merging.

    A pair is proposed when either split-half comparison direction exceeds the
    probability threshold by default. Pairs are considered from highest to
    lowest mean probability, and each unit is included at most once. Run
    UnitMatch again after merging to discover additional over-splits.
    """
    output_prob_array = np.asarray(output_prob_array)
    if output_prob_array.ndim != 2 or output_prob_array.shape[0] != output_prob_array.shape[1]:
        raise ValueError("output_prob_array must be a square matrix.")

    n_units = output_prob_array.shape[0]
    session_ids = np.asarray(clus_info["session_id"])
    original_ids = np.asarray(clus_info["original_ids"])
    if len(session_ids) != n_units or len(original_ids) != n_units:
        raise ValueError(
            "clus_info session_id and original_ids must align with output_prob_array."
        )
    if "spike_times" not in clus_info and "KS_dirs" not in param:
        raise ValueError(
            "Automatic merge proposals require clus_info['spike_times'] "
            "or param['KS_dirs'] for the ISI safety check."
        )

    threshold = (
        param["match_threshold"] if match_threshold is None else match_threshold
    )
    match_mode = match_mode.lower()
    if match_mode == "or":
        pair_mask = (output_prob_array > threshold) | (
            output_prob_array.T > threshold
        )
    elif match_mode == "and":
        pair_mask = (output_prob_array > threshold) & (
            output_prob_array.T > threshold
        )
    else:
        raise ValueError("match_mode must be 'or' or 'and'")
    pair_mask &= session_ids[:, None] == session_ids[None, :]
    candidate_pairs = np.argwhere(np.triu(pair_mask, k=1))
    if candidate_pairs.size == 0:
        return []

    isi_exclude = _filter_pairs_by_isi(candidate_pairs, clus_info, param)
    candidate_pairs = candidate_pairs[~isi_exclude]
    if candidate_pairs.size == 0:
        return []

    mean_probability = np.mean(
        np.column_stack(
            (
                output_prob_array[candidate_pairs[:, 0], candidate_pairs[:, 1]],
                output_prob_array[candidate_pairs[:, 1], candidate_pairs[:, 0]],
            )
        ),
        axis=1,
    )
    order = np.argsort(-mean_probability, kind="stable")

    merge_groups = []
    assigned_units = set()
    for unit_a, unit_b in candidate_pairs[order]:
        if unit_a in assigned_units or unit_b in assigned_units:
            continue
        merge_groups.append(
            [original_ids[unit_a].item(), original_ids[unit_b].item()]
        )
        assigned_units.update((unit_a, unit_b))
    return merge_groups


def _validate_identity_inputs(n_units, clus_info, session_names):
    original_ids = np.asarray(clus_info["original_ids"]).reshape(-1)
    session_ids = np.asarray(clus_info["session_id"]).reshape(-1)
    session_switch = np.asarray(clus_info["session_switch"])
    if len(original_ids) != n_units or len(session_ids) != n_units:
        raise ValueError(
            "clus_info original_ids and session_id must align with "
            "output_prob_array."
        )
    if (
        session_switch.ndim != 1
        or len(session_switch) == 0
        or not np.issubdtype(session_switch.dtype, np.integer)
        or session_switch[0] != 0
        or session_switch[-1] != n_units
        or np.any(np.diff(session_switch) < 0)
    ):
        raise ValueError(
            "clus_info['session_switch'] must be ordered boundaries from 0 "
            "through the number of units."
        )
    if len(original_ids) and not np.issubdtype(original_ids.dtype, np.integer):
        raise ValueError("clus_info['original_ids'] must contain integers.")
    try:
        original_ids = pd.array(original_ids, dtype="Int64")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "clus_info['original_ids'] must fit pandas nullable Int64."
        ) from exc

    n_sessions = len(session_switch) - 1
    if session_names is None:
        session_names = [f"Session{index + 1}" for index in range(n_sessions)]
    else:
        session_names = list(session_names)
        if len(session_names) != n_sessions:
            raise ValueError(
                "session_names must contain one name per session_switch interval."
            )
        if any(not isinstance(name, str) or not name for name in session_names):
            raise ValueError("session_names must contain non-empty strings.")
    if len(set(session_names)) != len(session_names) or "UUID" in session_names:
        raise ValueError("session_names must be unique and cannot include 'UUID'.")

    session_indices = np.searchsorted(
        session_switch[1:], np.arange(n_units), side="right"
    )
    session_labels = []
    for session_index in range(n_sessions):
        start, stop = session_switch[session_index : session_index + 2]
        labels = np.unique(session_ids[start:stop])
        if len(labels) > 1:
            raise ValueError(
                "clus_info session_id disagrees with session_switch in "
                f"session interval {session_index + 1}."
            )
        if len(labels) == 1:
            session_labels.append(labels[0])
    if len(np.unique(session_labels)) != len(session_labels):
        raise ValueError(
            "clus_info session_id must identify each non-empty "
            "session_switch interval uniquely."
        )

    unit_keys = pd.DataFrame(
        {"session": session_indices, "original_id": original_ids}
    )
    duplicates = unit_keys.duplicated(["session", "original_id"], keep=False)
    if duplicates.any():
        duplicate = unit_keys.loc[duplicates].iloc[0]
        raise ValueError(
            "clus_info contains duplicate original_ids within session "
            f"{int(duplicate['session']) + 1}: "
            f"{int(duplicate['original_id'])}."
        )

    return (
        original_ids,
        session_ids,
        session_switch.astype(np.int64, copy=False),
        session_indices,
        session_names,
    )


def _pair_array(pairs, name, n_units=None):
    if pairs is None:
        return np.empty((0, 2), dtype=np.int64)
    if isinstance(pairs, np.ndarray):
        values = pairs
    else:
        try:
            values = np.asarray(list(pairs))
        except TypeError as exc:
            raise ValueError(f"{name} must be an iterable of unit-index pairs.") from exc
    if values.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_pairs, 2).")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer unit indices.")

    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{name} contains a negative unit index.")
    if n_units is not None and np.any(values >= n_units):
        raise ValueError(
            f"{name} contains a unit index outside [0, {n_units})."
        )
    if np.any(values[:, 0] == values[:, 1]):
        raise ValueError(f"{name} cannot contain self-pairs.")
    return np.unique(np.sort(values, axis=1), axis=0)


def _valid_pair_constraint(valid_pairs):
    if valid_pairs is None:
        return None, None
    if isinstance(valid_pairs, np.ndarray) and valid_pairs.ndim == 2:
        if (
            valid_pairs.shape[0] != valid_pairs.shape[1]
            or not np.issubdtype(valid_pairs.dtype, np.bool_)
        ):
            raise ValueError("valid_pairs mask must be a square boolean matrix.")
        return valid_pairs, valid_pairs.shape[0]
    return {
        tuple(pair)
        for pair in _pair_array(valid_pairs, "valid_pairs").tolist()
    }, None


def curate_match_pairs(
    automatic_matches,
    is_match,
    not_match,
    *,
    valid_pairs=None,
):
    """Combine automatic and manual decisions into canonical accepted pairs.

    Every input is treated as an undirected pair collection. Explicit
    rejections override automatic and manual acceptances.

    ``valid_pairs`` constrains only the accepting collections. A rejection
    can only ever remove a pair, so rejecting one that was never eligible is
    a no-op rather than an error: reviewers reach such pairs in the GUI,
    whose Unit B list spans a whole session regardless of probe, and
    aborting a finished review over a harmless click would discard it all.
    """
    constraint, n_units = _valid_pair_constraint(valid_pairs)
    normalized = {
        name: _pair_array(pairs, name, n_units=n_units)
        for name, pairs in (
            ("automatic_matches", automatic_matches),
            ("is_match", is_match),
            ("not_match", not_match),
        )
    }

    for name in ("automatic_matches", "is_match"):
        for unit_a, unit_b in normalized[name]:
            pair = (int(unit_a), int(unit_b))
            if isinstance(constraint, np.ndarray):
                valid = bool(
                    constraint[unit_a, unit_b]
                    or constraint[unit_b, unit_a]
                )
            elif constraint is not None:
                valid = pair in constraint
            else:
                valid = True
            if not valid:
                raise ValueError(
                    f"{name} contains pair {pair} excluded by valid_pairs."
                )

    accepted = {
        tuple(pair)
        for pair in np.vstack(
            (normalized["automatic_matches"], normalized["is_match"])
        ).tolist()
    }
    accepted.difference_update(
        tuple(pair) for pair in normalized["not_match"].tolist()
    )
    if not accepted:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(accepted), dtype=np.int64)


def _intermediate_group_ids_from_pairs(ordered_pairs, n_units, session_ids):
    unique_id = np.arange(n_units, dtype=np.int64)
    if len(ordered_pairs) == 0:
        return unique_id

    accepted_pairs = {tuple(pair) for pair in ordered_pairs.tolist()}
    session_ids = np.asarray(session_ids)

    for unit_a, unit_b in ordered_pairs:
        group_a = np.flatnonzero(unique_id == unique_id[unit_a])
        group_b = np.flatnonzero(unique_id == unique_id[unit_b])
        checks = []
        for member in group_b:
            if member != unit_a:
                checks.append((min(unit_a, member), max(unit_a, member)))
        for member in group_a:
            if member != unit_b:
                checks.append((min(unit_b, member), max(unit_b, member)))

        nearby_checks = [
            pair
            for pair in checks
            if abs(session_ids[pair[0]] - session_ids[pair[1]]) <= 1
        ]
        if not nearby_checks or all(pair in accepted_pairs for pair in nearby_checks):
            group = np.union1d(group_a, group_b)
            unique_id[group] = np.min(unique_id[group])

    return unique_id


def _probability_pairs(output_prob_array, threshold, clus_info, param):
    pairs = np.argwhere(output_prob_array > threshold)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.sort(pairs, axis=1)
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs_unique, count = np.unique(pairs, axis=0, return_counts=True)
    pairs_unique = pairs_unique[count > 1]
    if len(pairs_unique) == 0:
        return pairs_unique

    isi_exclude = _filter_pairs_by_isi(pairs_unique, clus_info, param)
    pairs_unique = pairs_unique[~isi_exclude]
    if len(pairs_unique) == 0:
        return pairs_unique

    prob_mean = np.nanmean(
        np.column_stack(
            (
                output_prob_array[pairs_unique[:, 0], pairs_unique[:, 1]],
                output_prob_array[pairs_unique[:, 1], pairs_unique[:, 0]],
            )
        ),
        axis=1,
    )
    return pairs_unique[np.argsort(-prob_mean, kind="stable")]


def _identity_table(
    group_ids,
    original_ids,
    session_indices,
    session_names,
):
    columns = ["UUID", *session_names]
    records = []
    for group_id in dict.fromkeys(group_ids.tolist()):
        unit_indices = np.flatnonzero(group_ids == group_id)
        group_sessions = session_indices[unit_indices]
        if len(np.unique(group_sessions)) != len(group_sessions):
            conflicting_session = int(
                group_sessions[
                    np.flatnonzero(
                        np.bincount(group_sessions)[group_sessions] > 1
                    )[0]
                ]
            )
            conflicting_ids = [
                int(original_ids[index])
                for index in unit_indices[group_sessions == conflicting_session]
            ]
            raise ValueError(
                "Intermediate identity contains multiple units from "
                f"{session_names[conflicting_session]}: {conflicting_ids}."
            )

        record = {"UUID": str(uuid4())}
        for unit_index, session_index in zip(unit_indices, group_sessions):
            record[session_names[session_index]] = int(original_ids[unit_index])
        records.append(record)

    identities = pd.DataFrame.from_records(records, columns=columns)
    for session_name in session_names:
        identities[session_name] = pd.array(
            identities[session_name], dtype="Int64"
        )
    return identities


def assign_unique_id(
    match_pairs,
    clus_info,
    *,
    session_names=None,
):
    """Assign canonical accepted pairs to intermediate UnitMatch identities.

    A group merge must be supported by all supplied same/adjacent-session
    comparisons. Every input unit is represented, including unmatched
    singleton units.

    Parameters
    ----------
    match_pairs : iterable of pairs
        Accepted undirected pairs using UnitMatch matrix row indices.
    clus_info : dict
        Cluster metadata containing ``original_ids``, ``session_id``, and
        ``session_switch``.
    session_names : sequence of str, optional
        Output column names in ``session_switch`` order. Defaults to
        ``Session1``, ``Session2``, and so on.

    Returns
    -------
    pandas.DataFrame
        Columns are ``UUID`` followed by one nullable ``Int64`` column per
        session. UUID4 values are newly generated on every call; they are not a
        persistent identity registry.
    """
    n_units = len(np.asarray(clus_info["original_ids"]).reshape(-1))
    (
        original_ids,
        _session_ids,
        _session_switch,
        session_indices,
        session_names,
    ) = _validate_identity_inputs(n_units, clus_info, session_names)
    pairs = _pair_array(match_pairs, "match_pairs", n_units=n_units)
    group_ids = _intermediate_group_ids_from_pairs(
        pairs, n_units, session_indices
    )
    return _identity_table(
        group_ids, original_ids, session_indices, session_names
    )


def assign_unique_id_from_probabilities(
    output_prob_array,
    param,
    clus_info,
    *,
    session_names=None,
):
    """Assign identities from bidirectionally thresholded probabilities.

    This preserves the historical probability-based intermediate grouping
    workflow. New post-review code should call :func:`assign_unique_id` with
    curated accepted pairs instead.
    """
    output_prob_array = np.asarray(output_prob_array)
    if (
        output_prob_array.ndim != 2
        or output_prob_array.shape[0] != output_prob_array.shape[1]
    ):
        raise ValueError("output_prob_array must be a square matrix.")
    n_units = output_prob_array.shape[0]
    (
        original_ids,
        _session_ids,
        _session_switch,
        session_indices,
        session_names,
    ) = _validate_identity_inputs(n_units, clus_info, session_names)

    if param.get("use_data_driven_prob_thrs", False):
        stepsz = 0.1
        bin_edges = np.arange(0, 1 + stepsz, stepsz)
        plot_vec = np.arange(stepsz / 2, 1, stepsz)
        histogram, _ = np.histogram(
            np.diag(output_prob_array), bins=len(bin_edges), density=True
        )
        candidates = plot_vec[np.diff(histogram) > 0.1]
        if len(candidates) != 1:
            raise ValueError(
                "Data-driven probability threshold did not resolve to one value."
            )
        threshold = candidates[0]
    else:
        threshold = param["match_threshold"]

    grouping_clus_info = dict(clus_info)
    grouping_clus_info["session_id"] = session_indices
    ordered_pairs = _probability_pairs(
        output_prob_array, threshold, grouping_clus_info, param
    )
    group_ids = _intermediate_group_ids_from_pairs(
        ordered_pairs, n_units, session_indices
    )
    return _identity_table(
        group_ids, original_ids, session_indices, session_names
    )
