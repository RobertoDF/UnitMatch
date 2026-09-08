import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _new_identity_lookup(identities):
    if "UUID" not in identities.columns:
        raise ValueError("Identity DataFrame must contain a 'UUID' column.")
    if identities["UUID"].isna().any() or identities["UUID"].duplicated().any():
        raise ValueError("Identity DataFrame UUID values must be present and unique.")
    session_columns = [column for column in identities.columns if column != "UUID"]
    lookup_parts = []
    for session_number, column in enumerate(session_columns, start=1):
        part = identities.loc[identities[column].notna(), ["UUID", column]].copy()
        part["RecSes"] = session_number
        part = part.rename(columns={column: "OriginalID"})
        try:
            part["OriginalID"] = pd.array(part["OriginalID"], dtype="Int64")
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Identity column {column!r} must contain nullable integers."
            ) from exc
        lookup_parts.append(part)

    if lookup_parts:
        lookup = pd.concat(lookup_parts, ignore_index=True)
    else:
        lookup = pd.DataFrame(columns=["UUID", "OriginalID", "RecSes"])
        lookup["OriginalID"] = pd.array([], dtype="Int64")
    duplicates = lookup.duplicated(["RecSes", "OriginalID"], keep=False)
    if duplicates.any():
        duplicate = lookup.loc[duplicates].iloc[0]
        raise ValueError(
            "UnitIdentities contains multiple UUID mappings for session "
            f"{int(duplicate['RecSes'])}, original ID "
            f"{int(duplicate['OriginalID'])}."
        )
    return lookup


def _add_identity_columns(match_table, identities):
    lookup = _new_identity_lookup(identities)
    for endpoint in (1, 2):
        endpoint_lookup = lookup.rename(
            columns={
                "UUID": f"UUID{endpoint}",
                "OriginalID": f"ID{endpoint}",
                "RecSes": f"RecSes {endpoint}",
            }
        )
        match_table = match_table.merge(
            endpoint_lookup,
            on=[f"RecSes {endpoint}", f"ID{endpoint}"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if match_table[f"UUID{endpoint}"].isna().any():
            missing = match_table.loc[
                match_table[f"UUID{endpoint}"].isna(),
                [f"RecSes {endpoint}", f"ID{endpoint}"],
            ].iloc[0]
            raise ValueError(
                "UnitIdentities is missing a mapping for session "
                f"{int(missing[f'RecSes {endpoint}'])}, original ID "
                f"{int(missing[f'ID{endpoint}'])}."
            )
    return match_table


def _match_table_session_numbers(clus_info, n_units):
    if "session_switch" not in clus_info:
        return np.asarray(clus_info["session_id"]).reshape(-1) + 1

    session_switch = np.asarray(clus_info["session_switch"])
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
    return (
        np.searchsorted(
            session_switch[1:], np.arange(n_units), side="right"
        )
        + 1
    )


def save_auc_summary(save_dir, auc_summary):
    """
    Save the scalar AUC values computed for this output (one per functional
    score) as a small standalone JSON file. Otherwise these numbers only ever
    get printed to console/logs -- MatchTable.csv holds the raw per-pair
    functional-score matrices, which let you recompute an AUC but don't store
    the number itself anywhere discoverable alongside the rest of the output.
    """
    with open(os.path.join(save_dir, "AUC_summary.json"), "w") as f:
        json.dump(auc_summary, f, indent=2)


def make_match_table(
    scores_to_include,
    matches,
    output_prob,
    total_score,
    output_threshold,
    clus_info,
    param,
    UIDs=None,
    matches_curated=None,
    functional_scores=None,
):
    """
    Creates the match table showing information for every pair of units.

    Parameters
    ----------
    scores_to_include : dictionary
        The dictionary containing the used scores and the values for each pair of units
    matches : list
        The list of all matches
    output_prob : ndarray (n_units, n_units)
        The array containing the UnitMatch match probability for each pair of units for cv 1/2 and 2/1
    total_score : ndarray (n_units, n_units)
        The total score for each pair of units being a match for each pair of units for cv 1/2 and 2/1
    output_threshold : int
        The probability threshold value for deciding matches
    clus_info : dict
        The clus_info dictionary
    param : dict
        The param dictionary
    UIDs : pandas.DataFrame or list, optional
        The identity table from assign_unique_id. Legacy four-array UID input
        remains supported for loading/saving older workflows.
    matches_curated : list, optional
        A list of matches manually curated using the GUI, by default None

    Returns
    -------
    dataframe
        A pandas dataframe as a match table for each pair of units
    """
    # Making Match Table
    n_units = param["n_units"]

    # Give UMID add it as well?
    # xx, yy =np.meshgrid(np.arange(nUnits), np.arange(nUnits))
    # UnitA = np.reshape(xx, (nUnits * nUnits)).astype(np.int16)
    # UnitB = np.reshape(yy, (nUnits * nUnits)).astype(np.int16)

    original_ids = np.asarray(clus_info["original_ids"]).reshape(-1)
    xx, yy = np.meshgrid(original_ids, original_ids)
    unit_a_list = xx.reshape(n_units * n_units)
    unit_b_list = yy.reshape(n_units * n_units)

    session_numbers = _match_table_session_numbers(clus_info, n_units)
    xx, yy = np.meshgrid(session_numbers, session_numbers)
    unit_a_session_list = xx.reshape(n_units * n_units)
    unit_b_session_list = yy.reshape(n_units * n_units)

    all_matches = np.reshape(output_threshold, (n_units * n_units)).astype(
        np.int8
    )  # Uses Matches currated .. as well
    total_score_list = np.reshape(total_score, (n_units * n_units))
    prob_list = np.reshape(output_prob, (n_units * n_units))

    # create the initial array with the important info
    # see if there is a curated list and if soo add it
    if matches_curated is not None:
        matches_curated_list = np.zeros((n_units, n_units))
        for match in matches:
            matches_curated_list[match[0], match[1]] = 1

        matches_curated_list = np.reshape(matches_curated, (n_units * n_units)).astype(
            np.int8
        )

        df = pd.DataFrame({
            "ID1": unit_a_list,
            "ID2": unit_b_list,
            "RecSes 1": unit_a_session_list,
            "RecSes 2": unit_b_session_list,
            "Matches": all_matches,
            "Matches Currated": matches_curated_list,
            "UM Probabilities": prob_list,
            "TotalScore": total_score_list,
        })

    else:
        df = pd.DataFrame({
            "ID1": unit_a_list,
            "ID2": unit_b_list,
            "RecSes 1": unit_a_session_list,
            "RecSes 2": unit_b_session_list,
            "Matches": all_matches,
            "UM Probabilities": prob_list,
            "TotalScore": total_score_list,
        })

    # add a dictionary to the match table
    for key, value in scores_to_include.items():
        df[key] = np.reshape(value, (n_units * n_units)).T

    # add per-pair functional scores (n_units × n_units matrices) as columns
    if functional_scores is not None:
        for key, value in functional_scores.items():
            df[key] = np.reshape(value, (n_units * n_units))

    # if you have supplied UIDs create a data frame using them and merge it to the save table
    if UIDs is not None:
        if isinstance(UIDs, pd.DataFrame):
            return _add_identity_columns(df, UIDs)

        unique_id_liberal = UIDs[0]
        unique_id = UIDs[1]
        unique_id_conservative = UIDs[2]
        original_unique_id = UIDs[3]

        xx, yy = np.meshgrid(unique_id_liberal, unique_id_liberal)
        unit_a_liberal_id = xx.reshape(n_units * n_units)
        unit_b_liberal_id = yy.reshape(n_units * n_units)

        xx, yy = np.meshgrid(original_unique_id, original_unique_id)
        unit_a_original_id = xx.reshape(n_units * n_units)
        unit_b_original_id = yy.reshape(n_units * n_units)

        xx, yy = np.meshgrid(unique_id_conservative, unique_id_conservative)
        unit_a_conservative_id = xx.reshape(n_units * n_units)
        unit_b_conservative_id = yy.reshape(n_units * n_units)

        xx, yy = np.meshgrid(unique_id, unique_id)
        unit_a_int_id = xx.reshape(n_units * n_units)
        unit_b_int_id = yy.reshape(n_units * n_units)

        unique_id_df = pd.DataFrame(
            np.array(
                [
                    unit_a_original_id,
                    unit_b_original_id,
                    unit_a_liberal_id,
                    unit_b_liberal_id,
                    unit_a_int_id,
                    unit_b_int_id,
                    unit_a_conservative_id,
                    unit_b_conservative_id,
                ]
            ).T,
            columns=[
                "UID orig 1",
                "UID orig 2",
                "UID Lib 1",
                "UID Lib 2",
                "UID 1",
                "UID 2",
                "UID Cons 1",
                "UID Cons 2",
            ],
        )
        df = df.join(unique_id_df)

    return df


def save_to_output(
    save_dir,
    scores_to_include=None,
    matches=None,
    output_prob=None,
    avg_centroid=None,
    avg_waveform=None,
    avg_waveform_per_tp=None,
    max_site=None,
    total_score=None,
    output_threshold=None,
    clus_info=None,
    param=None,
    UIDs=None,
    matches_curated=None,
    save_match_table=True,
    functional_scores=None,
    *,
    review_snapshot=None,
):
    """
    Saves all useful information calculated by UnitMatch to a given save_dir

    Parameters
    ----------
    save_dir : str
        The absolute path to the save directory
    scores_to_include : dictionary
        The dictionary containing the used scores and the values for each pair of units
    matches : list
        The list of all matches
    output_prob : ndarray (n_units, n_units)
        The array containing the UnitMatch match probability for each pair of units for cv 1/2 and 2/1
    avg_centroid : _type_
        _description_
    avg_waveform : ndarray (n_units, spike_width, cv)
        The weighted average waveform over channels
    avg_waveform_per_tp : ndarray
        The average waveform per time point
    max_site : ndarray
        The maximum site for each unit
    total_score : ndarray (n_units, n_units)
        The total score for each pair of units being a match for each pair of units for cv 1/2 and 2/1
    output_threshold : int
        The probability threshold value for deciding matches
    clus_info : dict
        The clus_info dictionary
    param : dict
        The param dictionary
        _description_
    UIDs : pandas.DataFrame or list, optional
        Identity table from assign_unique_id, or legacy four-array UID input.
    matches_curated : list, optional
        A list of matches manually curated using the GUI, by default None
    save_match_table : bool, optional
        If True will save a match table containing the information for every pair of units in a table, by default True
    review_snapshot : AcceptedReviewSnapshot, optional
        Write the compact accepted-review archive instead of the legacy dense
        output. In this mode, do not provide any legacy scientific arguments.

    Returns
    -------
    pathlib.Path or None
        The compact archive path in review-snapshot mode. Legacy mode preserves
        its historical ``None`` return value.
    """

    if review_snapshot is not None:
        legacy_values = (
            scores_to_include,
            matches,
            output_prob,
            avg_centroid,
            avg_waveform,
            avg_waveform_per_tp,
            max_site,
            total_score,
            output_threshold,
            clus_info,
            param,
            UIDs,
            matches_curated,
            functional_scores,
        )
        if any(value is not None for value in legacy_values):
            raise TypeError(
                "Compact review_snapshot mode cannot be combined with legacy "
                "scientific output arguments."
            )
        from UnitMatchPy.review_archive import save_review_output

        return save_review_output(save_dir, review_snapshot)

    required = {
        "scores_to_include": scores_to_include,
        "matches": matches,
        "output_prob": output_prob,
        "avg_centroid": avg_centroid,
        "avg_waveform": avg_waveform,
        "avg_waveform_per_tp": avg_waveform_per_tp,
        "max_site": max_site,
        "total_score": total_score,
        "output_threshold": output_threshold,
        "clus_info": clus_info,
        "param": param,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(
            "Legacy save_to_output mode is missing required arguments: "
            + ", ".join(missing)
        )

    # Choose a file where the save directory will be made
    # options for create and overwrite?
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    # save scores
    UM_scores_path = os.path.join(save_dir, "UM Scores")
    np.savez(UM_scores_path, **scores_to_include)

    # #save ClusInfo
    clus_info_path = os.path.join(save_dir, "ClusInfo.pickle")
    with open(clus_info_path, "wb") as fp:
        pickle.dump(clus_info, fp)

    # Save param
    param_path = os.path.join(save_dir, "UMparam.pickle")
    with open(param_path, "wb") as fp:
        pickle.dump(param, fp)

    # Save output
    match_prob_path = os.path.join(save_dir, "MatchProb")
    # MAY WANT TO CHANGE TOSAVE PROB FOR BOTH CV AND AVG?
    np.save(match_prob_path, output_prob)

    # Save Waveform info
    waveform_info = {
        "avg_centroid": avg_centroid,
        "avg_waveform": avg_waveform,
        "avg_waveform_per_tp": avg_waveform_per_tp,
        "max_site": max_site,
    }
    waveform_info_path = os.path.join(save_dir, "WaveformInfo")
    np.savez(waveform_info_path, **waveform_info)

    # save autimatuc matches
    matches_path = os.path.join(save_dir, "Matches")
    np.save(matches_path, matches)

    if matches_curated is not None:
        matches_curated_path = os.path.join(save_dir, "Matches Currated")
        np.save(matches_curated_path, matches_curated)

    if isinstance(UIDs, pd.DataFrame):
        UIDs.to_csv(os.path.join(save_dir, "UnitIdentities.csv"), index=False)

    if save_match_table == True:
        df = make_match_table(
            scores_to_include,
            matches,
            output_prob,
            total_score,
            output_threshold,
            clus_info,
            param,
            UIDs=UIDs,
            matches_curated=None,
            functional_scores=functional_scores,
        )
        match_table_path = os.path.join(save_dir, "MatchTable.csv")
        df.to_csv(match_table_path, index=False)


def save_to_output_seperate_CV(
    save_dir,
    scores_to_include,
    matches,
    output_prob,
    avg_centroid,
    avg_waveform,
    avg_waveform_per_tp,
    max_site,
    total_score,
    match_threshold,
    clus_info,
    param,
    UIDs=None,
    matches_curated=None,
    save_match_table=True,
):
    """
    Saves all useful information calculated by UnitMatch to a given save_dir.
    This function will split the n_unit * n_units arrays into separate arrays depending on which pair of CV is used

    Parameters
    ----------
    save_dir : str
        The absolute path to the save directory
    scores_to_include : dictionary
        The dictionary containing the used scores and the values for each pair of units
    matches : list
        The list of all matches
    output_prob : ndarray (n_units, n_units)
        The array containing the UnitMatch match probability for each pair of units for cv 1/2 and 2/1
    avg_centroid : _type_
        _description_
    avg_waveform : ndarray (n_units, spike_width, cv)
        The weighted average waveform over channels
    avg_waveform_per_tp : ndarray
        The average waveform per time point
    max_site : ndarray
        The maximum site for each unit
    total_score : ndarray (n_units, n_units)
        The total score for each pair of units being a match for each pair of units for cv 1/2 and 2/1
    output_threshold : int
        The probability threshold value for deciding matches
    clus_info : dict
        The clus_info dictionary
    param : dict
        The param dictionary
        _description_
    UIDs : pandas.DataFrame or list, optional
        Identity table from assign_unique_id, or legacy four-array UID input.
    matches_curated : list, optional
        A list of matches manually curated using the GUI, by default None
    save_match_table : bool, optional
        If True will save a match table containing the infomation for every pair of units in a table, by default True
    """

    # Start by separating the info into CV
    matches12_part1 = np.argwhere(np.tril(output_prob) > match_threshold)
    matches12_part2 = np.argwhere(np.tril(output_prob).T > match_threshold)
    matches12 = np.unique(np.concatenate((matches12_part1, matches12_part2)), axis=0)

    matches21_part1 = np.argwhere(np.triu(output_prob) > match_threshold)
    matches21_part2 = np.argwhere(np.triu(output_prob).T > match_threshold)
    matches21 = np.unique(np.concatenate((matches21_part1, matches21_part2)), axis=0)

    output12_tmp1 = np.tril(output_prob)
    output12_tmp2 = np.tril(output_prob).T
    np.fill_diagonal(output12_tmp2, 0)
    output12 = output12_tmp1 + output12_tmp2

    output21_tmp1 = np.triu(output_prob)
    output21_tmp2 = np.triu(output_prob).T
    np.fill_diagonal(output21_tmp2, 0)
    output21 = output21_tmp1 + output21_tmp2

    scores_to_include12 = {}
    scores_to_include21 = {}
    for key, value in scores_to_include.items():
        tmp1 = np.tril(value)
        tmp2 = np.tril(value).T
        np.fill_diagonal(tmp2, 0)

        tmp3 = np.triu(value)
        tmp4 = np.triu(value).T
        np.fill_diagonal(tmp4, 0)

        scores_to_include12[key] = tmp1 + tmp2
        scores_to_include21[key] = tmp3 + tmp4

    # Now can save all of these like above
    # Choose a file where the save directory will be made
    # options for create and overwrite?
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    # save scores
    UM_scores_path_cv12 = os.path.join(save_dir, "UM Scores CV12")
    np.savez(UM_scores_path_cv12, **scores_to_include12)
    UM_scores_path_cv21 = os.path.join(save_dir, "UM Scores CV21")
    np.savez(UM_scores_path_cv21, **scores_to_include21)

    # #save ClusInfo
    clus_info_path = os.path.join(save_dir, "ClusInfo.pickle")
    with open(clus_info_path, "wb") as fp:
        pickle.dump(clus_info, fp)

    # Save param
    param_path = os.path.join(save_dir, "UMparam.pickle")
    with open(param_path, "wb") as fp:
        pickle.dump(param, fp)

    # Save output n_unit*n_units probability array
    match_prob_path_cv12 = os.path.join(save_dir, "MatchProb CV12")
    np.save(match_prob_path_cv12, output12)
    match_prob_path_cv21 = os.path.join(save_dir, "MatchProb CV21")
    np.save(match_prob_path_cv21, output21)

    # Save Waveform info
    waveform_info = {
        "avg_centroid": avg_centroid,
        "avg_waveform": avg_waveform,
        "avg_waveform_per_tp": avg_waveform_per_tp,
        "max_site": max_site,
    }
    wavefrom_info_path = os.path.join(save_dir, "WaveformInfo")
    np.savez(wavefrom_info_path, **waveform_info)

    # save automatic matches
    matches_path_cv12 = os.path.join(save_dir, "Matches CV12")
    np.save(matches_path_cv12, matches12)
    matches_path_cv21 = os.path.join(save_dir, "Matches CV21")
    np.save(matches_path_cv21, matches21)

    if matches_curated is not None:
        MatchesCuratedPath = os.path.join(save_dir, "Matches Currated")
        np.save(MatchesCuratedPath, matches_curated)

    if isinstance(UIDs, pd.DataFrame):
        UIDs.to_csv(os.path.join(save_dir, "UnitIdentities.csv"), index=False)

    output_threshold = np.zeros_like(output_prob)
    output_threshold[output_prob > match_threshold] = 1

    if save_match_table == True:
        df = make_match_table(
            scores_to_include,
            matches,
            output_prob,
            total_score,
            output_threshold,
            clus_info,
            param,
            UIDs=UIDs,
            matches_curated=None,
        )
        match_table_path = os.path.join(save_dir, "MatchTable.csv")
        df.to_csv(match_table_path, index=False)


def load_output(save_dir, load_match_table=True):
    """
    Will load all the information in the save directory

    Parameters
    ----------
    save_dir : str
        The absolute path to the save directory
    load_match_table : bool, optional
        If True will load in the match table, by default True

    Returns
    -------
    All data saved in the UM save directory
    """

    # load scores
    UM_scores_path = os.path.join(save_dir, "UM Scores.npz")
    UM_scores = dict(np.load(UM_scores_path))

    # load ClusInfo
    clus_info_path = os.path.join(save_dir, "ClusInfo.pickle")
    with open(clus_info_path, "rb") as fp:
        clus_info = pickle.load(fp)

    # load param
    param_path = os.path.join(save_dir, "UMparam.pickle")
    with open(param_path, "rb") as fp:
        param = pickle.load(fp)

    # load output
    match_prob_path = os.path.join(save_dir, "MatchProb.npy")
    match_prob = np.load(match_prob_path)

    # Load Waveform info
    wavefrom_info_path = os.path.join(save_dir, "WaveformInfo.npz")
    wavefrom_info = dict(np.load(wavefrom_info_path))

    if load_match_table == True:
        match_table_path = os.path.join(save_dir, "MatchTable.csv")
        match_table = pd.read_csv(match_table_path)

        return UM_scores, clus_info, param, match_prob, wavefrom_info, match_table
    return UM_scores, clus_info, param, wavefrom_info, match_prob


def load_output_separate_CV(save_dir, load_match_table=True):
    """
    Will load all the information in the save directory.
    This function will load in data if the CV are saved separately

    Parameters
    ----------
    save_dir : str
        The absolute path to the save directory
    load_match_table : bool, optional
        If True will load in the match table, by default True

    Returns
    -------
    All data saved in the UM save directory
    """

    UM_scores_path_cv12 = os.path.join(save_dir, "UM Scores CV12.npz")
    UM_scores_cv12 = dict(np.load(UM_scores_path_cv12))
    UM_scores_path_cv21 = os.path.join(save_dir, "UM Scores CV21.npz")
    UM_scores_cv21 = dict(np.load(UM_scores_path_cv21))

    # #Load ClusInfo
    clus_info_path = os.path.join(save_dir, "ClusInfo.pickle")
    with open(clus_info_path, "rb") as fp:
        clus_info = pickle.load(fp)

    # Load param
    param_path = os.path.join(save_dir, "UMparam.pickle")
    with open(param_path, "rb") as fp:
        param = pickle.load(fp)

    # Load output nUnit*nUnits probabilities array
    match_prob_path_cv12 = os.path.join(save_dir, "MatchProb CV12.npy")
    match_prob_cv12 = np.load(match_prob_path_cv12)
    match_prob_path_cv21 = os.path.join(save_dir, "MatchProb CV21.npy")
    match_prob_cv21 = np.load(match_prob_path_cv21)

    # Load Waveform info
    wavefrom_info_path = os.path.join(save_dir, "WaveformInfo.npz")
    wavefrom_info = dict(np.load(wavefrom_info_path))

    # save automatic matches
    matches_path_cv12 = os.path.join(save_dir, "Matches CV12.npy")
    matches_cv12 = np.load(matches_path_cv12)
    matches_path_cv21 = os.path.join(save_dir, "Matches CV21.npy")
    matches_cv21 = np.load(matches_path_cv21)

    if load_match_table == True:
        match_table_path = os.path.join(save_dir, "MatchTable.csv")
        match_table = pd.read_csv(match_table_path)

        return (
            UM_scores_cv12,
            UM_scores_cv21,
            clus_info,
            param,
            match_prob_cv12,
            match_prob_cv21,
            matches_cv12,
            matches_cv21,
            wavefrom_info,
            match_table,
        )
    return (
        UM_scores_cv12,
        UM_scores_cv21,
        clus_info,
        param,
        match_prob_cv12,
        match_prob_cv21,
        matches_cv12,
        matches_cv21,
        wavefrom_info,
    )


def save_prob_for_phy(probability, param, clus_info):
    """
    Saves the within session UnitMatch probabilities for each session in their KiloSort directory,
    to be used with the UnitMatch Phy plugin

    Parameters
    ----------
    Probability : ndarray (nUnits, nUnits)
        The calculates UnitMatch probability array
    param : dictionary
        The param dictionary
    ClusInfo : dictionary
        The ClusInfo dictionary
    """

    session_switch = clus_info["session_switch"]
    n_units_per_session = param["n_units_per_session"]

    for sid in range(session_switch.shape[0] - 1):
        # file to save the array in
        save_file_tmp = os.path.join(param["KSdirs"][sid], "probability_templates.npy")

        matrix_prob = np.full(
            (n_units_per_session[sid], n_units_per_session[sid]), np.nan
        )  # Make the size of all the units

        session_output = probability[
            session_switch[sid] : session_switch[sid + 1],
            session_switch[sid] : session_switch[sid + 1],
        ]
        # If Only good units where used add values know to a matrix of NaNs
        if session_switch[sid + 1] - session_switch[sid] != n_units_per_session[sid]:
            all_good_units = clus_info["good_units"][sid].squeeze().astype(int)
            for id, gid in enumerate(clus_info["good_units"][sid].astype(int)):
                matrix_prob[gid, all_good_units] = session_output[id, :]
        else:
            matrix_prob = session_output

        np.save(save_file_tmp, matrix_prob)


def make_UnitMatch_folder_from_sorting_analyzers(
    analyzers, save_dir, overwrite=False
):
    """
    Creates a folder called `save_dir` for a single session,
    which can be used to run UnitMatch.
    
    The output folder has the structure

    unitmatch_dir/
        Session{i}/
            # (num_units, 3) dimensional array of channel locations
            channel_locations.npy 
            
            # (num_time_samples, num_channels, 2) array of average waveforms 
            # for each unit. Each file contains the average waveform for both
            # the first half and the second half of the recording.
            Unit0_RawSpikes.npy
            Unit1_RawSpikes.npy
            ...                
            UnitN_RawSpikes.npy

            # waveform specific params, from si waveform extension
            waveform_params.json
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    required_extensions = ["random_spikes", "waveforms"]

    for session_index, analyzer in enumerate(analyzers):

        missing_extensions = []
        for extension in required_extensions:
            if not analyzer.has_extension(extension):
                missing_extensions.append(extension)
        if len(missing_extensions) > 0:
            raise ValueError(
                f"Analyzer must have {missing_extensions} extensions computed.\n"
                f"Please compute them by running "
                f"`sorting_analyzer.compute({missing_extensions})`"
            )

        session_dir = save_dir / f'Session{session_index}'
        if session_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{session_dir} already exists. Pass overwrite=True to replace it."
                )
            shutil.rmtree(session_dir)
        session_dir.mkdir()

        # TEMPLATES / RAWSPIKES

        half_samples = analyzer.get_num_samples() // 2
        waveforms = analyzer.get_extension('waveforms')
        random_spikes = analyzer.get_extension('random_spikes')

        for unit_id in tqdm(analyzer.unit_ids, desc=f"Exporting Session{session_index}"):

            both_halves_average_waveform_dense = np.zeros((np.shape(waveforms.get_data())[1], analyzer.get_num_channels(), 2))

            unit_random_spikes = random_spikes.get_selected_indices_in_spike_train(unit_id = unit_id, segment_index=0)
            first_half_mask = analyzer.sorting.get_unit_spike_train(unit_id = unit_id)[unit_random_spikes] < half_samples

            first_half_waveforms = waveforms.get_waveforms_one_unit(unit_id = unit_id)[first_half_mask]
            second_half_waveforms = waveforms.get_waveforms_one_unit(unit_id = unit_id)[~first_half_mask]

            first_half_average_waveform_sparse = np.average(first_half_waveforms, axis=0)
            second_half_average_waveform_sparse = np.average(second_half_waveforms, axis=0)

            channel_indices = analyzer.sparsity.unit_id_to_channel_indices[unit_id]

            both_halves_average_waveform_dense[:, channel_indices, 0] = first_half_average_waveform_sparse[:, :channel_indices.size]
            both_halves_average_waveform_dense[:, channel_indices, 1] = second_half_average_waveform_sparse[:, :channel_indices.size]

            np.save(session_dir / f"Unit{unit_id}_RawSpikes.npy", both_halves_average_waveform_dense)

        # CHANNEL POSITION
        
        channel_locations = analyzer.get_channel_locations()
        if channel_locations.shape[1] == 2:
            channel_locations_3D = np.zeros((np.shape(channel_locations)[0], 3))
            channel_locations_3D[:,0:2] = channel_locations
        else:
            channel_locations_3D = channel_locations

        np.save(session_dir / "channel_locations.npy", channel_locations_3D)

        # WAVEFORM PARAMS

        ms_before = analyzer.get_extension('waveforms').params.get('ms_before')
        ms_after = analyzer.get_extension('waveforms').params.get('ms_after')

        spike_width = int( (ms_after + ms_before) * analyzer.sampling_frequency/1000 )
        peak_loc = int( spike_width * ms_before / (ms_before + ms_after) )

        waveidx = np.arange(peak_loc - 8, peak_loc + 15, dtype=int)
        # make waveidx json serializable
        waveidx = [int(idx) for idx in waveidx]

        params_waveform = {
            'spike_width': spike_width,
            'peak_loc': peak_loc,
            'waveidx': waveidx
        }

        with open(session_dir / "waveform_params.json", "w") as f:
            json.dump(params_waveform, f, indent=2)
