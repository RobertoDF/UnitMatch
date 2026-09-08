"""Compact, immutable archives for inspecting accepted UnitMatch pairs."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
_UUID_COLUMN = re.compile(r"(^uuid$|uuid[12]$|canonical_uuid)", re.IGNORECASE)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_SNAPSHOT_TABLE_FIELDS = frozenset(
    {"unit_identities", "units", "accepted_pairs", "decisions", "availability"}
)


def _validate_run_id(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("run_id must be a non-empty string")
    if (
        len(value) > 255
        or value in {".", ".."}
        or not _SAFE_COMPONENT.fullmatch(value)
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            "run_id must be a filesystem-safe single-component identifier"
        )
    return value


def _copy_frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value.copy(deep=True)


def _copy_arrays(
    values: Mapping[str, Mapping[int, np.ndarray]], name: str
) -> Mapping[str, Mapping[int, np.ndarray]]:
    copied = {}
    for category, entries in values.items():
        if not _SAFE_COMPONENT.fullmatch(str(category)):
            raise ValueError(f"{name} category {category!r} is not a safe file name")
        copied_entries = {}
        for index, value in entries.items():
            index = int(index)
            array = np.array(value, copy=True)
            if array.dtype.hasobject:
                raise TypeError(f"{name}[{category!r}][{index}] cannot use object dtype")
            array.setflags(write=False)
            copied_entries[index] = array
        copied[str(category)] = MappingProxyType(copied_entries)
    return MappingProxyType(copied)


def _copy_context(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    copied = {}
    for name, value in values.items():
        if not _SAFE_COMPONENT.fullmatch(str(name)):
            raise ValueError(f"plot context name {name!r} is not a safe file name")
        array = np.array(value, copy=True)
        if array.dtype.hasobject:
            raise TypeError(f"plot_context[{name!r}] cannot use object dtype")
        array.setflags(write=False)
        copied[str(name)] = array
    return MappingProxyType(copied)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("JSON metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON metadata value {type(value).__name__}")


def _without_identity_metadata(value):
    if isinstance(value, Mapping):
        return {
            str(key): _without_identity_metadata(item)
            for key, item in value.items()
            if "uuid" not in str(key).lower()
            and "registry" not in str(key).lower()
            and "identity" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_without_identity_metadata(item) for item in value]
    return value


def _copy_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze_json(_json_value(value))


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _frame_records(frame: pd.DataFrame, *, exclude_uuids: bool) -> list[dict]:
    columns = [
        column
        for column in frame.columns
        if not (exclude_uuids and _UUID_COLUMN.search(str(column)))
    ]
    normalized = frame.loc[:, columns].copy()
    for column in normalized.columns:
        if isinstance(normalized[column].dtype, pd.CategoricalDtype):
            normalized[column] = normalized[column].astype("string")
    return json.loads(normalized.to_json(orient="records", date_format="iso"))


def _update_array_hash(digest, arrays: Mapping[str, Mapping[int, np.ndarray]]) -> None:
    for category in sorted(arrays):
        for index in sorted(arrays[category]):
            array = np.asarray(arrays[category][index])
            digest.update(category.encode())
            digest.update(str(index).encode())
            digest.update(array.dtype.str.encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes(order="C"))


def _snapshot_digest(snapshot: AcceptedReviewSnapshot, *, decisions: bool) -> str:
    digest = sha256()
    if decisions:
        payload = {
            "decisions": _frame_records(snapshot.decisions, exclude_uuids=True),
            "settings": _json_value(snapshot.settings),
        }
    else:
        payload = {
            "units": _frame_records(snapshot.units, exclude_uuids=True),
            "accepted_pairs": _frame_records(
                snapshot.accepted_pairs, exclude_uuids=True
            ),
            "run_metadata": _without_identity_metadata(
                _json_value(snapshot.run_metadata)
            ),
            "availability": _frame_records(
                snapshot.availability, exclude_uuids=True
            ),
        }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    if not decisions:
        _update_array_hash(digest, snapshot.unit_arrays)
        _update_array_hash(digest, snapshot.pair_arrays)
        for name in sorted(snapshot.plot_context):
            array = np.asarray(snapshot.plot_context[name])
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _identity_digest(snapshot: AcceptedReviewSnapshot) -> str:
    digest = sha256()
    digest.update(
        json.dumps(
            _frame_records(snapshot.unit_identities, exclude_uuids=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def _snapshot_seal(snapshot: AcceptedReviewSnapshot) -> tuple[str, str, str]:
    return (
        _snapshot_digest(snapshot, decisions=False),
        _snapshot_digest(snapshot, decisions=True),
        _identity_digest(snapshot),
    )


@dataclass(frozen=True)
class AcceptedReviewSnapshot:
    """Immutable scientific and decision state for accepted-pair inspection."""

    run_id: str
    unit_identities: pd.DataFrame
    units: pd.DataFrame
    accepted_pairs: pd.DataFrame
    decisions: pd.DataFrame
    settings: Mapping[str, Any]
    run_metadata: Mapping[str, Any]
    unit_arrays: Mapping[str, Mapping[int, np.ndarray]] = field(default_factory=dict)
    pair_arrays: Mapping[str, Mapping[int, np.ndarray]] = field(default_factory=dict)
    plot_context: Mapping[str, np.ndarray] = field(default_factory=dict)
    availability: pd.DataFrame = field(default_factory=pd.DataFrame)
    schema_version: int = SCHEMA_VERSION
    _seal: tuple[str, str, str] = field(
        init=False, repr=False, compare=False, default=()
    )

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name in _SNAPSHOT_TABLE_FIELDS and isinstance(value, pd.DataFrame):
            return value.copy(deep=True)
        return value

    def __post_init__(self):
        _validate_run_id(self.run_id)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported snapshot schema {self.schema_version}")
        object.__setattr__(
            self, "unit_identities", _copy_frame(self.unit_identities, "unit_identities")
        )
        object.__setattr__(self, "units", _copy_frame(self.units, "units"))
        object.__setattr__(
            self, "accepted_pairs", _copy_frame(self.accepted_pairs, "accepted_pairs")
        )
        object.__setattr__(self, "decisions", _copy_frame(self.decisions, "decisions"))
        object.__setattr__(self, "settings", _copy_mapping(self.settings, "settings"))
        object.__setattr__(
            self, "run_metadata", _copy_mapping(self.run_metadata, "run_metadata")
        )
        object.__setattr__(
            self, "unit_arrays", _copy_arrays(self.unit_arrays, "unit_arrays")
        )
        object.__setattr__(
            self, "pair_arrays", _copy_arrays(self.pair_arrays, "pair_arrays")
        )
        object.__setattr__(self, "plot_context", _copy_context(self.plot_context))
        object.__setattr__(
            self, "availability", _copy_frame(self.availability, "availability")
        )
        _validate_snapshot(self)
        object.__setattr__(self, "_seal", _snapshot_seal(self))

    def _assert_untampered(self) -> None:
        if self._seal != _snapshot_seal(self):
            raise RuntimeError(
                "AcceptedReviewSnapshot was modified after construction"
            )

    def source_digest(self) -> str:
        """Hash scientific inputs while intentionally excluding UUID spelling."""
        self._assert_untampered()
        return self._seal[0]

    def decision_settings_digest(self) -> str:
        """Hash decisions and frozen settings while excluding UUID spelling."""
        self._assert_untampered()
        return self._seal[1]

    def identity_digest(self) -> str:
        self._assert_untampered()
        return self._seal[2]

    def with_unit_identities(
        self, aligned_table: pd.DataFrame
    ) -> AcceptedReviewSnapshot:
        """Return a remapped copy using stable session-name/unit-ID keys."""
        self._assert_untampered()
        identities = _validate_identity_table(aligned_table, self.unit_identities.columns)
        lookup = _identity_lookup(identities)

        units = self.units.copy(deep=True)
        units["UUID"] = _map_frame_uuids(units, lookup, "")

        pairs = self.accepted_pairs.copy(deep=True)
        pairs["UUID1"] = _map_frame_uuids(pairs, lookup, "1")
        pairs["UUID2"] = _map_frame_uuids(pairs, lookup, "2")

        decisions = self.decisions.copy(deep=True)
        decisions["UUID1"] = _map_frame_uuids(decisions, lookup, "1")
        decisions["UUID2"] = _map_frame_uuids(decisions, lookup, "2")

        remapped = replace(
            self,
            unit_identities=identities,
            units=units,
            accepted_pairs=pairs,
            decisions=decisions,
        )
        if remapped.source_digest() != self.source_digest():
            raise RuntimeError("Identity remapping changed the scientific snapshot")
        if remapped.decision_settings_digest() != self.decision_settings_digest():
            raise RuntimeError("Identity remapping changed decisions or settings")
        return remapped


def _validate_identity_table(
    identities: pd.DataFrame, expected_columns=None
) -> pd.DataFrame:
    identities = _copy_frame(identities, "unit_identities")
    if "UUID" not in identities:
        raise ValueError("unit_identities must contain a UUID column")
    if expected_columns is not None and list(identities.columns) != list(expected_columns):
        raise ValueError("Aligned identities must preserve the session columns and order")
    if identities["UUID"].isna().any() or identities["UUID"].duplicated().any():
        raise ValueError("Identity UUID values must be present and unique")
    for value in identities["UUID"]:
        try:
            UUID(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid identity UUID {value!r}") from exc
    for column in identities.columns[1:]:
        try:
            identities[column] = pd.array(identities[column], dtype="Int64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Identity session {column!r} must contain integers") from exc
        duplicate = identities[column].dropna().duplicated()
        if duplicate.any():
            value = identities.loc[identities[column].notna(), column][duplicate].iloc[0]
            raise ValueError(
                f"Identity session {column!r} maps unit {int(value)} more than once"
            )
    identities["UUID"] = identities["UUID"].astype("string")
    return identities


def _identity_lookup(identities: pd.DataFrame) -> dict[tuple[str, int], str]:
    lookup = {}
    for _, row in identities.iterrows():
        for session_name in identities.columns[1:]:
            if pd.notna(row[session_name]):
                lookup[(str(session_name), int(row[session_name]))] = str(row["UUID"])
    return lookup


def _map_frame_uuids(frame, lookup, suffix):
    session_column = f"session_name{suffix}"
    unit_column = f"session_unit_id{suffix}"
    if session_column not in frame or unit_column not in frame:
        raise ValueError(
            f"Snapshot table is missing {session_column!r}/{unit_column!r}"
        )
    values = []
    for session, unit in zip(frame[session_column], frame[unit_column]):
        key = (str(session), int(unit))
        if key not in lookup:
            raise ValueError(f"Aligned identities are missing stable unit key {key!r}")
        values.append(lookup[key])
    return pd.Series(values, index=frame.index, dtype="string")


def _validate_snapshot(snapshot: AcceptedReviewSnapshot) -> None:
    if snapshot.run_metadata.get("run_id") != snapshot.run_id:
        raise ValueError("Snapshot run_id does not match run_metadata['run_id']")
    identities = _validate_identity_table(snapshot.unit_identities)
    required_units = {
        "gui_row",
        "archive_unit_index",
        "session_index",
        "session_name",
        "session_unit_id",
        "UUID",
    }
    required_pairs = {
        "pair_id",
        "gui_row1",
        "gui_row2",
        "archive_unit_index1",
        "archive_unit_index2",
        "session_name1",
        "session_name2",
        "session_unit_id1",
        "session_unit_id2",
        "UUID1",
        "UUID2",
    }
    required_decisions = {
        "gui_row1",
        "gui_row2",
        "session_name1",
        "session_name2",
        "session_unit_id1",
        "session_unit_id2",
        "UUID1",
        "UUID2",
        "automatic_accept",
        "manual_accept",
        "manual_reject",
        "final_accept",
    }
    for name, frame, required in (
        ("units", snapshot.units, required_units),
        ("accepted_pairs", snapshot.accepted_pairs, required_pairs),
        ("decisions", snapshot.decisions, required_decisions),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    if snapshot.units["gui_row"].duplicated().any():
        raise ValueError("units.gui_row must be unique")
    accepted_units = snapshot.units["archive_unit_index"].dropna()
    if accepted_units.duplicated().any():
        raise ValueError("Non-null units.archive_unit_index values must be unique")
    if snapshot.accepted_pairs["pair_id"].duplicated().any():
        raise ValueError("accepted_pairs.pair_id must be unique")
    accepted_unit_indexes = {int(value) for value in accepted_units}
    pair_ids = {int(value) for value in snapshot.accepted_pairs["pair_id"]}
    for scope, arrays, valid_indexes in (
        ("unit_arrays", snapshot.unit_arrays, accepted_unit_indexes),
        ("pair_arrays", snapshot.pair_arrays, pair_ids),
    ):
        for category, entries in arrays.items():
            unknown = set(entries).difference(valid_indexes)
            if unknown:
                raise ValueError(
                    f"{scope}[{category!r}] references unknown indices "
                    f"{sorted(unknown)}"
                )
    lookup = _identity_lookup(identities)
    for frame, suffixes in (
        (snapshot.units, ("",)),
        (snapshot.accepted_pairs, ("1", "2")),
        (snapshot.decisions, ("1", "2")),
    ):
        for suffix in suffixes:
            expected = _map_frame_uuids(frame, lookup, suffix)
            actual = frame[f"UUID{suffix}"].astype("string")
            if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                raise ValueError("Snapshot UUID endpoints do not match unit_identities")
    _validate_endpoint_relationships(snapshot.units, snapshot.accepted_pairs)


def _validate_loaded_tables(tables) -> None:
    identities = _validate_identity_table(tables["unit_identities"])
    units = tables["units"]
    pairs = tables["accepted_pairs"]
    decisions = tables["decisions"]
    required_units = {
        "gui_row",
        "archive_unit_index",
        "session_index",
        "session_name",
        "session_unit_id",
        "UUID",
    }
    required_pairs = {
        "pair_id",
        "gui_row1",
        "gui_row2",
        "archive_unit_index1",
        "archive_unit_index2",
        "session_name1",
        "session_name2",
        "session_unit_id1",
        "session_unit_id2",
        "UUID1",
        "UUID2",
    }
    required_decisions = {
        "gui_row1",
        "gui_row2",
        "session_name1",
        "session_name2",
        "session_unit_id1",
        "session_unit_id2",
        "UUID1",
        "UUID2",
        "automatic_accept",
        "manual_accept",
        "manual_reject",
        "final_accept",
    }
    for name, frame, required in (
        ("units", units, required_units),
        ("accepted_pairs", pairs, required_pairs),
        ("decisions", decisions, required_decisions),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    if units["gui_row"].duplicated().any():
        raise ValueError("units.gui_row must be unique")
    if units["archive_unit_index"].dropna().duplicated().any():
        raise ValueError("units.archive_unit_index must be unique when present")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("accepted_pairs.pair_id must be unique")
    lookup = _identity_lookup(identities)
    for frame, suffixes in (
        (units, ("",)),
        (pairs, ("1", "2")),
        (decisions, ("1", "2")),
    ):
        for suffix in suffixes:
            expected = _map_frame_uuids(frame, lookup, suffix)
            actual = frame[f"UUID{suffix}"].astype("string")
            if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
                raise ValueError("Archived UUID endpoints do not match identities")

    _validate_endpoint_relationships(units, pairs)


def _validate_endpoint_relationships(units, pairs) -> None:
    by_gui_row = units.set_index("gui_row")
    for endpoint in (1, 2):
        for _, pair in pairs.iterrows():
            gui_row = pair[f"gui_row{endpoint}"]
            if gui_row not in by_gui_row.index:
                raise ValueError(f"Accepted pair references unknown GUI row {gui_row}")
            unit = by_gui_row.loc[gui_row]
            expected = (
                unit["archive_unit_index"],
                unit["session_name"],
                unit["session_unit_id"],
                unit["UUID"],
            )
            actual = (
                pair[f"archive_unit_index{endpoint}"],
                pair[f"session_name{endpoint}"],
                pair[f"session_unit_id{endpoint}"],
                pair[f"UUID{endpoint}"],
            )
            if any(
                (pd.isna(left) and not pd.isna(right))
                or (not pd.isna(left) and pd.isna(right))
                or (not pd.isna(left) and left != right)
                for left, right in zip(expected, actual)
            ):
                raise ValueError("Accepted pair endpoint metadata does not match units")


def _canonical_pairs(
    values, name: str, *, ignore_self_pairs: bool = False
) -> set[tuple[int, int]]:
    if values is None:
        return set()
    array = np.asarray(list(values) if isinstance(values, set) else values)
    if array.size == 0:
        return set()
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.issubdtype(array.dtype, np.integer) and (
        not np.issubdtype(array.dtype, np.number)
        or not np.all(np.equal(array, np.floor(array)))
    ):
        raise ValueError(f"{name} must contain integer row indices")
    pairs = set()
    for left, right in array.astype(np.int64, copy=False):
        if left < 0 or right < 0:
            raise ValueError(f"{name} contains an invalid pair ({left}, {right})")
        if left == right:
            if ignore_self_pairs:
                continue
            raise ValueError(f"{name} contains an invalid pair ({left}, {right})")
        pairs.add((int(min(left, right)), int(max(left, right))))
    return pairs


def _session_indices(clus_info, n_units):
    if "session_indices" in clus_info:
        values = np.asarray(clus_info["session_indices"]).reshape(-1)
    else:
        switch = np.asarray(clus_info["session_switch"]).reshape(-1)
        values = np.searchsorted(switch[1:], np.arange(n_units), side="right")
    if len(values) != n_units:
        raise ValueError("clus_info session mapping does not match original_ids")
    return values.astype(np.int64, copy=False)


def _session_names(gui, identities, n_sessions):
    names = list(identities.columns[1:])
    if len(names) != n_sessions:
        raise ValueError(
            "unit_identities session columns must match clus_info sessions"
        )
    return names


def _table_endpoint(unit_rows, left, right):
    first = unit_rows.loc[left]
    second = unit_rows.loc[right]
    return {
        "gui_row1": left,
        "gui_row2": right,
        "session_name1": first["session_name"],
        "session_name2": second["session_name"],
        "session_unit_id1": first["session_unit_id"],
        "session_unit_id2": second["session_unit_id"],
        "UUID1": first["UUID"],
        "UUID2": second["UUID"],
    }


def _matrix_value(matrix, left, right):
    if matrix is None:
        return np.nan
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("Pair score/probability inputs must be 2-D")
    return float(array[left, right])


def _unit_scalar(gui, source, row):
    values = getattr(gui, source, None)
    if values is None:
        return np.nan
    value = np.asarray(np.asarray(values)[row], dtype=float)
    return float(np.mean(value)) if value.size else np.nan


def _format_scalar_value(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def _minimum_positive_spacing(values, fallback):
    unique = np.unique(np.asarray(values, dtype=float))
    unique = unique[np.isfinite(unique)]
    differences = np.diff(np.sort(unique))
    differences = differences[differences > np.finfo(float).eps]
    return float(differences.min()) if differences.size else float(fallback)


def _json_settings(gui):
    before, after, bin_size = getattr(gui, "event_view_settings", (1.0, 2.0, 0.01))
    return {
        "match_threshold": float(gui.match_threshold),
        "review_threshold": float(getattr(gui, "review_threshold", 0.0)),
        "automatic_match_mode": str(
            getattr(gui, "automatic_match_mode", "or")
        ),
        "displacement_min_references": int(
            getattr(gui, "displacement_min_references", 5)
        ),
        "displacement_angle_threshold_degrees": float(
            getattr(gui, "DISPLACEMENT_ANGLE_THRESHOLD", 60.0)
        ),
        "consistency_filter_threshold": float(
            getattr(gui, "consistency_filter_threshold", 20.0)
        ),
        "acg": {
            "bin_size_s": 0.001,
            "max_lag_s": 0.05,
            "normalization": "count / (bin_size * n_spikes * recording_span)",
            "positive_lags_only": True,
        },
        "event_psth": {
            "before_s": float(before),
            "after_s": float(after),
            "bin_size_s": float(bin_size),
            "fixed": True,
        },
        "read_only": True,
        "supports_candidate_navigation": False,
        "supports_curation": False,
        "supports_event_window_editing": False,
    }


def _gui_capture_guard(gui):
    sources = (
        "clus_info",
        "output_avg",
        "output_GUI",
        "scores_to_include_avg",
        "scores_to_include_GUI",
        "raw_output",
        "waveform",
        "amplitude",
        "spatial_decay",
        "avg_waveform",
        "avg_waveform_per_tp",
        "avg_centroid",
        "raw_avg_centroid",
        "wave_idx",
        "max_site",
        "max_site_mean",
        "channel_pos",
        "within_session",
        "param",
        "event_data",
    )
    return {
        "source_ids": tuple(id(getattr(gui, name, None)) for name in sources),
        "automatic": tuple(
            sorted(
                _canonical_pairs(
                    getattr(gui, "automatic_match_pairs", set()),
                    "automatic_match_pairs",
                    ignore_self_pairs=True,
                )
            )
        ),
        "manual_accept": tuple(
            sorted(_canonical_pairs(getattr(gui, "is_match", set()), "is_match"))
        ),
        "manual_reject": tuple(
            sorted(
                _canonical_pairs(getattr(gui, "not_match", set()), "not_match")
            )
        ),
        "settings": json.dumps(_json_settings(gui), sort_keys=True),
    }


def capture_accepted_review(
    gui,
    *,
    unit_identities: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    unit_metadata: pd.DataFrame | Mapping[str, Any] | None = None,
) -> AcceptedReviewSnapshot:
    """Copy accepted-review state from the active displacement GUI module.

    The function reads scientific arrays and decisions but never initializes,
    resets, selects, or otherwise mutates GUI state.
    """
    identities = _validate_identity_table(unit_identities)
    initial_guard = _gui_capture_guard(gui)
    run_metadata = dict(run_metadata)
    run_id = run_metadata.get("run_id")
    if run_id is None:
        raise ValueError("run_metadata must contain an immutable run_id")
    _validate_run_id(run_id)

    clus_info = getattr(gui, "clus_info", None)
    if not isinstance(clus_info, Mapping):
        raise TypeError("GUI clus_info is unavailable")
    original_ids = np.asarray(clus_info["original_ids"]).reshape(-1)
    n_units = len(original_ids)
    session_indices = _session_indices(clus_info, n_units)
    n_sessions = int(session_indices.max()) + 1 if n_units else len(identities.columns) - 1
    session_names = _session_names(gui, identities, n_sessions)
    lookup = _identity_lookup(identities)

    automatic = _canonical_pairs(
        getattr(gui, "automatic_match_pairs", set()),
        "automatic_match_pairs",
        ignore_self_pairs=True,
    )
    manual_accept = _canonical_pairs(getattr(gui, "is_match", set()), "is_match")
    manual_reject = _canonical_pairs(getattr(gui, "not_match", set()), "not_match")
    final_pairs = (automatic | manual_accept) - manual_reject
    for name, pairs in (
        ("automatic_match_pairs", automatic),
        ("is_match", manual_accept),
        ("not_match", manual_reject),
    ):
        if any(right >= n_units for _, right in pairs):
            raise ValueError(f"{name} contains a row outside clus_info")

    metadata_by_row = {}
    if unit_metadata is not None:
        if isinstance(unit_metadata, pd.DataFrame):
            if "gui_row" not in unit_metadata:
                raise ValueError("unit_metadata DataFrame must contain gui_row")
            metadata_by_row = {
                int(row["gui_row"]): row.to_dict()
                for _, row in unit_metadata.iterrows()
            }
        elif isinstance(unit_metadata, Mapping):
            metadata_by_row = {int(key): dict(value) for key, value in unit_metadata.items()}
        else:
            raise TypeError("unit_metadata must be a DataFrame or row mapping")

    accepted_rows = sorted({row for pair in final_pairs for row in pair})
    archive_index = {row: index for index, row in enumerate(accepted_rows)}
    unit_records = []
    for row in range(n_units):
        session_index = int(session_indices[row])
        session_name = session_names[session_index]
        unit_id = int(original_ids[row])
        key = (session_name, unit_id)
        if key not in lookup:
            raise ValueError(f"unit_identities is missing GUI unit {key!r}")
        record = {
            "gui_row": row,
            "archive_unit_index": archive_index.get(row, pd.NA),
            "session_index": session_index,
            "session_name": session_name,
            "session_unit_id": unit_id,
            "UUID": lookup[key],
        }
        for source, target in (
            ("probe_numbers", "probe"),
            ("analyzer_unit_ids", "analyzer_unit_id"),
            ("shank_ids", "shank"),
        ):
            values = clus_info.get(source)
            if values is not None:
                record[target] = np.asarray(values).reshape(-1)[row]
        for key_name, value in metadata_by_row.get(row, {}).items():
            if key_name not in record and key_name != "gui_row":
                record[str(key_name)] = value
        unit_records.append(record)
    units = pd.DataFrame(unit_records)
    units["archive_unit_index"] = pd.array(
        units["archive_unit_index"], dtype="Int64"
    )
    units["session_index"] = pd.array(units["session_index"], dtype="Int64")
    units["session_unit_id"] = pd.array(units["session_unit_id"], dtype="Int64")
    units["UUID"] = units["UUID"].astype("string")
    unit_rows = units.set_index("gui_row", drop=False)

    output_avg = getattr(gui, "output_avg", None)
    output_gui = getattr(gui, "output_GUI", None)
    raw_output = getattr(gui, "raw_output", None)
    within_session = getattr(gui, "within_session", None)
    directional = list(output_gui) if output_gui is not None else [None, None]
    score_avg = getattr(gui, "scores_to_include_avg", {}) or {}
    score_gui = getattr(gui, "scores_to_include_GUI", None) or [{}, {}]
    pair_records = []
    for pair_id, (left, right) in enumerate(sorted(final_pairs)):
        record = {
            "pair_id": pair_id,
            **_table_endpoint(unit_rows, left, right),
            "archive_unit_index1": archive_index[left],
            "archive_unit_index2": archive_index[right],
            "probability_average": _matrix_value(output_avg, left, right),
            "probability_1_to_2": _matrix_value(
                directional[0] if len(directional) > 0 else None, left, right
            ),
            "probability_2_to_1": _matrix_value(
                directional[1] if len(directional) > 1 else None, left, right
            ),
            "probability_raw_1_to_2": _matrix_value(raw_output, left, right),
            "probability_raw_2_to_1": _matrix_value(raw_output, right, left),
            "amplitude1": _unit_scalar(gui, "amplitude", left),
            "amplitude2": _unit_scalar(gui, "amplitude", right),
            "spatial_decay1": _unit_scalar(gui, "spatial_decay", left),
            "spatial_decay2": _unit_scalar(gui, "spatial_decay", right),
            "within_session": bool(
                _matrix_value(within_session, left, right)
            )
            if within_session is not None
            else False,
        }
        for name, matrix in score_avg.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
            record[f"score_average_{safe_name}"] = _matrix_value(
                matrix, left, right
            )
            if len(score_gui) > 0 and name in score_gui[0]:
                record[f"score_1_to_2_{safe_name}"] = _matrix_value(
                    score_gui[0][name], left, right
                )
            if len(score_gui) > 1 and name in score_gui[1]:
                record[f"score_2_to_1_{safe_name}"] = _matrix_value(
                    score_gui[1][name], left, right
                )
        pair_records.append(record)
    accepted_pairs = pd.DataFrame(pair_records)
    if accepted_pairs.empty:
        accepted_pairs = pd.DataFrame(
            columns=[
                "pair_id",
                "gui_row1",
                "gui_row2",
                "archive_unit_index1",
                "archive_unit_index2",
                "session_name1",
                "session_name2",
                "session_unit_id1",
                "session_unit_id2",
                "UUID1",
                "UUID2",
                "probability_average",
                "probability_1_to_2",
                "probability_2_to_1",
            ]
        )

    decision_records = []
    for left, right in sorted(automatic | manual_accept | manual_reject):
        decision_records.append(
            {
                **_table_endpoint(unit_rows, left, right),
                "automatic_accept": (left, right) in automatic,
                "manual_accept": (left, right) in manual_accept,
                "manual_reject": (left, right) in manual_reject,
                "final_accept": (left, right) in final_pairs,
                "provenance": (
                    "manual_reject"
                    if (left, right) in manual_reject
                    else "manual_accept"
                    if (left, right) in manual_accept
                    else "automatic_accept"
                ),
            }
        )
    decisions = pd.DataFrame(decision_records)
    if decisions.empty:
        decisions = pd.DataFrame(
            columns=[
                "gui_row1",
                "gui_row2",
                "session_name1",
                "session_name2",
                "session_unit_id1",
                "session_unit_id2",
                "UUID1",
                "UUID2",
                "automatic_accept",
                "manual_accept",
                "manual_reject",
                "final_accept",
                "provenance",
            ]
        )

    (
        unit_arrays,
        pair_arrays,
        plot_context,
        availability,
        pair_metrics,
    ) = _capture_plot_payload(
        gui,
        accepted_rows,
        sorted(final_pairs),
        archive_index,
        unit_rows,
    )
    if pair_metrics:
        metric_frame = pd.DataFrame(pair_metrics)
        accepted_pairs = accepted_pairs.merge(
            metric_frame, on="pair_id", how="left", validate="one_to_one"
        )
    snapshot = AcceptedReviewSnapshot(
        run_id=run_id,
        unit_identities=identities,
        units=units,
        accepted_pairs=accepted_pairs,
        decisions=decisions,
        settings=_json_settings(gui),
        run_metadata=run_metadata,
        unit_arrays=unit_arrays,
        pair_arrays=pair_arrays,
        plot_context=plot_context,
        availability=availability,
    )
    if _gui_capture_guard(gui) != initial_guard:
        raise RuntimeError(
            "GUI scientific sources, decisions, or settings changed during capture"
        )
    return snapshot


def _capture_plot_payload(
    gui, accepted_rows, final_pairs, archive_index, unit_rows
):
    """Capture array-backed plot inputs without touching GUI controls."""
    unit_arrays = {}
    availability = []

    def availability_row(scope, artifact, available, reason="", **metadata):
        availability.append(
            {
                "scope": scope,
                "artifact": artifact,
                "available": bool(available),
                "reason": str(reason),
                **metadata,
            }
        )

    def capture_units(name, source, *, unit_axis=0, required=False):
        value = getattr(gui, source, None)
        if value is None:
            availability_row(
                "unit",
                name,
                False,
                f"GUI source {source} is unavailable",
            )
            if required and accepted_rows:
                raise ValueError(f"Required GUI plot source {source!r} is unavailable")
            return
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"GUI plot source {source!r} uses object dtype")
        if array.shape[unit_axis] <= max(accepted_rows, default=-1):
            raise ValueError(f"GUI plot source {source!r} does not cover accepted units")
        entries = {}
        for row in accepted_rows:
            entries[archive_index[row]] = np.take(array, row, axis=unit_axis)
        unit_arrays[name] = entries
        availability_row("unit", name, True)

    capture_units("waveform", "waveform", required=True)
    capture_units("avg_waveform", "avg_waveform", unit_axis=1, required=True)
    capture_units("avg_waveform_per_tp", "avg_waveform_per_tp", unit_axis=1)
    capture_units("avg_centroid", "avg_centroid", unit_axis=1, required=True)
    capture_units("raw_avg_centroid", "raw_avg_centroid", unit_axis=1)
    capture_units("wave_idx", "wave_idx")
    capture_units("max_site", "max_site")
    capture_units("max_site_mean", "max_site_mean")
    capture_units("amplitude", "amplitude")
    capture_units("spatial_decay", "spatial_decay")

    pair_arrays = {}
    context = {}
    channel_pos = getattr(gui, "channel_pos", None)
    if channel_pos is not None:
        for index, positions in enumerate(channel_pos):
            context[f"channel_positions_session_{index}"] = np.asarray(positions)
    wave_idx = getattr(gui, "wave_idx", None)
    if wave_idx is not None:
        context["wave_idx"] = np.asarray(wave_idx)

    histogram_helper = getattr(gui, "get_score_histograms", None)
    histogram_sources = (
        (
            "average",
            getattr(gui, "scores_to_include_avg", {}) or {},
            getattr(gui, "output_avg", None),
        ),
        (
            "cv",
            (
                (getattr(gui, "scores_to_include_GUI", None) or [{}])[0]
            ),
            (
                (getattr(gui, "output_GUI", None) or [None])[0]
            ),
        ),
    )
    if callable(histogram_helper):
        for view, scores, probability in histogram_sources:
            if probability is None:
                continue
            names, histograms, matched_histograms = histogram_helper(
                scores,
                np.asarray(probability) > float(gui.match_threshold),
            )
            for name, histogram, matched in zip(
                names, histograms, matched_histograms
            ):
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
                context[f"histogram_{view}_{safe_name}_density"] = histogram[0]
                context[f"histogram_{view}_{safe_name}_edges"] = histogram[1]
                context[f"histogram_{view}_{safe_name}_matched_density"] = matched[0]
                context[f"histogram_{view}_{safe_name}_matched_edges"] = matched[1]
        availability_row("global", "score_histograms", True)
    else:
        availability_row(
            "global",
            "score_histograms",
            False,
            "Active GUI histogram helper is unavailable",
        )

    _capture_acgs(gui, accepted_rows, archive_index, unit_arrays, availability_row)
    _capture_event_psths(
        gui, accepted_rows, archive_index, unit_arrays, availability_row
    )
    displacement_metrics = _capture_displacement(
        gui,
        final_pairs,
        pair_arrays,
        context,
        availability_row,
    )
    event_metrics = _capture_event_correlations(
        final_pairs,
        archive_index,
        unit_arrays,
        availability,
        availability_row,
    )
    metrics_by_pair = {}
    for metric in displacement_metrics + event_metrics:
        metrics_by_pair.setdefault(metric["pair_id"], {}).update(metric)
    pair_metrics = list(metrics_by_pair.values())

    return (
        unit_arrays,
        pair_arrays,
        context,
        pd.DataFrame(availability),
        pair_metrics,
    )


def _accepted_unit_spike_times(gui, row):
    event_data = getattr(gui, "event_data", None)
    if isinstance(event_data, Mapping) and callable(event_data.get("get_spike_times")):
        return np.asarray(event_data["get_spike_times"](row), dtype=float).reshape(-1)
    loader = getattr(gui, "get_spike_times_for_unit_precalc", None)
    if not callable(loader):
        return None
    return np.asarray(loader(row, gui.clus_info, gui.param), dtype=float).reshape(-1)


def _capture_acgs(gui, accepted_rows, archive_index, unit_arrays, availability_row):
    compute = getattr(gui, "compute_acg_precalc", None)
    if not callable(compute):
        availability_row(
            "unit",
            "acg",
            False,
            "Active GUI deterministic ACG helper is unavailable",
        )
        return
    rates = {}
    axes = {}
    for row in accepted_rows:
        try:
            spike_times = _accepted_unit_spike_times(gui, row)
            if spike_times is None:
                availability_row(
                    "unit",
                    "acg",
                    False,
                    "No spike-time source is configured",
                    gui_row=row,
                )
                continue
            if len(spike_times) < 2:
                availability_row(
                    "unit",
                    "acg",
                    False,
                    "Fewer than two spikes are available",
                    gui_row=row,
                )
                continue
            rate, axis = compute(spike_times, bin_size=0.001, max_lag=0.05)
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Failed to materialize ACG for GUI row {row}") from exc
        mask = np.asarray(axis) >= 0
        rates[archive_index[row]] = np.asarray(rate)[mask]
        axes[archive_index[row]] = np.asarray(axis)[mask] * 1000
        availability_row("unit", "acg", True, gui_row=row)
    if rates:
        unit_arrays["acg_rate_hz"] = rates
        unit_arrays["acg_axis_ms"] = axes


def _capture_event_psths(
    gui, accepted_rows, archive_index, unit_arrays, availability_row
):
    event_data = getattr(gui, "event_data", None)
    if event_data is None:
        availability_row(
            "unit",
            "event_psth",
            False,
            "No event data was supplied to the active GUI",
        )
        return
    compute = getattr(gui, "_compute_event_psth", None)
    if not isinstance(event_data, Mapping) or not callable(compute):
        raise TypeError("Active GUI event data/helper is malformed")
    get_spikes = event_data.get("get_spike_times")
    events_by_session = event_data.get("event_times_by_session")
    if not callable(get_spikes) or events_by_session is None:
        raise ValueError("Active GUI event data is missing required sources")
    before, after, bin_size = getattr(
        gui, "event_view_settings", (1.0, 2.0, 0.01)
    )
    sessions = _session_indices(gui.clus_info, len(gui.clus_info["original_ids"]))
    rates = {}
    axes = {}
    counts = {}
    for row in accepted_rows:
        try:
            spikes = np.asarray(get_spikes(row), dtype=float).reshape(-1)
            events = events_by_session[int(sessions[row])]
            names = sorted(events)
            unit_rates = []
            unit_counts = []
            axis = None
            for event_name in names:
                axis, rate, event_count = compute(
                    spikes,
                    events[event_name],
                    before_s=before,
                    after_s=after,
                    bin_size_s=bin_size,
                )
                unit_rates.append(np.asarray(rate))
                unit_counts.append(event_count)
                availability_row(
                    "unit",
                    "event_psth",
                    True,
                    gui_row=row,
                    event_index=len(unit_rates) - 1,
                    event_name=str(event_name),
                )
        except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to materialize fixed event PSTHs for GUI row {row}"
            ) from exc
        index = archive_index[row]
        if unit_rates:
            rates[index] = np.stack(unit_rates)
            axes[index] = np.asarray(axis)
            counts[index] = np.asarray(unit_counts, dtype=np.int64)
        else:
            availability_row(
                "unit",
                "event_psth",
                False,
                "This unit's session has no finite events",
                gui_row=row,
            )
    if rates:
        unit_arrays["event_psth_rate_hz"] = rates
        unit_arrays["event_psth_axis_s"] = axes
        unit_arrays["event_psth_trial_count"] = counts


def _capture_event_correlations(
    final_pairs,
    archive_index,
    unit_arrays,
    availability,
    availability_row,
):
    rates = unit_arrays.get("event_psth_rate_hz", {})
    counts = unit_arrays.get("event_psth_trial_count", {})
    names_by_row = {}
    for item in availability:
        if (
            item.get("artifact") == "event_psth"
            and item.get("available")
            and "gui_row" in item
            and "event_index" in item
            and "event_name" in item
        ):
            names_by_row.setdefault(int(item["gui_row"]), {})[
                str(item["event_name"])
            ] = int(item["event_index"])

    metrics = []
    for pair_id, (left, right) in enumerate(final_pairs):
        left_index = archive_index[left]
        right_index = archive_index[right]
        correlations = []
        reason = ""
        if (
            left_index not in rates
            or right_index not in rates
            or left_index not in counts
            or right_index not in counts
        ):
            reason = "Fixed event PSTHs are unavailable for one or both units"
        else:
            shared = sorted(
                set(names_by_row.get(left, {}))
                & set(names_by_row.get(right, {}))
            )
            for name in shared:
                left_event = names_by_row[left][name]
                right_event = names_by_row[right][name]
                if min(
                    counts[left_index][left_event],
                    counts[right_index][right_event],
                ) < 2:
                    continue
                profiles = (
                    np.asarray(rates[left_index][left_event], dtype=float),
                    np.asarray(rates[right_index][right_event], dtype=float),
                )
                if any(
                    not np.all(np.isfinite(profile))
                    or np.ptp(profile)
                    <= np.finfo(float).eps
                    * max(1, np.max(np.abs(profile)))
                    * 32
                    for profile in profiles
                ):
                    continue
                correlation = float(np.corrcoef(*profiles)[0, 1])
                if np.isfinite(correlation):
                    correlations.append(correlation)
            if not correlations:
                reason = (
                    "No shared event has at least two trials and finite "
                    "nonconstant PSTHs"
                )
        event_r = float(np.mean(correlations)) if correlations else np.nan
        metrics.append(
            {
                "pair_id": pair_id,
                "event_r": event_r,
                "event_r_reason": reason,
            }
        )
        availability_row(
            "pair",
            "event_correlation",
            bool(correlations),
            reason,
            pair_id=pair_id,
        )
    return metrics


def _capture_displacement(
    gui, final_pairs, pair_arrays, context, availability_row
):
    raw_centroids = getattr(gui, "raw_avg_centroid", None)
    clus_info = gui.clus_info
    accepted_graph = np.asarray(final_pairs, dtype=np.int64).reshape(-1, 2)
    context["accepted_reference_graph_gui_rows"] = accepted_graph
    if raw_centroids is None:
        availability_row(
            "pair",
            "displacement",
            False,
            "Raw pre-drift centroids are unavailable",
        )
        return []
    if "probe_numbers" not in clus_info:
        availability_row(
            "pair",
            "displacement",
            False,
            "Probe metadata is unavailable",
        )
        return []
    try:
        from UnitMatchPy.displacement_consistency import DisplacementConsistency
    except ImportError as exc:
        raise ImportError(
            "Displacement snapshot capture requires the active "
            "UnitMatchPy.displacement_consistency module"
        ) from exc
    minimum = int(getattr(gui, "displacement_min_references", 5))
    model = DisplacementConsistency(
        raw_centroids,
        clus_info["session_id"],
        clus_info["probe_numbers"],
        final_pairs,
        shank_ids=clus_info.get("shank_ids"),
        min_references=minimum,
    )
    approved_helper = getattr(gui, "_approved_displacement", None)
    selected_helper = getattr(gui, "_selected_displacement", None)
    summaries = {}
    vectors = {}
    metrics = []
    for pair_id, (left, right) in enumerate(final_pairs):
        result = model.score(left, right)
        summaries[pair_id] = np.asarray(
            [
                np.nan if result.score is None else result.score,
                np.nan if result.distance is None else result.distance,
                np.nan if result.residual_um is None else result.residual_um,
                result.reference_count,
                minimum,
            ],
            dtype=np.float64,
        )
        approved = (
            approved_helper(left, right, final_pairs, raw_centroids, clus_info)
            if callable(approved_helper)
            else {"available": False, "message": "Approved helper unavailable"}
        )
        selected = (
            selected_helper(left, right, final_pairs, raw_centroids, clus_info)
            if callable(selected_helper)
            else {"available": False, "message": "Selected helper unavailable"}
        )
        vectors[pair_id] = np.asarray(
            [
                approved.get("dx", np.nan),
                approved.get("dy", np.nan),
                selected.get("dx", np.nan),
                selected.get("dy", np.nan),
            ],
            dtype=np.float64,
        )
        metrics.append(
            {
                "pair_id": pair_id,
                "displacement_consistency_score": result.score,
                "displacement_distance": result.distance,
                "displacement_residual_um": result.residual_um,
                "displacement_reference_count": result.reference_count,
                "displacement_min_references": minimum,
                "displacement_reason": result.reason,
                "approved_displacement_count": approved.get("count", pd.NA),
                "approved_displacement_reason": approved.get("message", ""),
                "selected_displacement_approved": selected.get("approved", False),
                "selected_displacement_reason": selected.get("message", ""),
            }
        )
        availability_row(
            "pair",
            "displacement",
            result.score is not None,
            result.reason,
            pair_id=pair_id,
        )
    pair_arrays["displacement_summary"] = summaries
    pair_arrays["displacement_vectors_xy"] = vectors
    return metrics


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require_parquet():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Compact review archives require PyArrow. Install UnitMatchPy[archive]."
        ) from exc


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, **{name: np.asarray(array) for name, array in arrays.items()}
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(path: Path, root: Path, kind: str, *, lazy=False):
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "lazy": bool(lazy),
    }


def _array_artifact(path, root, arrays, **metadata):
    item = _artifact(path, root, "npz", lazy=True)
    item.update(entries={
        name: {
            "dtype": np.asarray(array).dtype.str,
            "shape": list(np.asarray(array).shape),
        }
        for name, array in arrays.items()
    }, **metadata)
    return item


def _table_schema(frame):
    return [
        {"name": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]


def save_review_output(save_dir, snapshot: AcceptedReviewSnapshot) -> Path:
    """Write a compact snapshot into an empty caller-owned staging directory."""
    if not isinstance(snapshot, AcceptedReviewSnapshot):
        raise TypeError("review_snapshot must be an AcceptedReviewSnapshot")
    snapshot._assert_untampered()
    _validate_snapshot(snapshot)
    _require_parquet()
    root = Path(save_dir)
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"Archive path is not a directory: {root}")
        if any(root.iterdir()):
            raise FileExistsError(f"Archive staging directory is not empty: {root}")
    else:
        root.mkdir(parents=True)

    artifacts = []
    table_schemas = {}
    tables = {
        "unit_identities": snapshot.unit_identities,
        "units": snapshot.units,
        "accepted_pairs": snapshot.accepted_pairs,
        "decisions": snapshot.decisions,
        "availability": snapshot.availability,
    }
    for name, frame in tables.items():
        path = root / f"{name}.parquet"
        frame.to_parquet(path, engine="pyarrow", index=False)
        artifacts.append(_artifact(path, root, "table"))
        table_schemas[name] = _table_schema(frame)

    for scope, arrays in (
        ("unit_arrays", snapshot.unit_arrays),
        ("pair_arrays", snapshot.pair_arrays),
    ):
        indexes = sorted(
            {
                int(index)
                for entries in arrays.values()
                for index in entries
            }
        )
        for index in indexes:
            bundle = {
                category: entries[index]
                for category, entries in arrays.items()
                if index in entries
            }
            path = root / scope / f"{index}.npz"
            _write_npz(path, bundle)
            artifacts.append(
                _array_artifact(
                    path,
                    root,
                    bundle,
                    scope=scope,
                    index=index,
                )
            )
    if snapshot.plot_context:
        path = root / "plot_context.npz"
        _write_npz(path, snapshot.plot_context)
        item = _array_artifact(
            path,
            root,
            snapshot.plot_context,
            scope="plot_context",
        )
        artifacts.append(item)

    manifest = {
        "schema": "unitmatch.accepted-review",
        "schema_version": snapshot.schema_version,
        "status": "complete",
        "run_id": snapshot.run_id,
        "source_digest": snapshot.source_digest(),
        "decision_settings_digest": snapshot.decision_settings_digest(),
        "identity_digest": snapshot.identity_digest(),
        "settings": _json_value(snapshot.settings),
        "run_metadata": _json_value(snapshot.run_metadata),
        "table_schemas": table_schemas,
        "numeric_inventory": _numeric_inventory(snapshot),
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    _atomic_json(root / "run.json", manifest)
    return root


def _safe_artifact(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError("Manifest artifact paths must be strings")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe archive artifact path {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Archive artifact escapes run directory: {relative!r}") from exc
    return resolved


def _numeric_inventory(snapshot: AcceptedReviewSnapshot):
    items = [
        {"scope": scope, "category": category, "index": int(index)}
        for scope, arrays in (
            ("unit_arrays", snapshot.unit_arrays),
            ("pair_arrays", snapshot.pair_arrays),
        )
        for category, entries in arrays.items()
        for index in entries
    ]
    items.extend(
        {"scope": "plot_context", "category": category, "index": None}
        for category in snapshot.plot_context
    )
    return sorted(
        items,
        key=lambda item: (
            item["scope"],
            item["category"],
            -1 if item["index"] is None else item["index"],
        ),
    )


def _validate_numeric_manifest(root, manifest, artifacts, tables) -> None:
    inventory = manifest.get("numeric_inventory")
    if not isinstance(inventory, list):
        raise TypeError("Review archive manifest has no numeric inventory")
    expected = set()
    allowed_unit_indexes = {
        int(value)
        for value in tables["units"]["archive_unit_index"].dropna()
    }
    allowed_pair_indexes = {
        int(value) for value in tables["accepted_pairs"]["pair_id"]
    }
    for item in inventory:
        if (
            not isinstance(item, dict)
            or set(item) != {"scope", "category", "index"}
            or item["scope"] not in {"unit_arrays", "pair_arrays", "plot_context"}
            or not isinstance(item["category"], str)
        ):
            raise ValueError("Malformed numeric inventory entry")
        scope = item["scope"]
        index = item["index"]
        if scope == "plot_context":
            if index is not None:
                raise ValueError("Plot-context inventory entries cannot have an index")
        elif isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Indexed numeric inventory entries require integer indices")
        elif (
            scope == "unit_arrays" and index not in allowed_unit_indexes
        ) or (
            scope == "pair_arrays" and index not in allowed_pair_indexes
        ):
            raise ValueError(f"Numeric inventory references unknown {scope} index {index}")
        key = (scope, item["category"], index)
        if key in expected:
            raise ValueError(f"Duplicate numeric inventory entry {key!r}")
        expected.add(key)

    actual = set()
    for item in artifacts:
        if item["kind"] != "npz":
            continue
        scope = item.get("scope")
        if scope not in {"unit_arrays", "pair_arrays", "plot_context"}:
            raise ValueError(f"Invalid numeric artifact scope {scope!r}")
        index = item.get("index")
        if scope == "plot_context":
            if index is not None:
                raise ValueError("Plot-context artifacts cannot have an index")
        elif isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Indexed numeric artifacts require integer indices")
        entries = item.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Numeric bundle has no entries: {item['path']}")
        path = _safe_artifact(root, item["path"])
        try:
            with np.load(path, allow_pickle=False) as bundle:
                bundle_entries = set(bundle.files)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid numeric bundle: {item['path']}") from exc
        if bundle_entries != set(entries):
            raise ValueError(
                f"Numeric bundle entries do not match manifest: {item['path']}"
            )
        for category in entries:
            actual.add((scope, category, index))
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(
            f"Numeric artifact inventory mismatch; missing={missing}, extra={extra}"
        )


class LazyArrayStore:
    """Validated on-demand access to compressed numeric bundles."""

    def __init__(self, root: Path, entries):
        self._root = root
        self._entries = {}
        for item in entries:
            if item["kind"] != "npz":
                continue
            entry_schemas = item.get("entries")
            if not isinstance(entry_schemas, dict) or not entry_schemas:
                raise ValueError(f"Numeric bundle has no entries: {item['path']}")
            for category in entry_schemas:
                key = (
                    item.get("scope"),
                    category,
                    item.get("index"),
                )
                if key in self._entries:
                    raise ValueError(f"Duplicate logical numeric artifact {key!r}")
                self._entries[key] = (item, entry_schemas[category])
        self.loaded_paths = set()
        self._verified_paths = set()

    def categories(self, scope: str):
        return sorted(
            {
                category
                for entry_scope, category, _ in self._entries
                if entry_scope == scope
            }
        )

    def has(self, scope: str, category: str, index: int | None = None):
        return (
            scope,
            category,
            None if index is None else int(index),
        ) in self._entries

    def load(self, scope: str, category: str, index: int | None = None):
        key = (scope, category, None if index is None else int(index))
        if key not in self._entries:
            raise KeyError(key)
        item, schema = self._entries[key]
        path = _safe_artifact(self._root, item["path"])
        if item["path"] not in self._verified_paths:
            _validate_file(path, item)
            self._verified_paths.add(item["path"])
        with np.load(path, allow_pickle=False) as bundle:
            if category not in bundle.files:
                raise ValueError(
                    f"Archive bundle is missing {category!r}: {item['path']}"
                )
            array = np.array(bundle[category], copy=True)
        if array.dtype.str != schema.get("dtype") or list(array.shape) != schema.get(
            "shape"
        ):
            raise ValueError(f"Archive array schema mismatch: {item['path']}")
        self.loaded_paths.add(item["path"])
        return array

    def verify_all(self):
        checked = set()
        for item, _ in self._entries.values():
            if item["path"] in checked:
                continue
            _validate_file(_safe_artifact(self._root, item["path"]), item)
            checked.add(item["path"])
            self._verified_paths.add(item["path"])


@dataclass(frozen=True)
class LoadedReviewOutput:
    path: Path
    manifest: Mapping[str, Any]
    unit_identities: pd.DataFrame
    units: pd.DataFrame
    accepted_pairs: pd.DataFrame
    decisions: pd.DataFrame
    availability: pd.DataFrame
    arrays: LazyArrayStore

    def load_unit_array(self, name: str, archive_unit_index: int):
        return self.arrays.load("unit_arrays", name, archive_unit_index)

    def load_pair_array(self, name: str, pair_id: int):
        return self.arrays.load("pair_arrays", name, pair_id)

    def load_plot_context(self, name: str):
        return self.arrays.load("plot_context", name)

    def verify_all(self):
        self.arrays.verify_all()

    def with_resolved_aliases(self, resolver: Callable[[str], str]):
        units = self.units.copy()
        pairs = self.accepted_pairs.copy()
        decisions = self.decisions.copy()
        units["CanonicalUUID"] = units["UUID"].map(resolver).astype("string")
        for frame in (pairs, decisions):
            frame["CanonicalUUID1"] = frame["UUID1"].map(resolver).astype("string")
            frame["CanonicalUUID2"] = frame["UUID2"].map(resolver).astype("string")
        return replace(self, units=units, accepted_pairs=pairs, decisions=decisions)


def _validate_file(path: Path, item) -> None:
    if not path.is_file():
        raise ValueError(f"Archive artifact is missing: {item['path']}")
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Archive artifact size mismatch: {item['path']}")
    if _sha256_file(path) != item["sha256"]:
        raise ValueError(f"Archive artifact checksum mismatch: {item['path']}")


def _validate_file_presence(path: Path, item) -> None:
    if not path.is_file():
        raise ValueError(f"Archive artifact is missing: {item['path']}")
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Archive artifact size mismatch: {item['path']}")


def load_review_output(
    run_path, *, resolve_aliases: Callable[[str], str] | None = None
) -> LoadedReviewOutput:
    """Load and validate compact review metadata; numeric arrays remain lazy."""
    _require_parquet()
    root = Path(run_path)
    manifest_path = root / "run.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete review archive (missing run.json): {root}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema") != "unitmatch.accepted-review"
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Unsupported or incomplete review archive manifest")
    _validate_run_id(manifest.get("run_id"))
    if manifest.get("run_metadata", {}).get("run_id") != manifest["run_id"]:
        raise ValueError("Manifest run_id does not match captured run metadata")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("Review archive manifest has no artifact list")
    paths = set()
    for item in artifacts:
        required = {"path", "kind", "size", "sha256"}
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("Malformed artifact entry in review archive manifest")
        if item["kind"] not in {"table", "npz"}:
            raise ValueError(f"Unsupported archive artifact kind {item['kind']!r}")
        _safe_artifact(root, item["path"])
        if item["path"] in paths:
            raise ValueError(f"Duplicate artifact path {item['path']!r}")
        paths.add(item["path"])
        _validate_file_presence(_safe_artifact(root, item["path"]), item)

    table_entries = {}
    for item in artifacts:
        if item["kind"] != "table":
            continue
        name = Path(item["path"]).stem
        if name in table_entries:
            raise ValueError(f"Duplicate logical table artifact {name!r}")
        table_entries[name] = item
    required_tables = {
        "unit_identities",
        "units",
        "accepted_pairs",
        "decisions",
        "availability",
    }
    if not required_tables.issubset(table_entries):
        raise ValueError("Review archive is missing required typed tables")
    tables = {}
    for name in required_tables:
        item = table_entries[name]
        path = _safe_artifact(root, item["path"])
        _validate_file(path, item)
        tables[name] = pd.read_parquet(path, engine="pyarrow")
        actual_schema = _table_schema(tables[name])
        expected_schema = manifest.get("table_schemas", {}).get(name)
        if expected_schema != actual_schema:
            raise ValueError(f"Archive table schema mismatch: {item['path']}")
    _validate_loaded_tables(tables)
    _validate_numeric_manifest(root, manifest, artifacts, tables)

    loaded = LoadedReviewOutput(
        path=root,
        manifest=MappingProxyType(manifest),
        unit_identities=tables["unit_identities"],
        units=tables["units"],
        accepted_pairs=tables["accepted_pairs"],
        decisions=tables["decisions"],
        availability=tables["availability"],
        arrays=LazyArrayStore(root, artifacts),
    )
    if resolve_aliases is not None:
        loaded = loaded.with_resolved_aliases(resolve_aliases)
    return loaded


class SavedReviewViewer:
    """Read-only access to archived accepted-pair plot inputs."""

    supports_candidate_navigation = False
    supports_curation = False
    supports_event_window_editing = False

    def __init__(self, output: LoadedReviewOutput):
        self.output = output

    @property
    def pairs(self):
        return self.output.accepted_pairs.copy()

    def pair_payload(self, pair_id: int):
        rows = self.output.accepted_pairs
        selected = rows.loc[rows["pair_id"] == pair_id]
        if len(selected) != 1:
            raise KeyError(f"Unknown accepted pair {pair_id}")
        row = selected.iloc[0].to_dict()
        unit_payload = {}
        for endpoint in (1, 2):
            index = int(row[f"archive_unit_index{endpoint}"])
            unit_payload[endpoint] = {}
            for name in self.output.arrays.categories("unit_arrays"):
                if self.output.arrays.has("unit_arrays", name, index):
                    unit_payload[endpoint][name] = self.output.load_unit_array(
                        name, index
                    )
        pair_payload = {}
        for name in self.output.arrays.categories("pair_arrays"):
            try:
                pair_payload[name] = self.output.load_pair_array(name, pair_id)
            except KeyError:
                continue
        plot_context = {
            name: self.output.load_plot_context(name)
            for name in self.output.arrays.categories("plot_context")
        }
        availability = self.output.availability.copy()
        if not availability.empty:
            gui_scoped = (
                availability["gui_row"].notna()
                if "gui_row" in availability
                else pd.Series(False, index=availability.index)
            )
            pair_scoped = (
                availability["pair_id"].notna()
                if "pair_id" in availability
                else pd.Series(False, index=availability.index)
            )
            global_scope = (
                availability["scope"].eq("global")
                if "scope" in availability
                else pd.Series(False, index=availability.index)
            )
            selected_gui_rows = {int(row["gui_row1"]), int(row["gui_row2"])}
            selected_gui = (
                availability["gui_row"].isin(selected_gui_rows)
                if "gui_row" in availability
                else pd.Series(False, index=availability.index)
            )
            selected_pair = (
                availability["pair_id"].eq(pair_id)
                if "pair_id" in availability
                else pd.Series(False, index=availability.index)
            )
            availability = availability.loc[
                global_scope
                | ~(gui_scoped | pair_scoped)
                | selected_gui
                | selected_pair
            ].copy()
        return {
            "pair": row,
            "units": unit_payload,
            "pair_arrays": pair_payload,
            "plot_context": plot_context,
            "settings": self.output.manifest["settings"],
            "availability": availability,
        }

    def spatial_waveforms(self, pair_id: int, *, cv="Avg", swapped=False):
        """Return the active GUI's anchor-neighborhood waveform payload."""
        payload = self.pair_payload(pair_id)
        first, second = ((2, 1) if swapped else (1, 2))
        pair = payload["pair"]
        anchor = payload["units"][first]
        partner = payload["units"][second]
        required = {"waveform", "max_site", "max_site_mean"}
        if not required.issubset(anchor) or "waveform" not in partner:
            raise ValueError("Archived spatial-waveform inputs are unavailable")
        session_index = int(pair[f"session_name{first}"] == pair["session_name2"])
        unit_row = self.output.units.loc[
            self.output.units["archive_unit_index"]
            == int(pair[f"archive_unit_index{first}"])
        ].iloc[0]
        session_index = int(unit_row["session_index"])
        positions = self.output.load_plot_context(
            f"channel_positions_session_{session_index}"
        )
        max_site = (
            int(np.asarray(anchor["max_site_mean"]).squeeze())
            if cv == "Avg"
            else int(np.asarray(anchor["max_site"]).reshape(-1)[int(cv)])
        )
        x = positions[max_site, 1]
        candidates = np.flatnonzero(
            (positions[:, 1] > x - 50) & (positions[:, 1] < x + 50)
        )
        channels = candidates[
            np.argsort(np.abs(positions[candidates, 2] - positions[max_site, 2]))[
                :18
            ]
        ]
        channels = channels[np.argsort(-positions[channels, 2], kind="stable")]
        for start in range(0, len(channels) - 1, 2):
            if positions[channels[start], 1] > positions[channels[start + 1], 1]:
                channels[[start, start + 1]] = channels[[start + 1, start]]
        anchor_waveform = np.asarray(anchor["waveform"])
        partner_waveform = np.asarray(partner["waveform"])
        if cv == "Avg":
            anchor_values = anchor_waveform[:, channels].mean(axis=-1)
            partner_values = partner_waveform[:, channels].mean(axis=-1)
        else:
            cv = int(cv)
            if cv not in (0, 1):
                raise ValueError("cv must be 'Avg', 0, or 1")
            anchor_values = anchor_waveform[:, channels, cv]
            partner_values = partner_waveform[:, channels, 1 - cv]
        return {
            "channels": channels,
            "positions_xy": positions[channels][:, [1, 2]],
            "anchor_waveforms": anchor_values,
            "partner_waveforms": partner_values,
            "anchor_endpoint": first,
            "partner_endpoint": second,
            "cv": cv,
        }

    def render_pair(self, pair_id: int, *, cv="Avg", swapped=False):
        """Render all supported archived plots without curation controls."""
        from matplotlib.figure import Figure

        payload = self.pair_payload(pair_id)
        first, second = ((2, 1) if swapped else (1, 2))
        unit_a = payload["units"][first]
        unit_b = payload["units"][second]
        figure = Figure(figsize=(15, 11), layout="constrained")
        axes = figure.subplots(3, 3)
        (
            waveform_axis,
            trajectory_axis,
            acg_axis,
            event_axis,
            spatial_axis,
            histogram_axis,
            displacement_axis,
            score_axis,
            availability_axis,
        ) = axes.reshape(-1)

        for endpoint, unit, label in (
            (first, unit_a, "Unit A"),
            (second, unit_b, "Unit B"),
        ):
            if "avg_waveform" in unit:
                values = np.asarray(unit["avg_waveform"])
                waveform = (
                    values.mean(axis=-1)
                    if cv == "Avg"
                    else values[:, int(cv if label == "Unit A" else 1 - int(cv))]
                )
                waveform_axis.plot(waveform, label=label)
            if {
                "avg_waveform_per_tp",
                "avg_centroid",
                "wave_idx",
            }.issubset(unit):
                trajectory = np.asarray(unit["avg_waveform_per_tp"])
                centroid = np.asarray(unit["avg_centroid"])
                mask_values = np.asarray(unit["wave_idx"])
                selected_cv = 0 if cv == "Avg" else int(
                    cv if label == "Unit A" else 1 - int(cv)
                )
                mask = mask_values[:, selected_cv].astype(bool)
                if cv == "Avg":
                    trajectory = trajectory.mean(axis=-1)
                    centroid = centroid.mean(axis=-1)
                else:
                    trajectory = trajectory[..., selected_cv]
                    centroid = centroid[..., selected_cv]
                trajectory_axis.plot(trajectory[1, mask], trajectory[2, mask], label=label)
                trajectory_axis.scatter(centroid[1], centroid[2])
            if {"acg_rate_hz", "acg_axis_ms"}.issubset(unit):
                acg_axis.plot(unit["acg_axis_ms"], unit["acg_rate_hz"], label=label)
            if {"event_psth_rate_hz", "event_psth_axis_s"}.issubset(unit):
                rates = np.asarray(unit["event_psth_rate_hz"])
                axis = np.asarray(unit["event_psth_axis_s"])
                counts = np.asarray(
                    unit.get("event_psth_trial_count", np.full(len(rates), -1))
                )
                gui_row = int(payload["pair"][f"gui_row{endpoint}"])
                event_rows = payload["availability"]
                if {"gui_row", "event_index", "event_name"}.issubset(event_rows):
                    event_rows = event_rows.loc[
                        (event_rows["artifact"] == "event_psth")
                        & (event_rows["available"].astype(bool))
                        & (event_rows["gui_row"] == gui_row)
                    ].sort_values("event_index")
                    event_names = event_rows["event_name"].astype(str).tolist()
                else:
                    event_names = []
                for event_index, rate in enumerate(rates):
                    event_name = (
                        event_names[event_index]
                        if event_index < len(event_names)
                        else f"event {event_index}"
                    )
                    count = int(counts[event_index]) if event_index < len(counts) else -1
                    count_text = f", n={count}" if count >= 0 else ""
                    event_axis.plot(
                        axis,
                        rate,
                        label=f"{label} {event_name}{count_text}",
                    )

        try:
            spatial = self.spatial_waveforms(pair_id, cv=cv, swapped=swapped)
        except (KeyError, ValueError):
            spatial_axis.text(
                0.5, 0.5, "Spatial waveform unavailable",
                ha="center", va="center", transform=spatial_axis.transAxes,
            )
        else:
            positions = spatial["positions_xy"]
            anchor = np.asarray(spatial["anchor_waveforms"], dtype=float)
            partner = np.asarray(spatial["partner_waveforms"], dtype=float)
            x_pitch = _minimum_positive_spacing(positions[:, 0], 20.0)
            y_pitch = _minimum_positive_spacing(positions[:, 1], x_pitch)
            finite_amplitudes = np.concatenate(
                [anchor[np.isfinite(anchor)], partner[np.isfinite(partner)]]
            )
            amplitude_scale = (
                float(np.max(np.abs(finite_amplitudes)))
                if finite_amplitudes.size
                else 0.0
            )
            vertical_gain = (
                0.4 * y_pitch / amplitude_scale
                if amplitude_scale > 0
                else 0.0
            )
            time_offsets = np.linspace(
                -0.35 * x_pitch,
                0.35 * x_pitch,
                anchor.shape[0],
            )
            spatial_axis.scatter(
                positions[:, 0],
                positions[:, 1],
                color="grey",
                alpha=0.3,
                s=12,
            )
            for index, channel in enumerate(spatial["channels"]):
                x_position, y_position = positions[index]
                spatial_axis.plot(
                    x_position + time_offsets,
                    y_position + anchor[:, index] * vertical_gain,
                    color="#2ca02c",
                    alpha=0.85,
                )
                spatial_axis.plot(
                    x_position + time_offsets,
                    y_position + partner[:, index] * vertical_gain,
                    color="#d62728",
                    alpha=0.75,
                )
                spatial_axis.text(
                    x_position + 0.4 * x_pitch,
                    y_position,
                    str(int(channel)),
                    fontsize=7,
                    va="center",
                )
            spatial_axis.set_xlim(
                np.nanmin(positions[:, 0]) - 0.5 * x_pitch,
                np.nanmax(positions[:, 0]) + 0.7 * x_pitch,
            )
            spatial_axis.set_ylim(
                np.nanmin(positions[:, 1]) - 0.5 * y_pitch,
                np.nanmax(positions[:, 1]) + 0.5 * y_pitch,
            )

        context = payload["plot_context"]
        if cv == "Avg":
            score_prefix = "score_average_"
            histogram_view = "average"
        else:
            reverse = bool(swapped) ^ (int(cv) == 1)
            score_prefix = "score_2_to_1_" if reverse else "score_1_to_2_"
            histogram_view = "cv"
        score_columns = [
            column for column in payload["pair"] if str(column).startswith(score_prefix)
        ]
        for column in score_columns:
            name = column.removeprefix(score_prefix)
            density_key = f"histogram_{histogram_view}_{name}_density"
            edges_key = f"histogram_{histogram_view}_{name}_edges"
            matched_key = (
                f"histogram_{histogram_view}_{name}_matched_density"
            )
            if density_key not in context or edges_key not in context:
                continue
            edges = context[edges_key]
            centers = edges[:-1] + np.diff(edges) / 2
            histogram_axis.plot(
                centers, context[density_key], alpha=0.5, label=f"{name}: all"
            )
            if matched_key in context:
                histogram_axis.plot(
                    centers,
                    context[matched_key],
                    alpha=0.8,
                    label=f"{name}: matches",
                )
            histogram_axis.axvline(
                payload["pair"][column], linestyle="--", alpha=0.8
            )

        displacement = payload["pair_arrays"].get("displacement_vectors_xy")
        if displacement is None:
            displacement_axis.text(
                0.5, 0.5, "Displacement unavailable",
                ha="center", va="center", transform=displacement_axis.transAxes,
            )
        else:
            approved_dx, approved_dy, selected_dx, selected_dy = displacement
            if swapped:
                approved_dx, approved_dy = -approved_dx, -approved_dy
                selected_dx, selected_dy = -selected_dx, -selected_dy
            if np.isfinite([approved_dx, approved_dy]).all():
                displacement_axis.arrow(
                    0, 0, approved_dx, approved_dy,
                    color="#2ca02c", width=0.03, length_includes_head=True,
                )
            if np.isfinite([selected_dx, selected_dy]).all():
                displacement_axis.arrow(
                    0, 0, selected_dx, selected_dy,
                    color="#1f77b4", linestyle="--", width=0.02,
                    length_includes_head=True,
                )
            displacement_axis.set_aspect("equal", adjustable="datalim")

        table_rows = [
            ["Average probability", payload["pair"].get("probability_average")],
            [
                "CV A to B",
                payload["pair"].get(f"probability_{first}_to_{second}"),
            ],
            [
                "CV B to A",
                payload["pair"].get(f"probability_{second}_to_{first}"),
            ],
            ["Event r", payload["pair"].get("event_r")],
            ["Event r availability", payload["pair"].get("event_r_reason", "")],
            ["Amplitude A", payload["pair"].get(f"amplitude{first}")],
            ["Amplitude B", payload["pair"].get(f"amplitude{second}")],
            ["Spatial decay A", payload["pair"].get(f"spatial_decay{first}")],
            ["Spatial decay B", payload["pair"].get(f"spatial_decay{second}")],
            [
                "Displacement consistency",
                payload["pair"].get("displacement_consistency_score"),
            ],
            ["Displacement distance", payload["pair"].get("displacement_distance")],
            ["Displacement residual (um)", payload["pair"].get("displacement_residual_um")],
            [
                "Displacement references",
                payload["pair"].get("displacement_reference_count"),
            ],
            [
                "Minimum references",
                payload["pair"].get("displacement_min_references"),
            ],
            [
                "Displacement availability",
                payload["pair"].get("displacement_reason", ""),
            ],
            [
                "Approved displacement references",
                payload["pair"].get("approved_displacement_count"),
            ],
            [
                "Approved displacement availability",
                payload["pair"].get("approved_displacement_reason", ""),
            ],
            [
                "Selected displacement approved",
                payload["pair"].get("selected_displacement_approved"),
            ],
            [
                "Selected displacement availability",
                payload["pair"].get("selected_displacement_reason", ""),
            ],
        ]
        table_rows.extend(
            [column.removeprefix(score_prefix), payload["pair"][column]]
            for column in score_columns
        )
        score_axis.axis("off")
        score_axis.table(
            cellText=[
                [name, _format_scalar_value(value)]
                for name, value in table_rows
            ],
            colLabels=["Metric", "Value"],
            loc="center",
        )

        availability_axis.axis("off")
        availability = payload["availability"]
        unavailable = (
            availability.loc[~availability["available"].astype(bool)]
            if "available" in availability
            else availability
        )
        availability_axis.text(
            0,
            1,
            "Archived availability\n"
            + (
                "\n".join(
                    f"{row.artifact}: {row.reason}"
                    for row in unavailable.itertuples()
                )
                if len(unavailable)
                else "All captured plot inputs available"
            ),
            va="top",
            wrap=True,
        )

        waveform_axis.set(title="Average waveform", xlabel="Time", ylabel="Amplitude")
        trajectory_axis.set(
            title="Trajectory", xlabel="X position (um)", ylabel="Y position (um)"
        )
        acg_axis.set(title="Autocorrelogram", xlabel="Lag (ms)", ylabel="Rate (Hz)")
        event_axis.set(
            title="Fixed event PSTHs",
            xlabel="Time from event (s)",
            ylabel="Firing rate (Hz)",
        )
        spatial_axis.set(
            title="Anchor-channel spatial waveforms",
            xlabel="X position (um)",
            ylabel="Y position (um)",
        )
        histogram_axis.set(title="Score histograms", xlabel="Score", ylabel="Density")
        displacement_axis.set(
            title="Raw-centroid displacement", xlabel="X (um)", ylabel="Y (um)"
        )
        for axis in figure.axes:
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
        return figure

    def show(self, *, block=True):
        """Open a Tk browser limited to archived accepted pairs and plot modes."""
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self.pairs.empty:
            raise ValueError("This archive has no accepted pairs to display")
        root = tk.Tk()
        root.title(f"Saved UnitMatch review - {self.output.manifest['run_id']}")
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        pair_ids = [int(value) for value in self.pairs["pair_id"]]
        position = tk.IntVar(value=0)
        cv = tk.StringVar(value="Avg")
        swapped = tk.BooleanVar(value=False)
        pair_text = tk.StringVar()
        canvas_holder = ttk.Frame(root)
        canvas_holder.pack(fill="both", expand=True)
        state = {"canvas": None}

        def refresh():
            index = position.get() % len(pair_ids)
            position.set(index)
            pair_id = pair_ids[index]
            row = self.pairs.loc[self.pairs["pair_id"] == pair_id].iloc[0]
            pair_text.set(
                f"{index + 1}/{len(pair_ids)}: "
                f"{row['session_name1']}:{int(row['session_unit_id1'])} - "
                f"{row['session_name2']}:{int(row['session_unit_id2'])}"
            )
            if state["canvas"] is not None:
                state["canvas"].get_tk_widget().destroy()
            figure = self.render_pair(
                pair_id,
                cv=("Avg" if cv.get() == "Avg" else int(cv.get())),
                swapped=swapped.get(),
            )
            state["canvas"] = FigureCanvasTkAgg(figure, master=canvas_holder)
            state["canvas"].draw()
            state["canvas"].get_tk_widget().pack(fill="both", expand=True)

        def move(delta):
            position.set((position.get() + delta) % len(pair_ids))
            refresh()

        ttk.Button(controls, text="Previous", command=lambda: move(-1)).pack(
            side="left"
        )
        ttk.Button(controls, text="Next", command=lambda: move(1)).pack(side="left")
        ttk.Label(controls, textvariable=pair_text, padding=(12, 0)).pack(side="left")
        ttk.Label(controls, text="View:", padding=(12, 0, 2, 0)).pack(side="left")
        cv_box = ttk.Combobox(
            controls,
            textvariable=cv,
            values=("Avg", "0", "1"),
            state="readonly",
            width=5,
        )
        cv_box.pack(side="left")
        cv_box.bind("<<ComboboxSelected>>", lambda _event: refresh())
        ttk.Checkbutton(
            controls,
            text="Swap A/B",
            variable=swapped,
            command=refresh,
        ).pack(side="left", padx=8)
        refresh()
        if block:
            root.mainloop()
        else:
            root.update_idletasks()
        return root


def open_saved_review(run_path_or_loaded, *, show=False, block=True) -> SavedReviewViewer:
    """Open the supported read-only accepted-pair archive interface."""
    loaded = (
        run_path_or_loaded
        if isinstance(run_path_or_loaded, LoadedReviewOutput)
        else load_review_output(run_path_or_loaded)
    )
    viewer = SavedReviewViewer(loaded)
    if show:
        viewer.show(block=block)
    return viewer
