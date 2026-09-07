"""Lazy access to the historical SpikeInterface merge-review diagnostics."""

from numbers import Integral

import numpy as np

from .spikeinterface_diagnostics import _load_unit_diagnostics


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def prepare_unitmatch_diagnostic_data(analyzers_by_session, row_metadata):
    """Provide lazy historical diagnostics for explicit UnitMatch matrix rows.

    Existing ``random_spikes`` and ``waveforms`` extensions are required. The
    callback never computes an extension and never combines recording segments.
    """
    analyzers = tuple(dict(probes) for probes in analyzers_by_session)
    rows = tuple(dict(row) for row in row_metadata)

    def get_diagnostics(row_index):
        row_index = _index(row_index, "UnitMatch row")
        if row_index >= len(rows):
            raise IndexError(f"UnitMatch row {row_index} is outside row_metadata")
        metadata = rows[row_index]
        session = _index(metadata["session_index"], "Session index")
        probe = _index(metadata["probe_n"], "Probe number")
        unit_id = metadata["unit_id"]
        if session >= len(analyzers) or probe not in analyzers[session]:
            raise KeyError(f"No analyzer for session {session}, probe {probe}")
        analyzer = analyzers[session][probe]
        if unit_id not in analyzer.sorting.unit_ids:
            raise KeyError(
                f"Analyzer unit {unit_id!r} is missing from session "
                f"{session}, probe {probe}"
            )
        missing = [
            name
            for name in ("random_spikes", "waveforms")
            if not analyzer.has_extension(name)
        ]
        if missing:
            raise ValueError(
                f"Existing {', '.join(missing)} extension(s) required for "
                f"session {session}, probe {probe}; review does not compute them"
            )
        return _load_unit_diagnostics(analyzer, unit_id)

    return {"get_diagnostics": get_diagnostics}
