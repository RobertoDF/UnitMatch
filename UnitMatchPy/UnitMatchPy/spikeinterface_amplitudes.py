"""Read existing analyzer spike amplitudes for UnitMatch review rows."""

import inspect
from numbers import Integral

import numpy as np


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _spike_indices_by_segment(sorting, unit_id):
    if hasattr(sorting, "get_spike_vector_to_indices"):
        indexed = sorting.get_spike_vector_to_indices()
        return [
            np.asarray(indexed[segment_index][unit_id])
            for segment_index in range(sorting.get_num_segments())
        ]

    spike_vector = np.asarray(sorting.to_spike_vector())
    fields = spike_vector.dtype.names or ()
    if "unit_index" not in fields:
        raise ValueError("Spike vector must contain unit_index")
    unit_ids = list(sorting.unit_ids)
    try:
        unit_index = unit_ids.index(unit_id)
    except ValueError as error:
        raise KeyError(f"Analyzer unit {unit_id!r} is missing from sorting") from error
    segment_indices = (
        spike_vector["segment_index"]
        if "segment_index" in fields
        else np.zeros(spike_vector.size, dtype=np.int64)
    )
    return [
        np.flatnonzero(
            (spike_vector["unit_index"] == unit_index)
            & (segment_indices == segment_index)
        )
        for segment_index in range(sorting.get_num_segments())
    ]


def _amplitude_vector(extension):
    public_parameters = inspect.signature(extension.get_data).parameters.values()
    supports_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in public_parameters
    )
    implementation = getattr(extension, "_get_data", None)
    implementation_parameters = (
        inspect.signature(implementation).parameters
        if implementation is not None
        else {}
    )
    public_parameter_names = {
        parameter.name for parameter in public_parameters
    }
    if "copy" in public_parameter_names or "copy" in implementation_parameters:
        return np.asarray(extension.get_data(outputs="numpy", copy=False))
    if supports_keywords:
        return np.asarray(extension.get_data(outputs="numpy"))

    segmented = extension.get_data(outputs="concatenated")
    if isinstance(segmented, (list, tuple)):
        arrays = [np.asarray(values).reshape(-1) for values in segmented]
        return np.concatenate(arrays) if arrays else np.array([], dtype=float)
    return np.asarray(segmented)


def _load_amplitude_extension(analyzer):
    if hasattr(analyzer, "get_extension"):
        return analyzer.get_extension("spike_amplitudes")
    if hasattr(analyzer, "load_extension"):
        return analyzer.load_extension("spike_amplitudes")
    raise TypeError("Analyzer cannot load its spike_amplitudes extension")


def _amplitude_units(analyzer, extension):
    if hasattr(analyzer, "return_in_uV"):
        return "uV" if analyzer.return_in_uV else "ADC units"
    if hasattr(analyzer, "get_extension") and hasattr(analyzer, "return_scaled"):
        return "uV" if analyzer.return_scaled else "ADC units"

    parameters = getattr(extension, "_params", None) or {}
    if not parameters.get("return_scaled", False):
        return "ADC units"
    recording = None
    if hasattr(analyzer, "has_recording"):
        if analyzer.has_recording():
            recording = analyzer.recording
    elif hasattr(analyzer, "recording"):
        recording = analyzer.recording
    if recording is None:
        return "ADC units"
    has_scaled = getattr(
        recording,
        "has_scaled_traces",
        getattr(recording, "has_scaled", None),
    )
    return "uV" if callable(has_scaled) and has_scaled() else "ADC units"


def prepare_unitmatch_amplitude_data(analyzers_by_session, row_metadata):
    """Provide lazy, spike-aligned amplitude data without computing extensions.

    Resolve each matrix row through its session, probe, and analyzer unit ID,
    never through an exported unit ID. Times follow SpikeInterface's
    ``AmplitudesWidget``: ``get_unit_spike_train(return_times=True)`` in seconds.
    Keep segments separate rather than inventing gaps or concatenation offsets.
    The consumer should cache selected rows and treat returned arrays as read-only.
    """
    analyzers = tuple(dict(probes) for probes in analyzers_by_session)
    rows = tuple(dict(row) for row in row_metadata)

    def get_amplitudes(row_index):
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
        sorting = analyzer.sorting
        if unit_id not in sorting.unit_ids:
            raise KeyError(
                f"Analyzer unit {unit_id!r} is missing from session {session}, probe {probe}"
            )
        if not analyzer.has_extension("spike_amplitudes"):
            raise ValueError(
                f"Existing spike_amplitudes extension is required for session "
                f"{session}, probe {probe}; review does not compute it"
            )
        extension = _load_amplitude_extension(analyzer)
        if extension is None:
            raise ValueError(
                f"Could not load spike_amplitudes for session {session}, probe {probe}"
            )
        amplitudes = _amplitude_vector(extension)
        if amplitudes.ndim != 1:
            raise ValueError("Spike amplitudes must be a one-dimensional spike vector")

        # These absolute indices are also used by SpikeInterface's by_unit output.
        indices_by_segment = _spike_indices_by_segment(sorting, unit_id)
        segments = []
        for segment_index in range(sorting.get_num_segments()):
            indices = np.asarray(indices_by_segment[segment_index])
            times = np.asarray(sorting.get_unit_spike_train(
                unit_id, segment_index=segment_index, return_times=True
            ))
            if times.ndim != 1 or indices.ndim != 1 or times.size != indices.size:
                raise ValueError(
                    f"Spike-time/amplitude index mismatch for unit {unit_id!r}, "
                    f"segment {segment_index}"
                )
            if indices.size and (
                not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or np.any(indices >= amplitudes.size)
            ):
                raise ValueError(
                    f"Invalid amplitude indices for unit {unit_id!r}, segment {segment_index}"
                )
            unit_amplitudes = amplitudes[indices.astype(np.intp, copy=False)]
            if not np.all(np.isfinite(times)) or not np.all(np.isfinite(unit_amplitudes)):
                raise ValueError(
                    f"Nonfinite spike times or amplitudes for unit {unit_id!r}, "
                    f"segment {segment_index}"
                )
            segments.append({
                "segment_index": segment_index,
                "times_s": times,
                "amplitudes": unit_amplitudes,
            })
        return {
            "segments": segments,
            "amplitude_units": _amplitude_units(analyzer, extension),
        }

    return {"get_amplitudes": get_amplitudes}
