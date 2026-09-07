"""Shared scientific machinery for SpikeInterface waveform review."""

from dataclasses import dataclass

import numpy as np


@dataclass
class _UnitDiagnostics:
    times_ms: np.ndarray
    waveforms: np.ndarray
    mean_waveforms: np.ndarray
    channel_indices: np.ndarray
    peak_channel_index: int
    peak_location: np.ndarray
    channel_ids: np.ndarray
    channel_locations: np.ndarray
    spike_times_s: np.ndarray
    spike_amplitudes: np.ndarray
    spike_amplitudes_by_channel: np.ndarray
    all_spike_times_s: np.ndarray
    amplitude_units: str

    def has_channel(self, channel_index):
        return np.any(self.channel_indices == channel_index)

    def _local_channel_index(self, channel_index):
        local_indices = np.flatnonzero(self.channel_indices == channel_index)
        if local_indices.size == 0:
            raise ValueError(
                f"Global channel index {channel_index} is outside the unit's "
                "waveform sparsity."
            )
        return local_indices[0]

    def waveform_on_channel(self, channel_index):
        return self.mean_waveforms[:, self._local_channel_index(channel_index)]

    def waveform_on_channel_for_spikes(self, channel_index, spike_mask):
        local_channel_index = self._local_channel_index(channel_index)
        selected_waveforms = self.waveforms[spike_mask]
        if selected_waveforms.size == 0:
            return np.full(self.times_ms.shape, np.nan)
        return np.mean(
            selected_waveforms[:, :, local_channel_index],
            axis=0,
        )

    def spike_amplitudes_on_channel(self, channel_index):
        return self.spike_amplitudes_by_channel[
            :,
            self._local_channel_index(channel_index),
        ]


def _probe_plot_axes(channel_locations):
    coordinate_ranges = np.ptp(channel_locations, axis=0)
    depth_axis = int(np.argmax(coordinate_ranges))
    remaining_axes = [
        axis for axis in range(channel_locations.shape[1]) if axis != depth_axis
    ]
    horizontal_axis = (
        max(remaining_axes, key=lambda axis: coordinate_ranges[axis])
        if remaining_axes
        else depth_axis
    )
    return horizontal_axis, depth_axis


def _select_waveform_channels(peak_channel_indices, channel_locations):
    """Select three global channels around one or two unit peaks."""
    channel_locations = np.asarray(channel_locations)
    if channel_locations.shape[0] < 3:
        raise ValueError(
            "Waveform review requires at least three recording channels."
        )

    first_peak, second_peak = map(int, peak_channel_indices)
    channel_indices = np.arange(channel_locations.shape[0])
    if first_peak == second_peak:
        distances = np.linalg.norm(
            channel_locations - channel_locations[first_peak],
            axis=1,
        )
        candidates = channel_indices[channel_indices != first_peak]
        nearest = sorted(
            candidates,
            key=lambda index: (distances[index], index),
        )
        return [first_peak, int(nearest[0]), int(nearest[1])]

    excluded = {first_peak, second_peak}
    candidates = [index for index in channel_indices if index not in excluded]
    distance_sums = np.linalg.norm(
        channel_locations - channel_locations[first_peak],
        axis=1,
    ) + np.linalg.norm(
        channel_locations - channel_locations[second_peak],
        axis=1,
    )
    third_channel = min(
        candidates,
        key=lambda index: (distance_sums[index], index),
    )
    return [first_peak, second_peak, int(third_channel)]


def _waveform_units(analyzer, extension):
    if hasattr(analyzer, "return_in_uV"):
        return "uV" if analyzer.return_in_uV else "ADC units"
    if hasattr(analyzer, "return_scaled"):
        return "uV" if analyzer.return_scaled else "ADC units"
    parameters = (
        getattr(extension, "_params", None)
        or getattr(extension, "params", None)
        or {}
    )
    return "uV" if parameters.get("return_scaled", False) else "ADC units"


def _load_existing_extension(analyzer, name):
    if hasattr(analyzer, "get_extension"):
        extension = analyzer.get_extension(name)
    elif hasattr(analyzer, "load_extension"):
        extension = analyzer.load_extension(name)
    else:
        raise TypeError(f"Analyzer cannot load its existing {name} extension")
    if extension is None:
        raise ValueError(f"Could not load existing {name} extension")
    return extension


def _load_unit_diagnostics(analyzer, unit_id):
    """Load the stored waveform diagnostics used by merge review."""
    if analyzer.get_num_segments() != 1:
        raise ValueError(
            "Historical waveform review requires a single recording segment; "
            "segments are never concatenated."
        )
    waveforms_extension = _load_existing_extension(analyzer, "waveforms")
    waveforms = waveforms_extension.get_waveforms_one_unit(unit_id=unit_id)
    mean_waveforms = np.mean(waveforms, axis=0)

    if analyzer.sparsity is None:
        channel_indices = np.arange(analyzer.get_num_channels())
    else:
        channel_indices = analyzer.sparsity.unit_id_to_channel_indices[unit_id]
    mean_waveforms = mean_waveforms[:, : channel_indices.size]
    peak_channel_index = np.argmax(np.max(np.abs(mean_waveforms), axis=0))
    peak_sample_index = np.argmax(
        np.abs(mean_waveforms[:, peak_channel_index])
    )

    ms_before = waveforms_extension.params["ms_before"]
    times_ms = (
        np.arange(mean_waveforms.shape[0])
        / analyzer.sampling_frequency
        * 1000
        - ms_before
    )
    channel_locations = analyzer.get_channel_locations()
    if hasattr(analyzer, "channel_ids"):
        channel_ids = np.asarray(analyzer.channel_ids)
    elif hasattr(analyzer, "get_channel_ids"):
        channel_ids = np.asarray(analyzer.get_channel_ids())
    else:
        raise ValueError(
            "Historical waveform review requires analyzer channel IDs"
        )
    if channel_ids.ndim != 1 or channel_ids.size != channel_locations.shape[0]:
        raise ValueError(
            "Analyzer channel IDs must match channel geometry rows"
        )
    peak_location = channel_locations[channel_indices[peak_channel_index]]
    random_spikes = _load_existing_extension(analyzer, "random_spikes")
    selected_spike_indices = (
        random_spikes.get_selected_indices_in_spike_train(
            unit_id=unit_id,
            segment_index=0,
        )
    )
    if selected_spike_indices.size != waveforms.shape[0]:
        raise ValueError(
            f"Unit {unit_id} has {selected_spike_indices.size} selected spikes "
            f"but {waveforms.shape[0]} stored waveforms."
        )
    all_spike_times_s = (
        analyzer.sorting.get_unit_spike_train(unit_id=unit_id)
        / analyzer.sampling_frequency
    )
    spike_times_s = all_spike_times_s[selected_spike_indices]
    spike_amplitudes_by_channel = waveforms[
        :,
        peak_sample_index,
        : channel_indices.size,
    ]
    spike_amplitudes = spike_amplitudes_by_channel[:, peak_channel_index]
    return _UnitDiagnostics(
        times_ms=times_ms,
        waveforms=waveforms[:, :, : channel_indices.size],
        mean_waveforms=mean_waveforms,
        channel_indices=channel_indices,
        peak_channel_index=int(channel_indices[peak_channel_index]),
        peak_location=peak_location,
        channel_ids=channel_ids,
        channel_locations=channel_locations,
        spike_times_s=spike_times_s,
        spike_amplitudes=spike_amplitudes,
        spike_amplitudes_by_channel=spike_amplitudes_by_channel,
        all_spike_times_s=all_spike_times_s,
        amplitude_units=_waveform_units(analyzer, waveforms_extension),
    )


def _waveforms_for_interval(diagnostics, channel_indices, interval=None):
    """Return historical review means and sampled-spike counts for an interval."""
    if interval is not None:
        time_start_s, time_stop_s = sorted(map(float, interval))
    waveforms = {}
    selected_counts = {}
    for diagnostic_index, diagnostic in enumerate(diagnostics):
        spike_mask = None
        if interval is not None:
            spike_mask = (
                (diagnostic.spike_times_s >= time_start_s)
                & (diagnostic.spike_times_s <= time_stop_s)
            )
        selected_counts[diagnostic_index] = (
            len(diagnostic.spike_times_s)
            if spike_mask is None
            else int(np.count_nonzero(spike_mask))
        )
        for axis_index, channel_index in enumerate(channel_indices):
            if not diagnostic.has_channel(channel_index):
                continue
            waveform = (
                diagnostic.waveform_on_channel(channel_index)
                if spike_mask is None
                else diagnostic.waveform_on_channel_for_spikes(
                    channel_index,
                    spike_mask,
                )
            )
            waveforms[(axis_index, diagnostic_index)] = waveform
    return waveforms, selected_counts
