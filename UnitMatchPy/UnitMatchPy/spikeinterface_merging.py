import inspect
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import bayes_functions as bf
from . import overlord as ov
from . import utils as util
from .assign_unique_id import get_within_session_merge_groups
from .default_params import get_default_param
from .save_utils import make_UnitMatch_folder_from_sorting_analyzers


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


class SpikeInterfaceSessionMerger:
    """Review and soft-merge likely over-split units in one logical session.

    Multiple analyzers can be supplied when a curation pipeline stores unchanged
    and replacement units separately. In that case ``unit_ids`` must contain the
    final unit IDs from the consolidated metrics table. Same-session candidates
    must have centroids within 50 um, a more conservative limit than UnitMatch's
    100 um default for matching units across recording sessions.
    """

    MAX_DISTANCE_UM = 50

    def __init__(
        self,
        analyzer,
        unit_ids=None,
        match_threshold=0.5,
        censored_period_ms=None,
        merging_mode="soft",
    ):
        if isinstance(analyzer, Sequence):
            self.analyzers = list(analyzer)
        else:
            self.analyzers = [analyzer]
        if not self.analyzers:
            raise ValueError("At least one analyzer is required.")

        self.unit_ids = self._normalize_unit_ids(unit_ids)
        self._unit_sources = self._resolve_unit_sources()
        self.analyzer = self.analyzers[0]
        self.match_threshold = match_threshold
        self.censored_period_ms = censored_period_ms
        if merging_mode not in {"soft", "hard"}:
            raise ValueError("merging_mode must be either 'soft' or 'hard'.")
        self.merging_mode = merging_mode
        self.output_prob_matrix = None
        self.merge_groups = []
        self.decisions = {}
        self.merged_analyzer = None
        self._validate_analyzer()

    def _normalize_unit_ids(self, unit_ids):
        if unit_ids is None:
            if len(self.analyzers) > 1:
                raise ValueError(
                    "unit_ids is required for multiple analyzers. Pass the final "
                    "unit IDs from metrics.index."
                )
            return np.asarray(self.analyzers[0].unit_ids)

        normalized = np.asarray(list(unit_ids))
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError("unit_ids must be a non-empty one-dimensional sequence.")
        if np.unique(normalized).size != normalized.size:
            raise ValueError("unit_ids contains duplicates.")
        return normalized

    def _resolve_unit_sources(self):
        sources = {}
        missing = []
        for unit_id in self.unit_ids:
            matches = [
                analyzer
                for analyzer in self.analyzers
                if unit_id in set(analyzer.unit_ids.tolist())
            ]
            if not matches:
                missing.append(unit_id)
            elif len(matches) > 1:
                raise ValueError(
                    f"Unit {unit_id!r} occurs in more than one analyzer. "
                    "Pass complementary analyzers with unambiguous final units."
                )
            else:
                sources[unit_id] = matches[0]
        if missing:
            raise ValueError(f"Final unit IDs are missing from the analyzers: {missing}")
        return sources

    def _validate_analyzer(self):
        reference = self.analyzers[0]
        for analyzer in self.analyzers:
            missing = [
                name
                for name in ("random_spikes", "waveforms")
                if not analyzer.has_extension(name)
            ]
            if missing:
                raise ValueError(
                    f"Analyzer is missing required extensions: {missing}. "
                    "Compute them before creating the merger."
                )
            if analyzer.get_num_segments() != 1:
                raise ValueError("Only single-segment analyzers are supported.")
            if analyzer.sampling_frequency != reference.sampling_frequency:
                raise ValueError("All analyzers must have the same sampling frequency.")
            if analyzer.get_num_samples() != reference.get_num_samples():
                raise ValueError("All analyzers must cover the same recording duration.")
            if not np.array_equal(
                analyzer.get_channel_locations(),
                reference.get_channel_locations(),
            ):
                raise ValueError("All analyzers must use the same channel geometry.")

    def _export_composite_session(self, export_dir):
        source_dir = export_dir / "sources"
        make_UnitMatch_folder_from_sorting_analyzers(
            self.analyzers, source_dir
        )
        session_dir = export_dir / "Session0"
        session_dir.mkdir()

        source_indices = {
            id(analyzer): index for index, analyzer in enumerate(self.analyzers)
        }
        for unit_id in self.unit_ids:
            source_index = source_indices[id(self._unit_sources[unit_id])]
            shutil.copy2(
                source_dir
                / f"Session{source_index}"
                / f"Unit{unit_id}_RawSpikes.npy",
                session_dir / f"Unit{unit_id}_RawSpikes.npy",
            )

        first_source = source_dir / "Session0"
        shutil.copy2(
            first_source / "channel_locations.npy",
            session_dir / "channel_locations.npy",
        )
        shutil.copy2(
            first_source / "waveform_params.json",
            session_dir / "waveform_params.json",
        )
        return session_dir

    def compute_proposals(self):
        """Run session-local UnitMatch and return ISI-safe merge pairs."""
        with tempfile.TemporaryDirectory(prefix="unitmatch-session-") as temp_dir:
            export_dir = Path(temp_dir)
            wave_path = self._export_composite_session(export_dir)
            channel_pos = [np.load(wave_path / "channel_locations.npy")]
            with (wave_path / "waveform_params.json").open(
                encoding="utf-8"
            ) as stream:
                waveform_params = json.load(stream)

            param = get_default_param()
            param.update(waveform_params)
            param["waveidx"] = np.asarray(param["waveidx"], dtype=int)
            param["match_threshold"] = self.match_threshold
            param["max_dist"] = self.MAX_DISTANCE_UM
            param = util.get_probe_geometry(channel_pos[0], param)

            unit_ids_per_session = [self.unit_ids]
            (
                waveform,
                session_id,
                session_switch,
                within_session,
                unit_ids,
                param,
            ) = util.load_waveforms([wave_path], unit_ids_per_session, param)

        clus_info = {
            "good_units": unit_ids,
            "session_switch": session_switch,
            "session_id": session_id,
            "original_ids": np.concatenate(unit_ids),
            "spike_times": [
                self._unit_sources[unit_id].sorting.get_unit_spike_train(
                    unit_id=unit_id
                )
                / self._unit_sources[unit_id].sampling_frequency
                for unit_id in self.unit_ids
            ],
        }
        properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)
        _, candidate_pairs, scores, predictors = ov.extract_metric_scores(
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
        )
        self.output_prob_matrix = probability[:, 1].reshape(
            param["n_units"], param["n_units"]
        )
        self.merge_groups = get_within_session_merge_groups(
            self.output_prob_matrix,
            param,
            clus_info,
            match_threshold=self.match_threshold,
        )
        self.decisions = {tuple(group): None for group in self.merge_groups}
        self.merged_analyzer = None
        return self.merge_groups

    def approve(self, group):
        """Approve one proposed merge group."""
        self._set_decision(group, True)

    def reject(self, group):
        """Reject one proposed merge group."""
        self._set_decision(group, False)

    def _set_decision(self, group, approved):
        group = tuple(group)
        if group not in self.decisions:
            raise ValueError(f"{list(group)} is not a proposed merge group.")
        self.decisions[group] = approved

    @property
    def approved_groups(self):
        return [list(group) for group, approved in self.decisions.items() if approved]

    @property
    def undecided_groups(self):
        return [
            list(group)
            for group, decision in self.decisions.items()
            if decision is None
        ]

    def _get_unit_diagnostics(self, unit_id):
        analyzer = self._unit_sources[unit_id]
        waveforms_extension = analyzer.get_extension("waveforms")
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
            np.arange(mean_waveforms.shape[0]) / analyzer.sampling_frequency * 1000
            - ms_before
        )
        channel_locations = analyzer.get_channel_locations()
        peak_location = channel_locations[channel_indices[peak_channel_index]]
        random_spikes = analyzer.get_extension("random_spikes")
        selected_spike_indices = random_spikes.get_selected_indices_in_spike_train(
            unit_id=unit_id,
            segment_index=0,
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
        spike_amplitudes = waveforms[
            :, peak_sample_index, peak_channel_index
        ]
        return (
            times_ms,
            mean_waveforms[:, peak_channel_index],
            peak_location,
            channel_locations,
            spike_times_s,
            spike_amplitudes,
            all_spike_times_s,
        )

    def _make_group_figure(self, group):
        import matplotlib.pyplot as plt

        colors = ("tab:blue", "tab:orange")
        diagnostics = [
            self._get_unit_diagnostics(unit_id) for unit_id in group
        ]
        figure, (
            waveform_axis,
            probe_axis,
            amplitude_axis,
            rate_axis,
        ) = plt.subplots(
            1, 4, figsize=(16, 3), constrained_layout=True
        )

        for unit_id, color, diagnostic in zip(group, colors, diagnostics):
            times_ms, waveform = diagnostic[:2]
            waveform_axis.plot(
                times_ms, waveform, color=color, label=f"Unit {unit_id}"
            )
        waveform_axis.axvline(0, color="0.7", linewidth=0.8)
        waveform_axis.set(
            title="Mean waveform on peak channel",
            xlabel="Time (ms)",
            ylabel="Amplitude",
        )
        waveform_axis.legend()

        channel_locations = diagnostics[0][3]
        horizontal_axis, depth_axis = _probe_plot_axes(channel_locations)
        probe_axis.scatter(
            channel_locations[:, horizontal_axis],
            channel_locations[:, depth_axis],
            color="0.75",
            s=18,
            label="Channels",
        )
        for unit_id, color, diagnostic in zip(group, colors, diagnostics):
            peak_location = diagnostic[2]
            probe_axis.scatter(
                peak_location[horizontal_axis],
                peak_location[depth_axis],
                color=color,
                edgecolor="black",
                s=80,
                alpha=0.5,
                label=f"Unit {unit_id} peak",
                zorder=3,
            )
        probe_axis.set(
            title="Peak location on probe",
            xlabel=f"Coordinate {horizontal_axis} (um)",
            ylabel=f"Coordinate {depth_axis} (um)",
        )
        probe_axis.legend(fontsize="small")

        for unit_id, color, diagnostic in zip(group, colors, diagnostics):
            spike_times_s, spike_amplitudes = diagnostic[4:6]
            amplitude_axis.scatter(
                spike_times_s,
                spike_amplitudes,
                color=color,
                s=10,
                alpha=0.6,
                label=f"Unit {unit_id}",
            )
        amplitude_axis.set(
            title="Spike amplitudes over time",
            xlabel="Time (s)",
            ylabel="Signed peak amplitude",
        )
        amplitude_axis.legend(fontsize="small")

        duration_s = max(
            analyzer.get_num_samples() / analyzer.sampling_frequency
            for analyzer in (
                self._unit_sources[unit_id] for unit_id in group
            )
        )
        bin_width_s = min(10.0, max(0.1, duration_s / 20))
        bin_edges = np.arange(0, duration_s, bin_width_s)
        bin_edges = np.append(bin_edges, duration_s)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        for unit_id, color, diagnostic in zip(group, colors, diagnostics):
            spike_counts, _ = np.histogram(diagnostic[6], bins=bin_edges)
            firing_rates = spike_counts / bin_widths
            rate_axis.plot(
                bin_centers,
                firing_rates,
                color=color,
                alpha=0.5,
                label=f"Unit {unit_id}",
            )
        rate_axis.set(
            title="Spike rate over time",
            xlabel="Time (s)",
            ylabel="Firing rate (Hz)",
        )
        rate_axis.legend(fontsize="small")
        return figure

    def display_review(self, show_diagnostics=True):
        """Display candidate diagnostics and Approve/Reject controls."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as error:
            raise ImportError(
                "Install UnitMatchPy with the 'notebooks' extra to use review widgets."
            ) from error

        if self.output_prob_matrix is None:
            raise RuntimeError("Call compute_proposals() before display_review().")
        if not self.merge_groups:
            print("No pairs passed the probability and ISI checks.")
            return None

        unit_index = {
            unit_id: index
            for index, unit_id in enumerate(self.unit_ids.tolist())
        }
        navigator = widgets.IntSlider(
            value=1,
            min=1,
            max=len(self.merge_groups),
            step=1,
            description="Pair",
            continuous_update=False,
            readout_format="d",
            layout=widgets.Layout(width="500px"),
        )
        instructions = widgets.HTML(
            value=(
                "Click the pair slider, then use the keyboard "
                "<b>left/right arrow keys</b> to move between candidates."
            )
        )
        label = widgets.HTML(layout=widgets.Layout(width="500px"))
        approve_button = widgets.Button(
            description="Approve", button_style="success"
        )
        reject_button = widgets.Button(
            description="Reject", button_style="danger"
        )
        status = widgets.HTML()
        plot_output = widgets.Output()

        def current_group():
            return self.merge_groups[navigator.value - 1]

        def render_candidate(*_):
            group = current_group()
            unit_a, unit_b = (unit_index[unit_id] for unit_id in group)
            probability_12 = self.output_prob_matrix[unit_a, unit_b]
            probability_21 = self.output_prob_matrix[unit_b, unit_a]
            label.value = (
                f"<b>Units {group[0]} + {group[1]}</b> &nbsp; "
                f"CV12={probability_12:.3f}, CV21={probability_21:.3f}, "
                f"mean={(probability_12 + probability_21) / 2:.3f}"
            )

            decision = self.decisions[tuple(group)]
            if decision is None:
                status.value = "<b>UNDECIDED</b>"
            else:
                text = "APPROVED" if decision else "REJECTED"
                color = "green" if decision else "firebrick"
                status.value = f"<b style='color:{color}'>{text}</b>"

            plot_output.clear_output(wait=True)
            if show_diagnostics:
                import matplotlib.pyplot as plt

                with plot_output:
                    figure = self._make_group_figure(group)
                    display(figure)
                    plt.close(figure)

        def record(approved):
            self._set_decision(tuple(current_group()), approved)
            if navigator.value < navigator.max:
                navigator.value += 1
            else:
                render_candidate()

        approve_button.on_click(lambda _: record(True))
        reject_button.on_click(lambda _: record(False))
        navigator.observe(render_candidate, names="value")

        controls = widgets.HBox(
            [label, approve_button, reject_button, status]
        )
        children = [instructions, navigator, controls]
        if show_diagnostics:
            children.append(plot_output)
        review = widgets.VBox(children)
        render_candidate()
        display(review)
        return review

    def apply_merges(self):
        """Merge approved groups after every proposal is reviewed."""
        if self.undecided_groups:
            raise RuntimeError(
                "Approve or reject every proposed merge before applying: "
                f"{self.undecided_groups}"
            )
        if len(self.analyzers) == 1:
            if np.array_equal(self.unit_ids, self.analyzer.unit_ids):
                combined_analyzer = self.analyzer
            else:
                combined_analyzer = self.analyzer.select_units(self.unit_ids)
        else:
            try:
                from spikeinterface import (
                    SortingAnalyzer,
                    aggregate_units,
                    create_sorting_analyzer,
                )
            except ImportError as error:
                raise ImportError(
                    "SpikeInterface is required to combine multiple analyzers."
                ) from error

            selected_sortings = [
                self._unit_sources[unit_id].sorting.select_units([unit_id])
                for unit_id in self.unit_ids
            ]
            combined_sorting = aggregate_units(
                selected_sortings, renamed_unit_ids=self.unit_ids
            )
            recording = getattr(self.analyzer, "recording", None)
            if recording is not None:
                combined_analyzer = create_sorting_analyzer(
                    sorting=combined_sorting,
                    recording=recording,
                    format="memory",
                    sparse=False,
                )
            else:
                combined_analyzer = SortingAnalyzer.create_memory(
                    sorting=combined_sorting,
                    recording=None,
                    sparsity=None,
                    return_in_uV=self.analyzer.return_in_uV,
                    peak_sign=self.analyzer.peak_sign,
                    peak_mode=self.analyzer.peak_mode,
                    rec_attributes=self.analyzer.rec_attributes,
                )

        if self.approved_groups:
            if (
                self.merging_mode == "hard"
                and getattr(combined_analyzer, "recording", None) is None
            ):
                raise RuntimeError(
                    "Hard merging requires access to the recording traces. "
                    "Use merging_mode='soft' for a recordingless analyzer."
                )
            merge_parameters = inspect.signature(
                combined_analyzer.merge_units
            ).parameters
            if "censor_ms" in merge_parameters:
                censor_argument = {"censor_ms": self.censored_period_ms}
            elif "censored_period_ms" in merge_parameters:
                censor_argument = {
                    "censored_period_ms": self.censored_period_ms
                }
            else:
                raise TypeError(
                    "Unsupported SpikeInterface merge_units() signature: "
                    "missing censor_ms/censored_period_ms."
                )
            self.merged_analyzer = combined_analyzer.merge_units(
                merge_unit_groups=self.approved_groups,
                merging_mode=self.merging_mode,
                **censor_argument,
            )
        else:
            self.merged_analyzer = combined_analyzer
        return self.merged_analyzer

    def save(self, folder, format="binary_folder", overwrite=False):
        """Persist the applied analyzer and its propagated extensions."""
        if self.merged_analyzer is None:
            raise RuntimeError("Call apply_merges() before save().")
        folder = Path(folder)
        if folder.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{folder} already exists. Pass overwrite=True to replace it."
                )
            shutil.rmtree(folder)
        return self.merged_analyzer.save_as(folder=folder, format=format)

    def prepare_for_unitmatch(self, channel_radius=None, job_kwargs=None):
        """Rebuild the merged analyzer with the data required by UnitMatch."""
        if self.merged_analyzer is None:
            raise RuntimeError("Call apply_merges() before preparing the analyzer.")

        recording = getattr(self.merged_analyzer, "recording", None)
        if recording is None:
            raise RuntimeError(
                "Preparing a merged analyzer for UnitMatch requires recording traces."
            )

        if channel_radius is None:
            channel_radius = get_default_param()["channel_radius"]
        if job_kwargs is None:
            job_kwargs = {}

        from spikeinterface import create_sorting_analyzer

        self.merged_analyzer = create_sorting_analyzer(
            sorting=self.merged_analyzer.sorting,
            recording=recording,
            format="memory",
            sparse=True,
            method="radius",
            radius_um=channel_radius,
            **job_kwargs,
        )
        self.merged_analyzer.compute(
            ["random_spikes", "waveforms"],
            **job_kwargs,
        )
        return self.merged_analyzer

    def apply_and_save(
        self,
        folder,
        format="binary_folder",
        overwrite=False,
        channel_radius=None,
        job_kwargs=None,
    ):
        """Apply merges, prepare for UnitMatch, and persist the analyzer."""
        self.apply_merges()
        self.prepare_for_unitmatch(
            channel_radius=channel_radius,
            job_kwargs=job_kwargs,
        )
        return self.save(folder, format=format, overwrite=overwrite)
