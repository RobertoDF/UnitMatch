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


class SpikeInterfaceSessionMerger:
    """Review and soft-merge likely over-split units in one logical session.

    Multiple analyzers can be supplied when a curation pipeline stores unchanged
    and replacement units separately. In that case ``unit_ids`` must contain the
    final unit IDs from the consolidated metrics table.
    """

    def __init__(
        self,
        analyzer,
        unit_ids=None,
        match_threshold=0.5,
        censored_period_ms=0.5,
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

    def display_review(self):
        """Display Approve/Reject controls in a Jupyter notebook."""
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
        rows = []
        for group in self.merge_groups:
            group_key = tuple(group)
            unit_a, unit_b = (unit_index[unit_id] for unit_id in group)
            probability_12 = self.output_prob_matrix[unit_a, unit_b]
            probability_21 = self.output_prob_matrix[unit_b, unit_a]
            label = widgets.HTML(
                value=(
                    f"<b>Units {group[0]} + {group[1]}</b> &nbsp; "
                    f"CV12={probability_12:.3f}, CV21={probability_21:.3f}, "
                    f"mean={(probability_12 + probability_21) / 2:.3f}"
                ),
                layout=widgets.Layout(width="500px"),
            )
            approve_button = widgets.Button(
                description="Approve", button_style="success"
            )
            reject_button = widgets.Button(
                description="Reject", button_style="danger"
            )
            status = widgets.HTML(value="<b>UNDECIDED</b>")

            def record(approved, group_key=group_key, status=status):
                self._set_decision(group_key, approved)
                decision = "APPROVED" if approved else "REJECTED"
                color = "green" if approved else "firebrick"
                status.value = f"<b style='color:{color}'>{decision}</b>"

            approve_button.on_click(lambda _, record=record: record(True))
            reject_button.on_click(lambda _, record=record: record(False))
            rows.append(
                widgets.HBox([label, approve_button, reject_button, status])
            )

        review = widgets.VBox(rows)
        display(review)
        return review

    def apply_merges(self):
        """Soft-merge approved groups after every proposal is reviewed."""
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
            self.merged_analyzer = combined_analyzer.merge_units(
                merge_unit_groups=self.approved_groups,
                censored_period_ms=self.censored_period_ms,
                merging_mode="soft",
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
