import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import bayes_functions as bf
from . import overlord as ov
from . import utils as util
from .assign_unique_id import get_within_session_merge_groups
from .default_params import get_default_param
from .save_utils import make_UnitMatch_folder_from_sorting_analyzers


class SpikeInterfaceSessionMerger:
    """Review and soft-merge likely over-split units in one session."""

    def __init__(
        self,
        analyzer,
        match_threshold=0.5,
        censored_period_ms=0.5,
    ):
        self.analyzer = analyzer
        self.match_threshold = match_threshold
        self.censored_period_ms = censored_period_ms
        self.output_prob_matrix = None
        self.merge_groups = []
        self.decisions = {}
        self.merged_analyzer = None
        self._validate_analyzer()

    def _validate_analyzer(self):
        missing = [
            name
            for name in ("random_spikes", "waveforms")
            if not self.analyzer.has_extension(name)
        ]
        if missing:
            raise ValueError(
                f"Analyzer is missing required extensions: {missing}. "
                "Compute them before creating the merger."
            )
        if self.analyzer.get_num_segments() != 1:
            raise ValueError("Only single-segment analyzers are supported.")

    def compute_proposals(self):
        """Run session-local UnitMatch and return ISI-safe merge pairs."""
        with tempfile.TemporaryDirectory(prefix="unitmatch-session-") as temp_dir:
            export_dir = Path(temp_dir)
            make_UnitMatch_folder_from_sorting_analyzers(
                [self.analyzer], export_dir
            )
            wave_path = export_dir / "Session0"
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

            unit_ids_per_session = [self.analyzer.unit_ids]
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
                self.analyzer.sorting.get_unit_spike_train(unit_id=unit_id)
                / self.analyzer.sampling_frequency
                for unit_id in self.analyzer.unit_ids
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
            for index, unit_id in enumerate(self.analyzer.unit_ids.tolist())
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
        if self.approved_groups:
            self.merged_analyzer = self.analyzer.merge_units(
                merge_unit_groups=self.approved_groups,
                censored_period_ms=self.censored_period_ms,
                merging_mode="soft",
            )
        else:
            self.merged_analyzer = self.analyzer
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
