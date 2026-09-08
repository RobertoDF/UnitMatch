from tkinter import *
from tkinter import ttk
from tkinter import font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib import rcParams
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event


UNIT_A_COLOR = "#EB6534"
UNIT_B_COLOR = "#FBFAF8"
APPROVED_MATCH_COLOR = "#43A047"
BETTER_ALTERNATIVE_COLOR = "#AB47BC"
NO_ACCEPTED_ALTERNATIVE_COLOR = "#E53935"
ALL_SCORES_COLOR = "#C0E6DE"
DISPLACEMENT_ANGLE_THRESHOLD = 60
GUI_REFERENCE_WIDTH = 1820
GUI_REFERENCE_HEIGHT = 980
GUI_MIN_SCALE = 0.5
RAW_WAVEFORM_AXES_BOUNDS = (0.2, 0.12, 0.74, 0.48)
RAW_DISPLACEMENT_AXES_BOUNDS = (0.08, 0.65, 0.48, 0.32)
gui_scale = 1.0
event_data = None
event_psth_cache = {}
event_spike_times_cache = {}
event_correlation_cache = {}
event_view_settings = (1.0, 2.0, 0.01)
event_metric_executor = None
displacement_metric_executor = None
displacement_context = None
displacement_context_key = None
displacement_min_references = 5
consistency_filter_threshold = 20.0
consistency_filter_pairs = set()
consistency_filter_unavailable = 0
consistency_filter_cancel = Event()
consistency_filter_future = None
consistency_filter_after = None
diagnostic_help_window = None
histogram_panel = None
raw_waveform_pair_lines = None
raw_waveform_view_key = None
pending_redraw = None
session_summary_label = None
session_group_summary = None
unit_tables_frame = None
score_histogram_cache = {}
event_view_window = None
event_view_refresh = None
raw_avg_centroid = None
raw_waveform_figure = None
raw_waveform_canvas = None
raw_displacement_axis = None
unusual_displacement_pairs = set()
unusual_displacement_group_info = {}
automatic_match_pairs = set()
is_match = []
not_match = []


def _get_gui_scale(screen_width, screen_height):
    usable_width = max(screen_width - 100, 1)
    usable_height = max(screen_height - 100, 1)
    available_scale = min(
        usable_width / GUI_REFERENCE_WIDTH,
        usable_height / GUI_REFERENCE_HEIGHT,
    )
    return min(1.0, max(GUI_MIN_SCALE, available_scale))


def _scaled_figsize(width, height):
    return width * gui_scale, height * gui_scale


def _scaled_font_size(size, minimum=8):
    return max(minimum, round(size * gui_scale))


class _UnitTable(ttk.Frame):
    """Common scrolling and selection behavior for the two unit tables."""

    def __init__(self, master, values):
        super().__init__(master)
        self._selected_unit = ""
        self.table = ttk.Treeview(
            self, columns=tuple(column for column, _, _ in self.columns),
            show="headings", selectmode="browse", height=5, style=self.table_style,
        )
        for column, title, width in self.columns:
            self.table.heading(column, text=title)
            self.table.column(column, width=round(width * gui_scale), minwidth=40)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = ttk.Scrollbar(
            self, orient="horizontal", command=self.table.xview,
        )
        self.table.configure(xscrollcommand=horizontal_scrollbar.set)
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        style = ttk.Style(self)
        style.configure(self.table_style, rowheight=_scaled_font_size(20))
        # Keep tag colors visible even on Windows themes and selected rows.
        style.map(self.table_style, foreground=[], background=[("selected", "#50585b")])
        self.set_options(values)
        self.table.bind("<<TreeviewSelect>>", self._select)

    def _clear_options(self):
        children = self.table.get_children()
        if children:
            self.table.delete(*children)
        self._selected_unit = ""

    def get(self):
        return self._selected_unit

    def set(self, unit):
        unit = str(unit)
        self._selected_unit = unit
        if self.table.exists(unit):
            self.table.selection_set(unit)
            self.table.focus(unit)
            self.table.see(unit)
        else:
            self.table.selection_remove(self.table.selection())

    def _select(self, event):
        selection = self.table.selection()
        if selection and selection[0] != self._selected_unit:
            self._selected_unit = selection[0]
            self.event_generate(self.selection_event)


class _DiagnosticUnitTable(_UnitTable):
    """Shared pair diagnostics, asynchronous calculation, and cancellation."""

    def __init__(self, master, values):
        self._metric_cancel = Event()
        self._metric_future = None
        self._metric_after = None
        self._metric_pairs = []
        self._displacement_cancel = Event()
        self._displacement_future = None
        self._displacement_after = None
        self._displacement_details = {}
        self._tooltip = None
        self._tooltip_after = None
        self._tooltip_row = None
        super().__init__(master, values)
        self.table.bind("<Motion>", self._hover_displacement)
        self.table.bind("<Leave>", self._hide_displacement_tooltip)
        self.table.bind("<ButtonPress>", self._hide_displacement_tooltip, add="+")

    def _clear_diagnostics(self):
        self._cancel_event_metrics()
        self._cancel_displacement_metrics()
        self._hide_displacement_tooltip()
        self._clear_options()
        self._metric_pairs = []
        self._displacement_details = {}

    def _set_pair_metrics(self, row, unit_a, unit_b):
        magnitude, angle = _pair_displacement_metrics(unit_a, unit_b)
        self.table.set(row, "angle", "n/a" if angle is None else f"{angle:.1f}")
        self.table.set(row, "distance", "n/a" if magnitude is None else f"{magnitude:.1f}")
        self.table.set(row, "event_r", "..." if event_data is not None else "n/a")
        self.table.set(row, "displacement", "...")
        self._metric_pairs.append((row, unit_a, unit_b))

    def _queue_event_metrics(self):
        if self._metric_pairs and event_data is not None:
            self._metric_after = self.after_idle(self.refresh_event_metrics)
        if self._metric_pairs:
            self._displacement_after = self.after_idle(self.refresh_displacement_metrics)

    def _cancel_displacement_metrics(self):
        self._displacement_cancel.set()
        if self._displacement_future is not None:
            self._displacement_future.cancel()
        if self._displacement_after is not None:
            self.after_cancel(self._displacement_after)
            self._displacement_after = None

    def refresh_displacement_metrics(self):
        global displacement_metric_executor
        self._cancel_displacement_metrics()
        self._hide_displacement_tooltip()
        self._displacement_details = {}
        if not self._metric_pairs:
            return
        model, reason, scope = _get_displacement_context()
        self._displacement_scope = scope
        for row, _, _ in self._metric_pairs:
            self.table.set(row, "displacement", "..." if model is not None else "n/a")
            self._displacement_details[row] = reason or "Calculating displacement consistency..."
        if model is None:
            return
        self._displacement_cancel = Event()
        self._displacement_results = Queue()
        if displacement_metric_executor is None:
            displacement_metric_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="unitmatch-displacement",
            )
        pairs = sorted(self._metric_pairs, key=lambda pair: pair[0] != self.get())
        self._displacement_future = displacement_metric_executor.submit(
            _compute_displacement_consistency, pairs, model,
            self._displacement_cancel, self._displacement_results,
        )
        self._displacement_after = self.after(50, self._poll_displacement_metrics)

    def _poll_displacement_metrics(self):
        self._displacement_after = None
        finished = self._displacement_future.done()
        while True:
            try:
                row, result = self._displacement_results.get_nowait()
            except Empty:
                break
            self.table.set(
                row, "displacement", "n/a" if result.score is None else f"{result.score:.1f}",
            )
            details = (
                f"Reference: {result.reference_count} independent accepted pairs "
                f"(minimum {displacement_min_references}).\n"
                f"Scope: {self._displacement_scope}; same session pair.\n"
                "Both candidate endpoints excluded."
            )
            if result.score is None:
                details = f"Unavailable: {result.reason}\n{details}"
            else:
                details = (
                    f"Consistency: {result.score:.1f}/100 (not a match probability).\n"
                    f"Residual: {result.residual_um:.2f} um; "
                    f"normalized distance: {result.distance:.2f}.\n{details}"
                )
            self._displacement_details[row] = details
            if self._tooltip is not None and self._tooltip_row == row:
                self._tooltip_label.configure(text=details)
        if finished:
            error = self._displacement_future.exception()
            if error is not None:
                for row, _, _ in self._metric_pairs:
                    if self.table.set(row, "displacement") == "...":
                        self.table.set(row, "displacement", "error")
                        self._displacement_details[row] = f"Calculation failed: {error}"
            self._displacement_future.result()
        else:
            self._displacement_after = self.after(50, self._poll_displacement_metrics)

    def _hover_displacement(self, event):
        column = self.table.identify_column(event.x)
        expected = f"#{tuple(self.table['columns']).index('displacement') + 1}"
        row = self.table.identify_row(event.y) if column == expected else ""
        if row == self._tooltip_row:
            return
        self._hide_displacement_tooltip()
        if row:
            self._tooltip_row = row
            self._tooltip_after = self.after(
                300, lambda: self._show_displacement_tooltip(event.x_root, event.y_root),
            )

    def _show_displacement_tooltip(self, x, y):
        self._tooltip_after = None
        self._tooltip = Toplevel(self)
        self._tooltip.overrideredirect(True)
        self._tooltip_label = ttk.Label(
            self._tooltip, text=self._displacement_details.get(self._tooltip_row, ""),
            padding=8, wraplength=350, justify="left",
        )
        self._tooltip_label.pack()
        self._tooltip.update_idletasks()
        x = min(x + 12, self.winfo_screenwidth() - self._tooltip.winfo_reqwidth() - 10)
        y = min(y + 16, self.winfo_screenheight() - self._tooltip.winfo_reqheight() - 10)
        self._tooltip.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _hide_displacement_tooltip(self, event=None):
        if self._tooltip_after is not None:
            self.after_cancel(self._tooltip_after)
            self._tooltip_after = None
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None
        self._tooltip_row = None

    def _cancel_event_metrics(self):
        self._metric_cancel.set()
        if self._metric_future is not None:
            self._metric_future.cancel()
        if self._metric_after is not None:
            self.after_cancel(self._metric_after)
            self._metric_after = None

    def refresh_event_metrics(self):
        global event_metric_executor
        self._cancel_event_metrics()
        if event_data is None or not self.table.get_children():
            return
        pairs = sorted(self._metric_pairs, key=lambda pair: pair[0] != self.get())
        for row, _, _ in pairs:
            self.table.set(row, "event_r", "...")
        context = (
            event_data, clus_info["session_indices"],
            event_spike_times_cache, event_psth_cache, event_correlation_cache,
        )
        self._metric_cancel = Event()
        self._metric_results = Queue()
        if event_metric_executor is None:
            event_metric_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="unitmatch-events"
            )
        self._metric_future = event_metric_executor.submit(
            _compute_candidate_event_correlations, pairs,
            event_view_settings, context, self._metric_cancel, self._metric_results,
        )
        self._metric_after = self.after(50, self._poll_event_metrics)

    def _poll_event_metrics(self):
        self._metric_after = None
        finished = self._metric_future.done()
        while True:
            try:
                unit, correlation, error = self._metric_results.get_nowait()
            except Empty:
                break
            if error:
                self.table.set(str(unit), "event_r", "error")
                warnings.warn(f"Event correlation for {type(self).__name__} row {unit}: {error}", RuntimeWarning)
            else:
                self.table.set(
                    str(unit), "event_r",
                    "n/a" if correlation is None else f"{correlation:.3f}",
                )
        if finished:
            self._metric_future.result()
        else:
            self._metric_after = self.after(50, self._poll_event_metrics)

    def destroy(self):
        self._cancel_event_metrics()
        self._cancel_displacement_metrics()
        self._hide_displacement_tooltip()
        super().destroy()


class UnitATable(_DiagnosticUnitTable):
    """Candidate pairs colored by acceptance and better accepted alternatives."""

    columns = (
        ("unit", "Unit A", 45), ("partner", "Best B", 45), ("status", "Pair status", 115),
        ("angle", "Angle (deg)", 70), ("distance", "Dist (um)", 60), ("event_r", "Event r", 60),
        ("displacement", "Disp. consistency", 145),
    )
    table_style = "UnitA.Treeview"
    selection_event = "<<UnitASelected>>"

    def set_options(self, values):
        self._clear_diagnostics()
        for value in values:
            unit, _, partner = value.split()
            self.table.insert("", "end", iid=unit, values=(unit, partner, ""))
            self._set_pair_metrics(unit, int(unit), int(partner))
        self._queue_event_metrics()

    def set_review_status(self, unit, category):
        labels = {
            "accepted": ("Accepted", APPROVED_MATCH_COLOR),
            "better_alternative": (
                "Alternative match accepted",
                BETTER_ALTERNATIVE_COLOR,
            ),
            "no_alternative": ("No better match", NO_ACCEPTED_ALTERNATIVE_COLOR),
        }
        label, color = labels[category]
        self.table.tag_configure(category, foreground=color)
        self.table.item(str(unit), tags=(category,))
        self.table.set(str(unit), "status", label)


class UnitBTable(_DiagnosticUnitTable):
    """All Unit B candidates, with automatic OR/AND threshold eligibility."""

    columns = (
        ("unit", "Unit B", 45), ("cv12", "P12", 45), ("cv21", "P21", 45),
        ("angle", "Angle (deg)", 70), ("distance", "Dist (um)", 60),
        ("event_r", "Event r", 60),
        ("displacement", "Disp. consistency", 145),
    )
    table_style = "UnitB.Treeview"
    selection_event = "<<UnitBSelected>>"

    def set_options(self, values):
        self._clear_diagnostics()
        unit_a = int(entry_a.get()) if values else None
        self.table.tag_configure("above_threshold", foreground=APPROVED_MATCH_COLOR)
        self.table.tag_configure("below_threshold", foreground=NO_ACCEPTED_ALTERNATIVE_COLOR)
        for unit_b in values:
            unit_b = int(unit_b)
            category = (
                "above_threshold"
                if (unit_a, unit_b) in automatic_candidate_pairs
                else "below_threshold"
            )
            self.table.insert(
                "", "end", iid=str(unit_b),
                values=(unit_b, f"{output_GUI[0][unit_a, unit_b]:.3f}", f"{output_GUI[1][unit_a, unit_b]:.3f}"),
                tags=(category,),
            )
            self._set_pair_metrics(str(unit_b), unit_a, unit_b)
        self._queue_event_metrics()

    def current(self, index):
        self.set(self.table.get_children()[index])


def create_unit_b_color_legend(master):
    legend = ttk.Frame(master)
    ttk.Label(
        legend,
        text=f"Unit B ({automatic_match_mode.upper()}, > {match_threshold:g}):",
    ).pack(side="left", padx=(0, 4))
    for text, color in (
        ("eligible", APPROVED_MATCH_COLOR),
        ("below threshold", NO_ACCEPTED_ALTERNATIVE_COLOR),
    ):
        ttk.Label(legend, text=text, foreground=color).pack(side="left", padx=4)
    return legend


def create_unit_b_diagnostic_legend(master):
    return ttk.Button(master, text="Columns and review help", command=open_diagnostic_help)


def open_diagnostic_help():
    global diagnostic_help_window
    if diagnostic_help_window is not None and _widget_exists(diagnostic_help_window):
        diagnostic_help_window.lift()
        return
    diagnostic_help_window = Toplevel(root)
    diagnostic_help_window.title("UnitMatch columns and review help")
    text = (
        "Session summary: accepted pairs between the two selected sessions, plus "
        "total loaded units and uniquely matched units in each session. Percentages "
        "are matched units / loaded units, independent of the display filter. "
        "Automatic accepts plus manual accepts minus rejections are counted once "
        "per pair, not once per CV direction.\n\n"
        "Unit A diagnostics refer to that row's listed Best B. Unit B diagnostics "
        "refer to the currently selected Unit A.\n\n"
        "Pair status: accepted, a better-or-tied accepted alternative exists, or no "
        "better accepted alternative exists. Unit B colors indicate automatic "
        "OR/AND threshold eligibility, not final acceptance.\n\n"
        "P12 / P21: original UM probabilities in each cross-validation direction.\n\n"
        "Angle: degrees from the automatic-match mean displacement. Dist: raw "
        "centroid displacement in micrometers. Dashed map rays remain +/-60 degrees "
        "around the automatic mean; they no longer define the filter.\n\n"
        "Event r: mean shared-event PSTH correlation using the Event Viewer settings. "
        "Functional similarity alone does not establish identity.\n\n"
        "Disp. consistency: 0-100 agreement with the robust displacement reference "
        "of accepted matches, excluding both candidate endpoints. This is not UM "
        "probability and does not change automatic matching. Hover over a value "
        "for residual, normalized distance, reference count and scope. References "
        "are probe-only unless shank IDs were supplied.\n\n"
        "Low consistency filter: show Unit A rows whose listed Best B has a score "
        "strictly below the chosen cutoff (default 20, a review heuristic, not a "
        "calibrated threshold). Unit B alternatives remain available. Pending "
        "and n/a results are not treated as low scores. The filter refreshes after "
        "manual decisions.\n\n"
        "Decisions are preserved in memory on reopening, not automatically saved "
        "to disk. Accepting a new partner replaces competitors within that "
        "session pair, not links to other sessions."
    )
    ttk.Label(
        diagnostic_help_window, text=text, wraplength=600, justify="left", padding=16,
    ).pack(fill="both", expand=True)
    ttk.Button(
        diagnostic_help_window, text="Close", command=diagnostic_help_window.destroy,
    ).pack(pady=(0, 12))


def _get_displacement_context():
    """Snapshot the curated reference once; the worker owns all model fitting."""
    global displacement_context, displacement_context_key
    if raw_avg_centroid is None:
        return None, "Raw pre-drift centroids are unavailable.", ""
    if "probe_numbers" not in clus_info:
        return None, "Probe metadata is unavailable.", ""
    accepted = tuple(sorted(_curated_accepted_pairs(
        automatic_match_pairs, is_match, not_match,
    )))
    key = (id(raw_avg_centroid), id(clus_info), displacement_min_references, accepted)
    if displacement_context_key != key:
        from UnitMatchPy.displacement_consistency import DisplacementConsistency

        shanks = clus_info.get("shank_ids")
        model = DisplacementConsistency(
            raw_avg_centroid, clus_info["session_id"], clus_info["probe_numbers"],
            accepted, shank_ids=shanks, min_references=displacement_min_references,
        )
        scope = "same probe and shank" if shanks is not None else "same probe (no shank metadata)"
        displacement_context = (model, "", scope)
        displacement_context_key = key
    return displacement_context


def _compute_displacement_consistency(pairs, model, cancel, results):
    for row, unit_a, unit_b in pairs:
        if cancel.is_set():
            return
        result = model.score(unit_a, unit_b)
        if cancel.is_set():
            return
        results.put((row, result))


def _refresh_displacement_consistency():
    global displacement_context, displacement_context_key
    displacement_context = None
    displacement_context_key = None
    filtering = _unusual_filter_enabled()
    for name in ("entry_a", "entry_b"):
        table = globals().get(name)
        if isinstance(table, _DiagnosticUnitTable) and _widget_exists(table):
            if filtering:
                table._cancel_displacement_metrics()
            else:
                table.refresh_displacement_metrics()
    if filtering:
        update_unusual_displacement_filter()


def _automatic_displacement_reference(unit_a, unit_b):
    sessions = clus_info["session_id"]
    probes = clus_info.get("probe_numbers")
    if probes is None or probes[unit_a] != probes[unit_b] or sessions[unit_a] == sessions[unit_b]:
        return None
    first, second = sorted((int(sessions[unit_a]), int(sessions[unit_b])))
    group = unusual_displacement_group_info.get((first, second, probes[unit_a]), {})
    if not group.get("mean_defined"):
        return None
    direction = 1 if sessions[unit_a] < sessions[unit_b] else -1
    return np.asarray(group["mean_vector"]) * direction


def _pair_displacement_metrics(unit_a, unit_b):
    if raw_avg_centroid is None:
        return None, None
    probes = clus_info.get("probe_numbers")
    if probes is None or probes[unit_a] != probes[unit_b]:
        return None, None
    vector = np.mean(
        raw_avg_centroid[1:3, unit_b, :] - raw_avg_centroid[1:3, unit_a, :],
        axis=1,
    )
    if not np.all(np.isfinite(vector)):
        return None, None
    magnitude = float(np.linalg.norm(vector))
    reference = _automatic_displacement_reference(unit_a, unit_b)
    if reference is None or magnitude <= np.finfo(float).eps * 32:
        return magnitude, None
    cosine = np.dot(vector, reference) / (magnitude * np.linalg.norm(reference))
    angle = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    return magnitude, angle


def _normalize_event_data(event_data_in, n_sessions):
    if event_data_in is None:
        return None
    if not isinstance(event_data_in, dict):
        raise TypeError("event_data_in must be a dictionary")

    get_spike_times = event_data_in.get("get_spike_times")
    if not callable(get_spike_times):
        raise TypeError("event_data_in['get_spike_times'] must be callable")

    session_events = event_data_in.get("event_times_by_session")
    if session_events is None or len(session_events) != n_sessions:
        raise ValueError(
            "event_times_by_session must contain one event mapping per session"
        )

    normalized_events = []
    for events in session_events:
        if not isinstance(events, dict):
            raise TypeError("Each session's event times must be a dictionary")
        normalized_session = {}
        for event_name, event_times in events.items():
            times = np.asarray(event_times, dtype=float).reshape(-1)
            times = np.sort(times[np.isfinite(times)])
            if times.size:
                normalized_session[str(event_name)] = times
        normalized_events.append(normalized_session)

    session_names = event_data_in.get("session_names")
    if session_names is not None and len(session_names) != n_sessions:
        raise ValueError("session_names must contain one name per session")

    return {
        "event_times_by_session": normalized_events,
        "get_spike_times": get_spike_times,
        "session_names": session_names,
    }


def _normalize_raw_avg_centroid(raw_avg_centroid_in, n_units):
    if raw_avg_centroid_in is None:
        return None
    raw_centroids = np.asarray(raw_avg_centroid_in, dtype=float)
    expected_shape = (3, n_units, 2)
    if raw_centroids.shape != expected_shape:
        raise ValueError(
            f"raw_avg_centroid_in must have shape {expected_shape}, "
            f"got {raw_centroids.shape}"
        )
    return raw_centroids.copy()


def _approved_displacement(
    unit_a,
    unit_b,
    approved_pairs,
    raw_centroids,
    cluster_info,
):
    if raw_centroids is None:
        return {
            "available": False,
            "message": "Raw centroid displacement unavailable",
        }
    if not {"session_id", "probe_numbers"}.issubset(cluster_info):
        return {
            "available": False,
            "message": "Session/probe metadata unavailable",
        }

    session_ids = np.asarray(cluster_info["session_id"])
    probe_numbers = np.asarray(cluster_info["probe_numbers"])
    session_a = session_ids[unit_a]
    session_b = session_ids[unit_b]
    probe_a = probe_numbers[unit_a]
    probe_b = probe_numbers[unit_b]
    if session_a == session_b or probe_a != probe_b:
        return {
            "available": False,
            "message": "Select different sessions on the same probe",
        }

    mean_centroids = np.mean(raw_centroids, axis=2)
    vectors = []
    excluded_nonfinite = 0
    seen_pairs = set()
    for pair in approved_pairs:
        if len(pair) != 2:
            continue
        first, second = map(int, pair)
        if first == second:
            continue
        pair_key = frozenset((first, second))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        first_session = session_ids[first]
        second_session = session_ids[second]
        first_probe = probe_numbers[first]
        second_probe = probe_numbers[second]
        if first_probe != second_probe or first_probe != probe_a:
            continue
        if first_session == session_a and second_session == session_b:
            source, target = first, second
        elif first_session == session_b and second_session == session_a:
            source, target = second, first
        else:
            continue

        source_centroid = mean_centroids[:, source]
        target_centroid = mean_centroids[:, target]
        coordinates = np.array(
            [
                target_centroid[1] - source_centroid[1],
                target_centroid[2] - source_centroid[2],
            ]
        )
        if not np.all(np.isfinite(coordinates)):
            excluded_nonfinite += 1
            continue
        vectors.append(coordinates)

    if not vectors:
        message = "No approved cross-session matches"
        if excluded_nonfinite:
            message = (
                f"No finite approved displacements "
                f"({excluded_nonfinite} excluded)"
            )
        return {
            "available": False,
            "message": message,
            "excluded_nonfinite": excluded_nonfinite,
        }

    mean_vector = np.mean(vectors, axis=0)
    return {
        "available": True,
        "dx": float(mean_vector[0]),
        "dy": float(mean_vector[1]),
        "magnitude": float(np.linalg.norm(mean_vector)),
        "count": len(vectors),
        "excluded_nonfinite": excluded_nonfinite,
        "session_a": session_a,
        "session_b": session_b,
        "probe": probe_a,
    }


def _selected_displacement(
    unit_a,
    unit_b,
    approved_pairs,
    raw_centroids,
    cluster_info,
):
    if raw_centroids is None:
        return {
            "available": False,
            "message": "Raw centroid displacement unavailable",
        }
    if not {"session_id", "probe_numbers"}.issubset(cluster_info):
        return {
            "available": False,
            "message": "Session/probe metadata unavailable",
        }

    session_ids = np.asarray(cluster_info["session_id"])
    probe_numbers = np.asarray(cluster_info["probe_numbers"])
    if (
        session_ids[unit_a] == session_ids[unit_b]
        or probe_numbers[unit_a] != probe_numbers[unit_b]
    ):
        return {
            "available": False,
            "message": "Selected pair is not cross-session on one probe",
        }

    mean_centroids = np.mean(raw_centroids, axis=2)
    source = mean_centroids[:, unit_a]
    target = mean_centroids[:, unit_b]
    vector = np.array([target[1] - source[1], target[2] - source[2]])
    if not np.all(np.isfinite(vector)):
        return {
            "available": False,
            "message": "Selected pair has nonfinite raw centroids",
        }

    selected_key = frozenset((int(unit_a), int(unit_b)))
    approved = any(
        len(pair) == 2
        and frozenset(map(int, pair)) == selected_key
        for pair in approved_pairs
    )
    return {
        "available": True,
        "dx": float(vector[0]),
        "dy": float(vector[1]),
        "magnitude": float(np.linalg.norm(vector)),
        "approved": approved,
    }


def _curated_accepted_pairs(automatic_pairs, manual_pairs, rejected_pairs):
    """Combine automatic and manual acceptances after explicit rejections."""
    rejected = {
        frozenset(map(int, pair))
        for pair in rejected_pairs
        if len(pair) == 2 and int(pair[0]) != int(pair[1])
    }
    accepted = set()
    for pair in list(automatic_pairs) + list(manual_pairs):
        if len(pair) != 2:
            continue
        first, second = map(int, pair)
        pair_key = frozenset((first, second))
        if first != second and pair_key not in rejected:
            accepted.add(pair_key)
    return [tuple(sorted(pair)) for pair in accepted]


def _automatic_displacement_anomalies(
    automatic_pairs,
    raw_centroids,
    cluster_info,
):
    """Find automatic matches exceeding the displacement-angle threshold."""
    result = {
        "flagged_pairs": set(),
        "group_info": {},
        "message": None,
    }
    if raw_centroids is None:
        result["message"] = "Raw centroid displacement unavailable"
        return result
    if not {"session_id", "probe_numbers"}.issubset(cluster_info):
        result["message"] = "Session/probe metadata unavailable"
        return result

    session_ids = np.asarray(cluster_info["session_id"])
    probe_numbers = np.asarray(cluster_info["probe_numbers"])
    mean_centroids = np.mean(raw_centroids, axis=2)
    grouped_vectors = {}
    seen_pairs = set()
    excluded_by_group = {}

    for pair in automatic_pairs:
        if len(pair) != 2:
            continue
        first, second = map(int, pair)
        if first == second:
            continue
        pair_key = frozenset((first, second))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        first_session = int(session_ids[first])
        second_session = int(session_ids[second])
        first_probe = probe_numbers[first]
        second_probe = probe_numbers[second]
        if first_session == second_session or first_probe != second_probe:
            continue
        if first_session < second_session:
            source, target = first, second
            session_a, session_b = first_session, second_session
        else:
            source, target = second, first
            session_a, session_b = second_session, first_session
        group_key = (session_a, session_b, first_probe.item() if hasattr(first_probe, "item") else first_probe)

        source_centroid = mean_centroids[:, source]
        target_centroid = mean_centroids[:, target]
        vector = np.array(
            [
                target_centroid[1] - source_centroid[1],
                target_centroid[2] - source_centroid[2],
            ],
            dtype=float,
        )
        exclusions = excluded_by_group.setdefault(
            group_key,
            {"nonfinite": 0, "zero": 0},
        )
        if not np.all(np.isfinite(vector)):
            exclusions["nonfinite"] += 1
            continue

        scale = max(1.0, float(np.max(np.abs(vector))))
        vector_epsilon = np.finfo(float).eps * scale * 32
        if np.linalg.norm(vector) <= vector_epsilon:
            exclusions["zero"] += 1
            continue
        grouped_vectors.setdefault(group_key, []).append(
            (source, target, vector)
        )

    all_group_keys = set(grouped_vectors) | set(excluded_by_group)
    for group_key in all_group_keys:
        vectors = grouped_vectors.get(group_key, [])
        exclusions = excluded_by_group.get(
            group_key,
            {"nonfinite": 0, "zero": 0},
        )
        group_info = {
            "valid_count": len(vectors),
            "excluded_nonfinite": exclusions["nonfinite"],
            "excluded_zero": exclusions["zero"],
            "mean_defined": False,
            "flagged_count": 0,
        }
        result["group_info"][group_key] = group_info
        if not vectors:
            continue

        vector_values = np.vstack([vector for _, _, vector in vectors])
        mean_vector = np.mean(vector_values, axis=0)
        reference_scale = max(
            1.0,
            float(np.max(np.linalg.norm(vector_values, axis=1))),
        )
        mean_epsilon = np.finfo(float).eps * reference_scale * len(vectors) * 32
        mean_norm = float(np.linalg.norm(mean_vector))
        if mean_norm <= mean_epsilon:
            continue

        group_info["mean_defined"] = True
        group_info["mean_vector"] = mean_vector
        for source, target, vector in vectors:
            dot_product = float(np.dot(vector, mean_vector))
            dot_epsilon = (
                np.finfo(float).eps
                * float(np.linalg.norm(vector))
                * mean_norm
                * 32
            )
            angle_boundary = (
                np.cos(np.deg2rad(DISPLACEMENT_ANGLE_THRESHOLD))
                * float(np.linalg.norm(vector))
                * mean_norm
            )
            if dot_product < angle_boundary - dot_epsilon:
                result["flagged_pairs"].add((source, target))
                result["flagged_pairs"].add((target, source))
                group_info["flagged_count"] += 1

    return result


def _compute_event_psth(
    spike_times,
    event_times,
    *,
    before_s=1.0,
    after_s=2.0,
    bin_size_s=0.01,
):
    if before_s <= 0 or after_s <= 0:
        raise ValueError("PSTH windows must be positive")
    if bin_size_s <= 0:
        raise ValueError("PSTH bin size must be positive")

    spike_times = np.sort(np.asarray(spike_times, dtype=float).reshape(-1))
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    event_times = event_times[np.isfinite(event_times)]
    bin_edges = np.arange(-before_s, after_s, bin_size_s)
    bin_edges = np.append(bin_edges, after_s)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    counts = np.zeros(bin_centers.size, dtype=float)
    if event_times.size == 0:
        return bin_centers, counts, 0

    starts = np.searchsorted(spike_times, event_times - before_s)
    stops = np.searchsorted(spike_times, event_times + after_s)
    for event_time, start, stop in zip(event_times, starts, stops):
        relative_times = spike_times[start:stop] - event_time
        counts += np.histogram(relative_times, bins=bin_edges)[0]

    firing_rate_hz = counts / (event_times.size * bin_widths)
    return bin_centers, firing_rate_hz, event_times.size


def _widget_exists(widget):
    try:
        return bool(widget.winfo_exists())
    except TclError:
        return False


def precalculate_all_acgs(
    clus_info, param, save_path=None, bin_size=0.001, max_lag=0.05
):
    """
    Pre-calculate and save autocorrelograms for all units to speed up GUI loading

    Parameters
    ----------
    clus_info : dict
        Cluster information dictionary containing unit metadata
    param : dict
        Parameters dictionary containing KS_dirs and other settings
    save_path : str, optional
        Path to save the ACG data. If None, saves in current directory as 'acg_cache.pkl'
    bin_size : float
        Bin size in seconds (default 1ms)
    max_lag : float
        Maximum lag in seconds (default 50ms)

    Returns
    -------
    acg_cache : dict
        Dictionary mapping unit_id to (autocorr, bin_centers) tuples
    """
    print("Pre-calculating autocorrelograms for all units...")

    if save_path is None:
        save_path = "acg_cache.pkl"

    acg_cache = {}
    n_units = len(clus_info.get("session_id", []))

    for unit_id in range(n_units):
        try:
            # Get spike times for this unit
            spike_times = get_spike_times_for_unit_precalc(unit_id, clus_info, param)

            if len(spike_times) > 1:
                # Compute ACG
                autocorr, bin_centers = compute_acg_precalc(
                    spike_times, bin_size, max_lag
                )
                acg_cache[unit_id] = (autocorr, bin_centers)

                if (unit_id + 1) % 50 == 0:  # Progress update every 50 units
                    print(f"Processed {unit_id + 1}/{n_units} units")
            else:
                acg_cache[unit_id] = (np.array([]), np.array([]))

        except Exception as e:
            print(f"Error computing ACG for unit {unit_id}: {e}")
            acg_cache[unit_id] = (np.array([]), np.array([]))

    # Save to file
    try:
        with open(save_path, "wb") as f:
            pickle.dump(acg_cache, f)
        print(f"ACG cache saved to {save_path}")
    except Exception as e:
        print(f"Error saving ACG cache: {e}")

    return acg_cache


def get_spike_times_for_unit_precalc(unit_id, clus_info, param):
    """
    Get spike times for a specific unit (used in pre-calculation)

    Parameters
    ----------
    unit_id : int
        UnitMatch unit ID
    clus_info : dict
        Cluster information dictionary
    param : dict
        Parameters dictionary

    Returns
    -------
    spike_times : array
        Spike times in seconds for the unit
    """
    try:
        # Get session information for this unit
        session_id = clus_info["session_id"][unit_id]
        original_id = clus_info["original_ids"][unit_id]

        # Get the Kilosort directory for this session
        if "KS_dirs" in param:
            ks_dir = param["KS_dirs"][session_id]
        else:
            return np.array([])

        # Load Kilosort spike times and cluster assignments
        spike_times_path = os.path.join(ks_dir, "spike_times.npy")
        spike_clusters_path = os.path.join(ks_dir, "spike_clusters.npy")

        if os.path.exists(spike_times_path) and os.path.exists(spike_clusters_path):
            # Load spike times (in samples) and cluster assignments
            spike_times_samples = np.load(spike_times_path).flatten()
            spike_clusters = np.load(spike_clusters_path).flatten()

            # Get sampling rate - try to load from params.py or use default
            try:
                params_path = os.path.join(ks_dir, "params.py")
                sample_rate = 30000  # Default sample rate
                if os.path.exists(params_path):
                    # Parse params.py for sample_rate
                    with open(params_path, "r") as f:
                        params_content = f.read()
                    # Extract sample_rate
                    for line in params_content.split("\n"):
                        if "sample_rate" in line and "=" in line:
                            sample_rate = float(line.split("=")[1].strip())
                            break
            except:
                sample_rate = 30000  # Fallback

            # Get spike times for this specific unit
            unit_mask = spike_clusters == original_id
            unit_spike_times = (
                spike_times_samples[unit_mask] / sample_rate
            )  # Convert to seconds

            return unit_spike_times
        else:
            return np.array([])

    except Exception:
        return np.array([])


def compute_acg_precalc(spike_times, bin_size=0.001, max_lag=0.05):
    """
    Compute autocorrelogram for a unit (used in pre-calculation)

    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    bin_size : float
        Bin size in seconds (default 1ms)
    max_lag : float
        Maximum lag in seconds (default 50ms)

    Returns
    -------
    autocorr : array
        Autocorrelogram counts
    bin_centers : array
        Bin centers in seconds
    """
    if len(spike_times) < 2:
        return np.array([]), np.array([])

    # Create bins centered around 0
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    # Calculate autocorrelogram
    autocorr = np.zeros(len(bin_centers))

    # For each spike, find all other spikes within max_lag
    for i, spike_time in enumerate(spike_times):
        # Find spikes within the lag window
        time_diffs = spike_times - spike_time
        valid_diffs = time_diffs[(np.abs(time_diffs) <= max_lag) & (time_diffs != 0)]

        # Bin the time differences
        hist, _ = np.histogram(valid_diffs, bins)
        autocorr += hist

    # Convert to firing rate (Hz)
    total_time = spike_times[-1] - spike_times[0] if len(spike_times) > 1 else 1
    n_spikes = len(spike_times)
    autocorr = autocorr / (bin_size * n_spikes * total_time)

    return autocorr, bin_centers


def load_acg_cache(cache_path="acg_cache.pkl"):
    """
    Load pre-calculated ACG cache from file

    Parameters
    ----------
    cache_path : str
        Path to the ACG cache file

    Returns
    -------
    acg_cache : dict or None
        Dictionary mapping unit_id to (autocorr, bin_centers) tuples, or None if loading failed
    """
    try:
        with open(cache_path, "rb") as f:
            acg_cache = pickle.load(f)
        print(f"ACG cache loaded from {cache_path}")
        return acg_cache
    except Exception as e:
        print(f"Could not load ACG cache from {cache_path}: {e}")
        return None


def run_GUI(*, preserve_decisions=True, block=True):
    """
    Open the review GUI and return its manual accept/reject lists.

    Set ``block=False`` in an IPython notebook to keep the kernel available
    while Tk processes events. Manual decisions are retained in memory by default.
    After closing the window, pass ``preserve_decisions=False`` for a fresh review.

    Returns
    -------
    is_match : list
        Manually accepted pairs, updated in place during review.
    not_match : list
        Manually rejected pairs, updated in place during review.

    Apply these labels to your automatic matches after reviewing. With
    ``block=False``, the returned lists remain live, not copies.
    """
    global CV_tkinter
    global root
    global entry_a
    global entry_b
    global session_entry_a
    global session_entry_b
    global match_idx
    global frame_table
    global score_table
    global avg_waveform_plot
    global trajectory_plot
    global bayes_label
    global original_id_label
    global raw_waveform_plot
    global hist_plot
    global is_match
    global not_match
    global option_a
    global option_b
    global entry_frame
    global toggle_raw_val
    global toggle_UM_score_val
    global unit_legend_plot
    global hist_legend_plot
    global acg_plot
    global toggle_acg_val
    global acg_cache
    global gui_scale
    global event_view_refresh
    global event_view_window
    global toggle_unusual_displacement_val
    global unusual_displacement_status_label
    global review_info_frame
    global displacement_context, displacement_context_key
    global consistency_threshold_var, match_button, non_match_button
    global diagnostic_help_window, histogram_panel, raw_waveform_view_key

    existing_root = globals().get("root")
    if existing_root is not None and _widget_exists(existing_root):
        existing_root.deiconify()
        existing_root.lift()
        return is_match, not_match

    shell = None
    if not block:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            raise RuntimeError("Nonblocking review requires an IPython event loop")

    # Try to load pre-calculated ACG cache
    acg_cache = load_acg_cache()
    event_view_refresh = None
    event_view_window = None

    np.set_printoptions(suppress=True)
    if not preserve_decisions:
        is_match = []
        not_match = []
    displacement_context = None
    displacement_context_key = None
    diagnostic_help_window = None
    histogram_panel = None
    raw_waveform_view_key = None
    root = Tk()
    previous_colors = {
        key: rcParams[key]
        for key in ("text.color", "axes.labelcolor", "xtick.color", "ytick.color")
    }

    def close_gui():
        global event_metric_executor, displacement_metric_executor
        global pending_redraw
        _cancel_consistency_filter()
        if pending_redraw is not None:
            root.after_cancel(pending_redraw)
            pending_redraw = None
        for table in (entry_a, entry_b):
            table._cancel_event_metrics()
            table._cancel_displacement_metrics()
        if event_metric_executor is not None:
            event_metric_executor.shutdown(wait=False, cancel_futures=True)
            event_metric_executor = None
        if displacement_metric_executor is not None:
            displacement_metric_executor.shutdown(wait=False, cancel_futures=True)
            displacement_metric_executor = None
        rcParams.update(previous_colors)
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_gui)
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    gui_scale = _get_gui_scale(screen_width, screen_height)

    rcParams.update({"figure.autolayout": True})
    rcParams.update({"font.size": _scaled_font_size(14)})
    rcParams.update({"font.family": "DejaVu Sans"})
    color = "white"
    rcParams["text.color"] = color
    rcParams["axes.labelcolor"] = color
    rcParams["xtick.color"] = color
    rcParams["ytick.color"] = color

    # Set window size to fit the screen with some padding
    window_width = screen_width - 100
    window_height = screen_height - 100
    root.geometry(f"{window_width}x{window_height}+50+50")

    # Configure column weights to prevent plot squashing
    for column, weight in enumerate((3, 2, 3, 4)):
        root.columnconfigure(column, weight=weight)
    for row, weight in enumerate((0, 0, 0, 2, 1, 1)):
        root.rowconfigure(row, weight=weight)

    # downloaded theme from https://sourceforge.net/projects/tcl-awthemes/
    theme_path_rel = os.path.join("TkinterTheme", "awthemes-10.4.0")
    theme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), theme_path_rel
    )

    root.tk.call("lappend", "auto_path", theme_path)
    root.tk.call("package", "require", "awdark")
    s = ttk.Style(root)
    s.theme_use("awdark")

    # Configure fonts for all TTK widgets to use DejaVu Sans
    control_font_size = _scaled_font_size(12)
    entry_font_size = _scaled_font_size(10)
    s.configure(".", font=("DejaVu Sans", control_font_size))
    s.configure("TLabel", font=("DejaVu Sans", control_font_size))
    s.configure("TButton", font=("DejaVu Sans", control_font_size))
    s.configure("TEntry", font=("DejaVu Sans", entry_font_size))
    s.configure("TCombobox", font=("DejaVu Sans", entry_font_size))
    s.configure("TCheckbutton", font=("DejaVu Sans", control_font_size))
    s.configure("TRadiobutton", font=("DejaVu Sans", control_font_size))
    s.configure(
        "TLabelFrame.Label",
        font=("DejaVu Sans", control_font_size, "bold"),
    )

    root.title("UMPy - Manual Curation")
    # root.geometry('800x800')

    # Construct the file path to the icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUI_icon.png")

    # Debugging print statements
    print(f"Icon path: {icon_path}")
    print(f"File exists: {os.path.exists(icon_path)}")

    # Load the icon
    try:
        icon = PhotoImage(master=root, file=icon_path)
        # Set the icon photo
        root.iconphoto(False, icon)
    except Exception as e:
        print(f"Error loading icon: {e}")

    background = ttk.Frame(root)
    background.place(x=0, y=0, relwidth=1.0, relheight=1.0)

    frame_table = ttk.LabelFrame(root)
    score_table = ttk.LabelFrame(root)
    avg_waveform_plot = Canvas(root)
    trajectory_plot = Canvas(root)
    bayes_label = ttk.Label(root)
    original_id_label = ttk.Label(root)
    raw_waveform_plot = Canvas(root)
    hist_plot = Canvas(root)
    acg_plot = Canvas(root)

    # Unit entry
    ######################################################################################
    # Keep unit colors consistent across every GUI panel.
    entry_frame = ttk.LabelFrame(root, text="Select Units")
    review_info_frame = ttk.Frame(entry_frame)
    review_info_frame.grid(
        row=3, column=3, columnspan=2, rowspan=6, sticky="new", padx=5,
    )
    label_a = ttk.Label(entry_frame, text="Unit A")
    label_a.configure(foreground=UNIT_A_COLOR)
    label_b = ttk.Label(entry_frame, text="Unit B")
    label_b.configure(foreground=UNIT_B_COLOR)

    # select the session
    sessions_list = np.arange(1, param["n_sessions"] + 1).tolist()
    session_entry_a = ttk.Combobox(entry_frame, value=sessions_list, width=2)
    session_entry_b = ttk.Combobox(entry_frame, value=sessions_list, width=2)
    session_entry_a.set(1)  # Start wiht session 1,2 if more than 1 session given
    if len(sessions_list) == 1:
        session_entry_b.set(1)
    else:
        session_entry_b.set(2)
    label_session_a = ttk.Label(entry_frame, text="Session No.")
    label_session_b = ttk.Label(entry_frame, text="Session No.")

    # select CV
    CV_options = [("Avg", 0), ("(1,2)", 1), ("(2,1)", 2)]
    CV_tkinter = IntVar(master=root)
    CV_tkinter.set(0)
    label_cv = ttk.Label(entry_frame, text="CV")
    for i, option in enumerate(CV_options):
        RadioCV = ttk.Radiobutton(
            entry_frame,
            text=option[0],
            value=option[1],
            variable=CV_tkinter,
            command=update_unit_cv,
        ).grid(row=i + 1, column=0)

    toggle_unusual_displacement_val = BooleanVar(master=root, value=False)
    consistency_threshold_var = DoubleVar(master=root, value=consistency_filter_threshold)
    filter_controls = ttk.Frame(entry_frame)
    unusual_displacement_toggle = ttk.Checkbutton(
        filter_controls,
        text="Low displacement consistency <",
        variable=toggle_unusual_displacement_val,
        command=update_unusual_displacement_filter,
    )
    unusual_displacement_toggle.pack(side="left")
    cutoff_entry = ttk.Spinbox(
        filter_controls, from_=0, to=100, increment=5, width=5,
        textvariable=consistency_threshold_var, command=update_unusual_displacement_filter,
    )
    cutoff_entry.pack(side="left", padx=4)
    cutoff_entry.bind("<Return>", lambda event: update_unusual_displacement_filter())
    cutoff_entry.bind("<FocusOut>", lambda event: update_unusual_displacement_filter())
    unusual_displacement_status_label = ttk.Label(
        entry_frame,
        text=(
            "Filter off"
        ),
        wraplength=round(360 * gui_scale),
    )
    unit_a_color_legend = ttk.Frame(entry_frame)
    ttk.Label(unit_a_color_legend, text="Unit A list:").pack(
        side="left",
        padx=(0, 4),
    )
    for text, color in (
        ("accepted", APPROVED_MATCH_COLOR),
        ("Alternative match accepted", BETTER_ALTERNATIVE_COLOR),
        ("no better match", NO_ACCEPTED_ALTERNATIVE_COLOR),
    ):
        ttk.Label(
            unit_a_color_legend,
            text=text,
            foreground=color,
            wraplength=round(100 * gui_scale),
        ).pack(side="left", padx=4)

    unit_b_color_legend = create_unit_b_color_legend(entry_frame)
    unit_b_diagnostic_legend = create_unit_b_diagnostic_legend(entry_frame)

    # selecting the unit
    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = UnitATable(
        entry_frame, values=get_unit_a_display_options()
    )
    entry_a.set(option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    entry_b.current(0)
    entry_a.bind("<<UnitASelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()

    session_entry_a.bind("<<ComboboxSelected>>", update_unit_entryA)
    session_entry_b.bind("<<ComboboxSelected>>", update_unit_entryB)

    # adding a button which swaps unit A and B
    swap_button = ttk.Button(entry_frame, text="Swap Units", command=swap_units)

    score_histogram_cache.clear()

    # place the widgets on the EntryFrame
    label_cv.grid(row=0, column=0)
    label_a.grid(row=0, column=1)
    label_b.grid(row=0, column=3)
    label_session_a.grid(row=1, column=1)
    session_entry_a.grid(row=1, column=2, padx=15)
    label_session_b.grid(row=1, column=3)
    session_entry_b.grid(row=1, column=4, padx=15)
    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)
    filter_controls.grid(
        row=3,
        column=1,
        columnspan=2,
        sticky="W",
        padx=5,
    )
    unusual_displacement_status_label.grid(
        row=4,
        column=1,
        columnspan=2,
        sticky="W",
        padx=5,
    )
    unit_b_diagnostic_legend.grid(row=5, column=1, columnspan=2, sticky="WE", padx=5)
    swap_button.grid(row=6, column=1, columnspan=2, sticky="WE")
    ######################################################################################

    # MatchButtons
    ######################################################################################
    match_controls = ttk.Frame(root)
    match_button = Button(
        match_controls, text="Set as Match", command=set_match,
        background="#2E7D32", foreground="white",
        activebackground="#388E3C", activeforeground="white",
        font=("DejaVu Sans", control_font_size, "bold"),
        relief="flat", padx=12, pady=5,
    )
    match_controls.columnconfigure(1, weight=1)
    match_button.grid(row=0, column=0, sticky="w")
    _create_session_summary(match_controls).grid(
        row=0, column=1, sticky="w", padx=(12, 0),
    )
    non_match_button = Button(
        root, text="Set as Non Match", command=set_not_match,
        background="#C62828", foreground="white",
        activebackground="#D32F2F", activeforeground="white",
        font=("DejaVu Sans", control_font_size, "bold"),
        relief="flat", padx=12, pady=5,
    )
    pair_lookup_button = ttk.Button(
        root,
        text="Inspect Pair",
        command=open_pair_lookup,
    )
    event_view_button = ttk.Button(
        root,
        text="Event Viewer",
        command=open_event_view,
        state="normal" if event_data is not None else "disabled",
    )

    # Toggle Plots
    ######################################################################################
    toggle_raw_val = BooleanVar(master=root)
    toggle_UM_score_val = BooleanVar(master=root)
    toggle_acg_val = BooleanVar(master=root)
    toggle_raw_val.set(False)
    toggle_UM_score_val.set(False)
    toggle_acg_val.set(False)

    # Set up Key-Board shortcuts
    root.bind_all("u", update)
    root.bind_all("<Return>", update)
    root.bind_all("q", set_match)
    root.bind_all("m", set_match)
    root.bind_all("e", set_not_match)
    root.bind_all("n", set_not_match)

    # Grid the units
    entry_frame.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky="nsew")
    match_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    non_match_button.grid(row=1, column=1, sticky="W", padx=10, pady=5)
    pair_lookup_button.grid(row=1, column=2, sticky="W", padx=10, pady=5)
    event_view_button.grid(row=1, column=3, sticky="W", padx=10, pady=5)

    # Create visual legend plots outside plots
    global unit_legend_plot
    global hist_legend_plot
    create_unit_legend()
    create_hist_legend()

    unit_legend_plot.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5)
    hist_legend_plot.grid(row=0, column=2, columnspan=2, sticky="ew", padx=5)

    # Configure grid weights to auto-adjust subpanels
    entry_frame.grid_columnconfigure(1, weight=0)
    entry_frame.grid_columnconfigure(3, weight=1)
    entry_frame.grid_rowconfigure(2, weight=1)
    entry_frame.update_idletasks()
    selection_min_width = entry_frame.winfo_reqwidth() + 10
    root.columnconfigure(0, minsize=selection_min_width * 3 // 5)
    root.columnconfigure(1, minsize=selection_min_width - selection_min_width * 3 // 5)

    update(None)
    match_idx = 0

    if block:
        root.mainloop()
    else:
        shell.enable_gui("tk")

    return is_match, not_match


def process_info_for_GUI(
    output,
    match_threshold_in,
    scores_to_include,
    total_score,
    amplitude_in,
    spatial_decay_in,
    avg_centroid_in,
    avg_waveform_in,
    avg_waveform_per_tp_in,
    wave_idx_in,
    max_site_in,
    max_site_mean_in,
    waveform_in,
    within_session_in,
    channel_pos_in,
    clus_info_in,
    param_in,
    automatic_matches_in=None,
    match_mode="or",
    review_threshold_margin=0.1,
    raw_output_in=None,
    event_data_in=None,
    raw_avg_centroid_in=None,
    displacement_min_references_in=5,
):
    """
    This function:
    1 - passes data to the GUI
    2 - processes the data so it is in a better form for the GUI

    ``displacement_min_references_in`` controls the post-hoc consistency
    diagnostic only (default 20 accepted pairs after endpoint exclusions).
    Optional per-unit ``clus_info_in["shank_ids"]`` further scopes its reference.
    """
    global matches_avg
    global matches_GUI
    global output_avg
    global output_GUI
    global scores_to_include_avg
    global scores_to_include_GUI
    global amplitude
    global amplitude_avg
    global spatial_decay
    global spatial_decay_avg
    global avg_centroid
    global avg_centroid_avg
    global avg_waveform
    global avg_waveform_avg
    global avg_waveform_per_tp
    global avg_waveform_per_tp_avg
    global wave_idx
    global max_site
    global max_site_mean
    global waveform
    global clus_info
    global session_switch
    global param
    global match_threshold
    global within_session
    global channel_pos
    global review_matches
    global automatic_match_pairs
    global automatic_candidate_pairs
    global automatic_match_mode
    global review_threshold
    global raw_output
    global event_data
    global event_psth_cache
    global event_spike_times_cache
    global raw_avg_centroid
    global unusual_displacement_pairs
    global unusual_displacement_group_info
    global event_correlation_cache
    global displacement_min_references, displacement_context, displacement_context_key
    global histogram_panel, raw_waveform_view_key
    if (
        isinstance(displacement_min_references_in, (bool, np.bool_))
        or not isinstance(displacement_min_references_in, (int, np.integer))
        or displacement_min_references_in < 3
    ):
        raise ValueError("displacement_min_references_in must be an integer of at least 3")
    displacement_min_references = int(displacement_min_references_in)
    _cancel_consistency_filter()
    histogram_panel = None
    raw_waveform_view_key = None
    displacement_context = None
    displacement_context_key = None
    for name in ("entry_a", "entry_b"):
        table = globals().get(name)
        if isinstance(table, _DiagnosticUnitTable) and _widget_exists(table):
            table._cancel_displacement_metrics()
    amplitude = amplitude_in
    spatial_decay = spatial_decay_in
    avg_centroid = avg_centroid_in
    avg_waveform = avg_waveform_in
    avg_waveform_per_tp = avg_waveform_per_tp_in
    wave_idx = wave_idx_in
    max_site = max_site_in
    max_site_mean = max_site_mean_in
    waveform = waveform_in
    clus_info = clus_info_in
    session_switch = clus_info["session_switch"]
    if "session_indices" not in clus_info:
        clus_info["session_indices"] = np.searchsorted(
            np.asarray(session_switch)[1:],
            np.arange(len(clus_info["original_ids"])),
            side="right",
        )
    param = param_in
    match_threshold = match_threshold_in
    within_session = within_session_in
    channel_pos = channel_pos_in
    automatic_match_mode = match_mode.lower()
    if automatic_match_mode not in {"and", "or"}:
        raise ValueError("match_mode must be 'and' or 'or'")
    if review_threshold_margin < 0:
        raise ValueError("review_threshold_margin must be non-negative")
    review_threshold = max(0, match_threshold - review_threshold_margin)
    raw_output = output if raw_output_in is None else np.asarray(raw_output_in)
    if raw_output.shape != output.shape:
        raise ValueError("raw_output_in must have the same shape as output")
    event_data = _normalize_event_data(event_data_in, param["n_sessions"])
    raw_avg_centroid = _normalize_raw_avg_centroid(
        raw_avg_centroid_in,
        len(clus_info["session_id"]),
    )
    event_psth_cache = {}
    event_spike_times_cache = {}
    event_correlation_cache = {}
    score_histogram_cache.clear()

    # Matrix [A, B] compares Unit A half 1 with Unit B half 2.
    # Its transpose is the reciprocal comparison, A half 2 with B half 1.
    output_GUI = [output, output.T]
    cv_12_matches = output > match_threshold
    cv_21_matches = cv_12_matches.T
    matches_GUI = [
        np.argwhere(cv_12_matches),
        np.argwhere(cv_21_matches),
    ]
    scores_to_include_GUI = [
        scores_to_include,
        {key: value.T for key, value in scores_to_include.items()},
    ]

    scores_to_include_avg = {}
    for key, value in scores_to_include.items():
        scores_to_include_avg[key] = (value + value.T) / 2

    output_avg = (output_GUI[0] + output_GUI[1]) / 2
    if automatic_match_mode == "or":
        automatic_candidate_mask = cv_12_matches | cv_21_matches
    else:
        automatic_candidate_mask = cv_12_matches & cv_21_matches
    automatic_candidate_pairs = set(
        map(tuple, np.argwhere(automatic_candidate_mask))
    )
    if automatic_matches_in is None:
        automatic_matches_in = np.argwhere(automatic_candidate_mask)

    automatic_matches_in = np.asarray(automatic_matches_in, dtype=int).reshape(-1, 2)
    automatic_match_pairs = set()
    for unit_a, unit_b in automatic_matches_in:
        automatic_match_pairs.add((int(unit_a), int(unit_b)))
        automatic_match_pairs.add((int(unit_b), int(unit_a)))
    anomaly_result = _automatic_displacement_anomalies(
        automatic_match_pairs,
        raw_avg_centroid,
        clus_info,
    )
    unusual_displacement_pairs = anomaly_result["flagged_pairs"]
    unusual_displacement_group_info = anomaly_result["group_info"]
    review_matches = np.argwhere(
        (output_GUI[0] >= review_threshold)
        | (output_GUI[1] >= review_threshold)
    )
    review_match_scores = output_avg[
        review_matches[:, 0],
        review_matches[:, 1],
    ]
    review_matches = review_matches[
        np.argsort(-review_match_scores, kind="stable")
    ]

    # output_avg is symmetric; argwhere already returns unique, sorted pairs.
    matches_avg = np.argwhere(output_avg > match_threshold)

    # or an simply average over both CV
    amplitude_avg = np.mean(amplitude, axis=-1)
    spatial_decay_avg = np.mean(spatial_decay, axis=-1)
    avg_centroid_avg = np.mean(avg_centroid, axis=-1)
    avg_waveform_avg = np.mean(avg_waveform, axis=-1)
    avg_waveform_per_tp_avg = np.mean(avg_waveform_per_tp, axis=-1)
    _refresh_session_summary()


def _session_pair_summary(accepted_pairs, switches, session_a, session_b, *, unit_mask=None):
    """Count unique accepted links and matched units in the selected sessions."""
    if not all(1 <= session < len(switches) for session in (session_a, session_b)):
        raise ValueError("Session numbers must identify loaded review sessions")
    if session_a == session_b:
        raise ValueError("Select two different sessions for a cross-session summary")
    start_a, stop_a = switches[session_a - 1:session_a + 1]
    start_b, stop_b = switches[session_b - 1:session_b + 1]
    if unit_mask is None:
        unit_mask = np.ones(switches[-1], dtype=bool)
    else:
        unit_mask = np.asarray(unit_mask, dtype=bool)
        if unit_mask.shape != (switches[-1],):
            raise ValueError("Unit mask must contain one value per loaded review unit")
    pairs = set()
    for first, second in accepted_pairs:
        if start_a <= first < stop_a and start_b <= second < stop_b:
            if unit_mask[first] and unit_mask[second]:
                pairs.add((int(first), int(second)))
        elif start_a <= second < stop_a and start_b <= first < stop_b:
            if unit_mask[first] and unit_mask[second]:
                pairs.add((int(second), int(first)))
    units_a = int(np.count_nonzero(unit_mask[start_a:stop_a]))
    units_b = int(np.count_nonzero(unit_mask[start_b:stop_b]))
    matched_a = len({first for first, _ in pairs})
    matched_b = len({second for _, second in pairs})
    return {
        "accepted_pairs": len(pairs),
        "units_a": units_a,
        "units_b": units_b,
        "matched_a": matched_a,
        "matched_b": matched_b,
        "percent_a": 100 * matched_a / units_a if units_a else None,
        "percent_b": 100 * matched_b / units_b if units_b else None,
    }


def _summary_metadata_text(value):
    if pd.isna(value):
        return "?"
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def _session_group_summaries(accepted_pairs, switches, session_a, session_b, probes, shanks=None):
    accepted_pairs = tuple(accepted_pairs)
    metadata = {"probe": probes}
    if shanks is not None:
        metadata["shank"] = shanks
    for name, values in metadata.items():
        if np.asarray(values).shape != (switches[-1],):
            raise ValueError(f"{name} metadata must contain one value per loaded review unit")
    units = pd.DataFrame(metadata)
    group_columns = "probe" if shanks is None else ["probe", "shank"]
    summaries = []
    for key, group in units.groupby(group_columns, sort=True, dropna=False):
        probe, shank = (key, None) if shanks is None else key
        mask = np.zeros(switches[-1], dtype=bool)
        mask[group.index] = True
        summary = _session_pair_summary(
            accepted_pairs, switches, session_a, session_b, unit_mask=mask,
        )
        if not summary["units_a"] and not summary["units_b"]:
            continue
        label = f"Probe {_summary_metadata_text(probe)}"
        if shanks is not None:
            label += f" / shank {_summary_metadata_text(shank)}"
        summaries.append((label, summary))
    return summaries


def _create_group_summary(master):
    global session_group_summary
    frame = ttk.Frame(master)
    font = ("DejaVu Sans", _scaled_font_size(8))
    style = ttk.Style(master)
    style.configure("Summary.Treeview", font=font, rowheight=_scaled_font_size(8) + 8)
    style.configure("Summary.Treeview.Heading", font=font)
    tree = ttk.Treeview(
        frame, columns=("group", "pairs", "a", "b"), show="headings",
        height=2, selectmode="none", takefocus=False, style="Summary.Treeview",
    )
    for column, width, heading in (
        ("group", 112, "Probe / shank"), ("pairs", 40, "Pairs"),
        ("a", 112, "S1 matched/total"), ("b", 112, "S2 matched/total"),
    ):
        tree.column(column, width=width, minwidth=width, stretch=False, anchor="w")
        tree.heading(column, text=heading)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    session_group_summary = tree
    return frame


def _set_group_summary_rows(rows, session_a, session_b):
    tree = session_group_summary
    if tree is None or not _widget_exists(tree):
        return
    content = (session_a, session_b, rows)
    if getattr(tree, "_summary_content", None) == content:
        return
    tree.heading("a", text=f"S{session_a} matched/total")
    tree.heading("b", text=f"S{session_b} matched/total")
    previous = set(tree.get_children())
    for index, values in enumerate(rows):
        row_id = f"group-{index}"
        if row_id in previous:
            tree.item(row_id, values=values)
            previous.remove(row_id)
        else:
            tree.insert("", "end", iid=row_id, values=values)
    for row_id in previous:
        tree.delete(row_id)
    tree._summary_content = content


def _refresh_group_summary(accepted_pairs, overall, session_a, session_b):
    if session_group_summary is None or not _widget_exists(session_group_summary):
        return
    probes = clus_info.get("probe_numbers")
    if probes is None:
        _set_group_summary_rows([("Probes unknown", "", "", "")], session_a, session_b)
        return
    groups = _session_group_summaries(
        accepted_pairs, session_switch, session_a, session_b,
        probes, clus_info.get("shank_ids"),
    )
    rows = []
    for label, summary in groups:
        values = [label, summary["accepted_pairs"]]
        for side in ("a", "b"):
            percent = summary[f"percent_{side}"]
            percentage = "n/a" if percent is None else f"{percent:.1f}%"
            values.append(
                f"{summary[f'matched_{side}']}/{summary[f'units_{side}']} ({percentage})"
            )
        rows.append(tuple(values))
    cross_group_pairs = overall["accepted_pairs"] - sum(
        summary["accepted_pairs"] for _, summary in groups
    )
    if cross_group_pairs:
        rows.append(("Across groups", cross_group_pairs, "See overall", "See overall"))
    _set_group_summary_rows(rows, session_a, session_b)


def _create_session_summary(master):
    global session_summary_label
    session_summary_label = ttk.Label(
        master, justify="left", font=("DejaVu Sans", _scaled_font_size(9)),
        wraplength=round(350 * gui_scale),
    )
    _create_group_summary(master).grid(row=0, column=2, sticky="w", padx=(12, 0))
    _refresh_session_summary()
    return session_summary_label


def _refresh_session_summary(accepted_pairs=None):
    if session_summary_label is None or not _widget_exists(session_summary_label):
        return
    session_a, session_b = int(session_entry_a.get()), int(session_entry_b.get())
    if session_a == session_b:
        session_summary_label.configure(text="Summary: select two different sessions")
        _set_group_summary_rows([], session_a, session_b)
        return
    if accepted_pairs is None:
        accepted_pairs = _curated_accepted_pairs(automatic_match_pairs, is_match, not_match)
    summary = _session_pair_summary(accepted_pairs, session_switch, session_a, session_b)
    lines = [f"Accepted pairs (S{session_a} / S{session_b}): {summary['accepted_pairs']}"]
    for side, session in (("a", session_a), ("b", session_b)):
        percent = summary[f"percent_{side}"]
        percentage = "n/a" if percent is None else f"{percent:.1f}%"
        lines.append(
            f"S{session}: {summary[f'units_{side}']} units; "
            f"{summary[f'matched_{side}']} matched ({percentage})"
        )
    text = "\n".join(lines)
    if session_summary_label.cget("text") != text:
        session_summary_label.configure(text=text)
    _refresh_group_summary(accepted_pairs, summary, session_a, session_b)


def _unusual_filter_enabled():
    toggle = globals().get("toggle_unusual_displacement_val")
    return bool(toggle is not None and toggle.get())


def get_ranked_unit_a_options(session_a, session_b, unusual_only=None):
    """Return one best-average partner per union-eligible Unit A."""
    if unusual_only is None:
        unusual_only = _unusual_filter_enabled()
    pairs = review_matches[
        (session_switch[session_a - 1] <= review_matches[:, 0])
        & (review_matches[:, 0] < session_switch[session_a])
        & (session_switch[session_b - 1] <= review_matches[:, 1])
        & (review_matches[:, 1] < session_switch[session_b])
    ]
    options = []
    seen_unit_a = set()
    for unit_a, unit_b in pairs:
        unit_a = int(unit_a)
        if unit_a not in seen_unit_a:
            options.append([unit_a, int(unit_b)])
            seen_unit_a.add(unit_a)
    if unusual_only:
        options = [pair for pair in options if tuple(pair) in consistency_filter_pairs]
    if not options and unusual_only:
        return []
    if not options:
        raise ValueError(
            f"No above-threshold candidates connect sessions "
            f"{session_a} and {session_b}"
        )
    return options


def get_ranked_unit_b_options(unit_a, session_b):
    """Keep all comparison candidates available when Unit A is filtered."""
    session_start = session_switch[session_b - 1]
    session_stop = session_switch[session_b]
    relative_order = np.argsort(
        -output_avg[unit_a, session_start:session_stop],
        kind="stable",
    )
    options = (relative_order + session_start).tolist()
    return options


def get_unit_a_display_options():
    display_options = []
    for unit_a, unit_b in option_a:
        session_b = np.searchsorted(session_switch, unit_b, side="right")
        session_start = session_switch[session_b - 1]
        session_stop = session_switch[session_b]
        match_count = sum(
            (unit_a, candidate_b) in automatic_candidate_pairs
            for candidate_b in range(session_start, session_stop)
        )
        display_options.append(f"{unit_a} ({match_count}) {unit_b}")
    return display_options


def _pair_review_category(
    unit_a,
    unit_b,
    accepted_pairs,
    score_matrix,
    session_ids,
):
    """Prefer accepted partners for A; compare scores for other A competitors."""
    pair_key = frozenset((int(unit_a), int(unit_b)))
    accepted = {
        frozenset(map(int, pair))
        for pair in accepted_pairs
        if len(pair) == 2 and int(pair[0]) != int(pair[1])
    }
    if pair_key in accepted:
        return "accepted"

    session_ids = np.asarray(session_ids)
    score_matrix = np.asarray(score_matrix)
    session_a = session_ids[unit_a]
    session_b = session_ids[unit_b]
    current_score = score_matrix[unit_a, unit_b]
    for alternative in accepted:
        if pair_key.isdisjoint(alternative):
            continue
        first, second = tuple(alternative)
        if (
            session_ids[first] == session_a
            and session_ids[second] == session_b
        ):
            source, target = first, second
        elif (
            session_ids[second] == session_a
            and session_ids[first] == session_b
        ):
            source, target = second, first
        else:
            continue
        if source == unit_a and target != unit_b:
            return "better_alternative"
        if target != unit_b or source == unit_a:
            continue
        alternative_score = score_matrix[source, target]
        if np.isfinite(alternative_score) and (
            not np.isfinite(current_score)
            or alternative_score >= current_score
        ):
            return "better_alternative"
    return "no_alternative"


def _unit_a_review_category(unit_a, unit_b):
    accepted_pairs = _curated_accepted_pairs(
        automatic_match_pairs,
        is_match,
        not_match,
    )
    return _pair_review_category(
        int(unit_a),
        int(unit_b),
        accepted_pairs,
        output_avg,
        clus_info["session_id"],
    )


def color_unit_a_options(event=None):
    """Color Unit A candidates by acceptance and conflict resolution."""
    accepted_pairs = _curated_accepted_pairs(
        automatic_match_pairs, is_match, not_match
    )
    _refresh_session_summary(accepted_pairs)
    for unit_a, unit_b in option_a:
        category = _pair_review_category(
            unit_a, unit_b, accepted_pairs, output_avg, clus_info["session_id"]
        )
        entry_a.set_review_status(unit_a, category)


def enable_unit_a_review_colors():
    root.after_idle(color_unit_a_options)


def create_unit_legend():
    """Create visual legend for unit colors"""
    global unit_legend_plot

    fig = Figure(figsize=_scaled_figsize(6, 0.5), dpi=100)
    fig.patch.set_facecolor("#33393b")

    ax = fig.add_subplot(111)
    ax.set_facecolor("#33393b")

    # Draw legend lines and text
    ax.plot([0, 0.15], [0.5, 0.5], color=UNIT_A_COLOR, lw=3)
    ax.text(
        0.18,
        0.5,
        "Unit A",
        color="white",
        fontsize=_scaled_font_size(12),
        va="center",
    )

    ax.plot([0.6, 0.75], [0.5, 0.5], color=UNIT_B_COLOR, lw=3)
    ax.text(
        0.78,
        0.5,
        "Unit B",
        color="white",
        fontsize=_scaled_font_size(12),
        va="center",
    )

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis("off")

    unit_legend_plot = FigureCanvasTkAgg(fig, master=root)
    unit_legend_plot.draw()
    unit_legend_plot = unit_legend_plot.get_tk_widget()


def create_hist_legend():
    """Create visual legend for histogram colors"""
    global hist_legend_plot

    fig = Figure(figsize=_scaled_figsize(8, 0.65), dpi=100)
    fig.patch.set_facecolor("#33393b")
    handles = [
        Line2D([], [], color=ALL_SCORES_COLOR, linewidth=3),
        Line2D([], [], color=APPROVED_MATCH_COLOR, linewidth=3),
        Line2D([], [], color="white", linewidth=2, linestyle="--"),
    ]
    legend = fig.legend(
        handles,
        ["All scores", "Expected matches", "Current pair"],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=_scaled_font_size(10),
        handlelength=3,
        handletextpad=0.7,
        columnspacing=1.8,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color("white")

    hist_legend_plot = FigureCanvasTkAgg(fig, master=root)
    hist_legend_plot.draw()
    hist_legend_plot = hist_legend_plot.get_tk_widget()


def compute_acg(spike_times, bin_size=0.001, max_lag=0.05):
    """
    Compute autocorrelogram for a unit

    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    bin_size : float
        Bin size in seconds (default 1ms)
    max_lag : float
        Maximum lag in seconds (default 50ms)

    Returns
    -------
    autocorr : array
        Autocorrelogram counts
    bin_centers : array
        Bin centers in seconds
    """
    if len(spike_times) < 2:
        return np.array([]), np.array([])

    # Create bins centered around 0
    bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    # Calculate autocorrelogram
    autocorr = np.zeros(len(bin_centers))

    # For efficiency, subsample spikes if there are too many
    if len(spike_times) > 10000:
        indices = np.random.choice(len(spike_times), 10000, replace=False)
        spike_subset = spike_times[indices]
    else:
        spike_subset = spike_times

    # Calculate cross-correlation with itself
    for i, spike_time in enumerate(spike_subset[::10]):  # Subsample further for speed
        # Find spikes within max_lag of this spike
        time_diffs = spike_times - spike_time
        valid_diffs = time_diffs[(np.abs(time_diffs) <= max_lag) & (time_diffs != 0)]

        if len(valid_diffs) > 0:
            hist, _ = np.histogram(valid_diffs, bins=bins)
            autocorr += hist

    # Convert to firing rate (spikes/sec)
    if len(spike_subset) > 0:
        recording_duration = (
            np.max(spike_times) - np.min(spike_times) if len(spike_times) > 0 else 1
        )
        autocorr = (
            autocorr / (len(spike_subset) * bin_size)
            if recording_duration > 0
            else autocorr
        )

    return autocorr, bin_centers


def plot_acgs(unit_a, unit_b):
    """Plot autocorrelograms for both units overlaid in single plot"""
    global acg_plot
    global clus_info
    global acg_cache

    # Destroy existing plot
    if "acg_plot" in globals() and _widget_exists(acg_plot):
        acg_plot.destroy()

    # Create figure for single overlaid ACG plot
    fig = Figure(figsize=_scaled_figsize(3, 3), dpi=100)
    fig.patch.set_facecolor("#33393b")

    # Create single subplot for overlaid ACGs
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#2d2d2d")

    max_rate = 0  # Track max rate for y-axis scaling

    # Try to get ACGs from cache first, otherwise compute them
    try:
        # Get ACG for Unit A
        if acg_cache is not None and unit_a in acg_cache:
            # Use cached ACG
            autocorr_a, bin_centers_a = acg_cache[unit_a]
        else:
            # Compute ACG on the fly
            spike_times_a = get_spike_times_for_unit(unit_a)
            if len(spike_times_a) > 1:
                autocorr_a, bin_centers_a = compute_acg(spike_times_a)
            else:
                autocorr_a, bin_centers_a = np.array([]), np.array([])

        # Plot ACG for Unit A
        if len(autocorr_a) > 0:
            positive_mask = bin_centers_a >= 0
            positive_centers = bin_centers_a[positive_mask] * 1000  # Convert to ms
            positive_autocorr = autocorr_a[positive_mask]

            ax.plot(
                positive_centers,
                positive_autocorr,
                color=UNIT_A_COLOR,
                linewidth=2,
                alpha=0.8,
                label=f"Unit A ({unit_a})",
            )
            max_rate = max(
                max_rate, np.max(positive_autocorr) if len(positive_autocorr) > 0 else 0
            )

        # Get ACG for Unit B
        if acg_cache is not None and unit_b in acg_cache:
            # Use cached ACG
            autocorr_b, bin_centers_b = acg_cache[unit_b]
        else:
            # Compute ACG on the fly
            spike_times_b = get_spike_times_for_unit(unit_b)
            if len(spike_times_b) > 1:
                autocorr_b, bin_centers_b = compute_acg(spike_times_b)
            else:
                autocorr_b, bin_centers_b = np.array([]), np.array([])

        # Plot ACG for Unit B
        if len(autocorr_b) > 0:
            positive_mask = bin_centers_b >= 0
            positive_centers = bin_centers_b[positive_mask] * 1000  # Convert to ms
            positive_autocorr = autocorr_b[positive_mask]

            ax.plot(
                positive_centers,
                positive_autocorr,
                color=UNIT_B_COLOR,
                linewidth=2,
                alpha=0.8,
                label=f"Unit B ({unit_b})",
            )
            max_rate = max(
                max_rate, np.max(positive_autocorr) if len(positive_autocorr) > 0 else 0
            )

        # Set labels and title
        ax.set_xlabel(
            "Time lag (ms)",
            fontsize=_scaled_font_size(10),
            color="white",
        )
        ax.set_ylabel(
            "Rate (Hz)",
            fontsize=_scaled_font_size(10),
            color="white",
        )
        ax.set_title(
            "Autocorrelograms",
            fontsize=_scaled_font_size(12),
            color="white",
        )
        ax.set_xlim(0, 50)  # 50ms
        if max_rate > 0:
            ax.set_ylim(0, max_rate * 1.1)

        # Add legend
        legend = ax.legend(
            fontsize=_scaled_font_size(8),
            loc="upper right",
        )
        legend.get_frame().set_facecolor("#33393b")
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color("white")

        # Style the plot
        ax.tick_params(colors="white", labelsize=_scaled_font_size(8))
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    except Exception:
        # If spike times not available, show placeholder
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#2d2d2d")
        ax.text(
            0.5,
            0.5,
            "ACG data\nnot available",
            ha="center",
            va="center",
            color="white",
            fontsize=_scaled_font_size(12),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Create canvas below the position/trajectory plot.
    acg_canvas = FigureCanvasTkAgg(fig, master=root)
    acg_canvas.draw()
    acg_plot = acg_canvas.get_tk_widget()
    acg_plot.grid(
        row=4,
        column=1,
        rowspan=2,
        padx=5,
        pady=5,
        sticky="nsew",
    )


def get_spike_times_for_unit(unit_id):
    """
    Get spike times for a specific unit from Kilosort data

    Parameters
    ----------
    unit_id : int
        UnitMatch unit ID

    Returns
    -------
    spike_times : array
        Spike times in seconds for the unit
    """
    try:
        # Get session information for this unit
        global clus_info, param
        session_id = clus_info["session_id"][unit_id]
        original_id = clus_info["original_ids"][unit_id]

        # Get the Kilosort directory for this session
        if "KS_dirs" in param:
            ks_dir = param["KS_dirs"][session_id]
        else:
            # Try to infer from existing paths
            return np.array([])

        # Load Kilosort spike times and cluster assignments
        spike_times_path = os.path.join(ks_dir, "spike_times.npy")
        spike_clusters_path = os.path.join(ks_dir, "spike_clusters.npy")

        if os.path.exists(spike_times_path) and os.path.exists(spike_clusters_path):
            # Load spike times (in samples) and cluster assignments
            spike_times_samples = np.load(spike_times_path).flatten()
            spike_clusters = np.load(spike_clusters_path).flatten()

            # Get sampling rate - try to load from params.py or use default
            try:
                params_path = os.path.join(ks_dir, "params.py")
                if os.path.exists(params_path):
                    # Parse params.py for sample_rate
                    with open(params_path, "r") as f:
                        params_content = f.read()
                    # Extract sample_rate
                    for line in params_content.split("\n"):
                        if "sample_rate" in line and "=" in line:
                            sample_rate = float(line.split("=")[1].strip())
                            break
                    else:
                        sample_rate = 30000.0  # Default Neuropixels sample rate
                else:
                    sample_rate = 30000.0  # Default
            except:
                sample_rate = 30000.0  # Default

            # Filter spikes for this unit
            unit_mask = spike_clusters == int(original_id)
            unit_spike_times = spike_times_samples[unit_mask]

            # Convert to seconds
            unit_spike_times_sec = unit_spike_times / sample_rate

            return unit_spike_times_sec
        else:
            print(f"Could not find spike_times.npy or spike_clusters.npy in {ks_dir}")
            return np.array([])

    except Exception as e:
        print(f"Error loading spike times for unit {unit_id}: {e}")
        return np.array([])


def update(event):
    """Coalesce queued navigation events before redrawing the selected pair."""
    global pending_redraw
    if pending_redraw is not None:
        root.after_cancel(pending_redraw)
    pending_redraw = root.after_idle(_render_selected_pair)


def _render_selected_pair():
    """
    Updates the GUI.
    """
    global pending_redraw
    pending_redraw = None
    _refresh_session_summary()
    if not entry_a.get() or not entry_b.get():
        return
    unit_a = int(entry_a.get())
    unit_b = int(entry_b.get())

    CV = get_cv_option()  # CV = 'Avg', if AVG is selcted else it equal [0,1] or [1,0]
    CV_option = CV_tkinter.get() - 1

    table = get_table_data(unit_a, unit_b, CV)
    MakeTable(table)

    scores_table = get_unit_score_table(unit_a, unit_b, CV_option)
    make_unit_score_table(scores_table)

    plot_avg_waveforms(unit_a, unit_b, CV)
    plot_trajectories(unit_a, unit_b, CV)

    if toggle_raw_val.get() is False:
        plot_raw_waveforms(unit_a, unit_b, CV)
    else:
        if raw_waveform_plot.winfo_exists() == 1:
            raw_waveform_plot.destroy()

    add_probability_label(unit_a, unit_b, CV_option)
    add_original_ID(unit_a, unit_b)

    if toggle_UM_score_val.get() is False:
        hist_names, hist, hist_matches = _get_score_histograms_for_cv(CV_option)
        scores = (
            scores_to_include_avg
            if CV_option == -1
            else scores_to_include_GUI[CV_option]
        )
        plot_histograms(hist_names, hist, hist_matches, scores, unit_a, unit_b)
    else:
        if hist_plot.winfo_exists() == 1:
            hist_plot.destroy()

    # Plot ACGs for both units if not hidden
    if toggle_acg_val.get() is False:
        plot_acgs(unit_a, unit_b)
    else:
        if "acg_plot" in globals() and acg_plot.winfo_exists() == 1:
            acg_plot.destroy()
    if callable(event_view_refresh):
        event_view_refresh()
    color_unit_a_options()
    install_navigation_bindtags(root)


def select_unit_b(unit_id):
    entry_b.current(option_b.index(int(unit_id)))


def up_options_b_list(event):
    """
    moves up the option B list, chose the unit with th next highest probabilty of being a match with unit A.
    """
    global entry_frame
    global option_b
    global entry_b

    if not option_b or not entry_b.get():
        return "break"
    tmp_entry_b = int(entry_b.get())
    current_idx = option_b.index(tmp_entry_b)
    if current_idx == 0:
        return "break"

    entry_b.current(current_idx - 1)
    update(event)
    return "break"


def down_options_b_list(event):
    """
    moves down the option B list, chose the unit with th next highest probabilty of being a match with unit A.
    """
    global entry_frame
    global option_b
    global entry_b

    if not option_b or not entry_b.get():
        return "break"
    tmp_entry_b = int(entry_b.get())
    current_idx = option_b.index(tmp_entry_b)
    if current_idx == (len(option_b) - 1):
        return "break"

    entry_b.current(current_idx + 1)
    update(event)
    return "break"


def bind_unit_b_navigation():
    root.bind_class("UnitMatchNavigation", "<Up>", up_options_b_list)
    root.bind_class("UnitMatchNavigation", "<Down>", down_options_b_list)
    root.bind_class("UnitMatchNavigation", "<Right>", next_pair)
    root.bind_class("UnitMatchNavigation", "<Left>", previous_pair)
    entry_b.bind("<<UnitBSelected>>", update)
    install_navigation_bindtags(root)


def install_navigation_bindtags(widget):
    if isinstance(widget, ttk.Treeview):
        return
    bindtags = widget.bindtags()
    if "UnitMatchNavigation" not in bindtags:
        widget.bindtags(("UnitMatchNavigation", *bindtags))
    for child in widget.winfo_children():
        install_navigation_bindtags(child)


# sort CV function as to be calleed as part of update
def get_cv_option():
    """
    Will read in the values of the radio button and assign an appropriate value to CV.
    In general this is a list where itis [Unit A cv, UnitB cv], however it could be the string 'Avg'
    """
    global CV_tkinter
    ChosenOption = CV_tkinter.get()
    if ChosenOption == 0:
        CV = "Avg"
    elif ChosenOption == 1:
        CV = [0, 1]
    elif ChosenOption == 2:
        CV = [1, 0]
    return CV


# These are the function used to selct units, including how the CV selctition radio buttons, session selction, unitselection andmoving left and rigthfor next units
def _unusual_filter_status(session_a, session_b, option_count):
    return (
        f"{option_count} Unit A rows below {consistency_filter_threshold:g}; "
        f"{consistency_filter_unavailable} n/a excluded"
    )


def _cancel_consistency_filter():
    global consistency_filter_after
    consistency_filter_cancel.set()
    if consistency_filter_future is not None:
        consistency_filter_future.cancel()
    if consistency_filter_after is not None:
        root.after_cancel(consistency_filter_after)
        consistency_filter_after = None


def _compute_consistency_filter(options, model, threshold, cancel):
    selected, unavailable = set(), 0
    for unit_a, unit_b in options:
        if cancel.is_set():
            return None
        result = model.score(unit_a, unit_b)
        if result.score is None:
            unavailable += 1
        elif result.score < threshold:
            selected.add((unit_a, unit_b))
    return selected, unavailable


def _set_pair_controls(enabled):
    for name in ("match_button", "non_match_button"):
        button = globals().get(name)
        if button is not None and _widget_exists(button):
            button.configure(state="normal" if enabled else "disabled")


def update_unusual_displacement_filter():
    """Fit/filter in the worker; never block Tk on robust covariance fitting."""
    global consistency_filter_cancel, consistency_filter_future, consistency_filter_after
    global displacement_metric_executor, consistency_filter_threshold
    _refresh_session_summary()
    _cancel_consistency_filter()
    try:
        threshold = float(consistency_threshold_var.get())
    except (TclError, ValueError):
        _set_pair_controls(bool(entry_a.get() and entry_b.get()))
        unusual_displacement_status_label.configure(text="Enter a cutoff between 0 and 100")
        return
    if not np.isfinite(threshold) or not 0 <= threshold <= 100:
        _set_pair_controls(bool(entry_a.get() and entry_b.get()))
        unusual_displacement_status_label.configure(text="Enter a cutoff between 0 and 100")
        return
    consistency_filter_threshold = threshold
    if not _unusual_filter_enabled():
        _apply_displacement_filter()
        return
    model, reason, _ = _get_displacement_context()
    if model is None:
        toggle_unusual_displacement_val.set(False)
        _apply_displacement_filter()
        unusual_displacement_status_label.configure(text=f"Unavailable: {reason}")
        return
    options = get_ranked_unit_a_options(
        int(session_entry_a.get()), int(session_entry_b.get()), unusual_only=False,
    )
    consistency_filter_cancel = Event()
    _set_pair_controls(False)
    unusual_displacement_status_label.configure(text="Calculating consistency filter...")
    if displacement_metric_executor is None:
        displacement_metric_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="unitmatch-displacement",
        )
    consistency_filter_future = displacement_metric_executor.submit(
        _compute_consistency_filter, options, model, threshold, consistency_filter_cancel,
    )
    consistency_filter_after = root.after(50, _poll_consistency_filter)


def _poll_consistency_filter():
    global consistency_filter_after, consistency_filter_pairs, consistency_filter_unavailable
    consistency_filter_after = None
    if not consistency_filter_future.done():
        consistency_filter_after = root.after(50, _poll_consistency_filter)
        return
    error = consistency_filter_future.exception()
    if error is not None:
        _set_pair_controls(True)
        unusual_displacement_status_label.configure(text=f"Consistency filter failed: {error}")
    result = consistency_filter_future.result()
    if result is not None:
        consistency_filter_pairs, consistency_filter_unavailable = result
        _apply_displacement_filter()


def _apply_displacement_filter():
    global match_idx
    global option_a
    global option_b

    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    unusual_only = _unusual_filter_enabled()
    previous_a = entry_a.get().split()[0] if entry_a.get() else None
    previous_b = entry_b.get() if entry_b.get() else None

    option_a = get_ranked_unit_a_options(
        session_a,
        session_b,
        unusual_only=unusual_only,
    )
    entry_a.set_options(get_unit_a_display_options())
    _set_pair_controls(bool(option_a))
    if not option_a:
        option_b = []
        entry_a.set("")
        entry_b.set_options([])
        entry_b.set("")
        match_idx = 0
        unusual_displacement_status_label.configure(
            text=_unusual_filter_status(session_a, session_b, 0)
        )
        return

    valid_unit_a_ids = [pair[0] for pair in option_a]
    preferred_a = int(previous_a) if previous_a else None
    unit_a = (
        preferred_a
        if preferred_a in valid_unit_a_ids
        else valid_unit_a_ids[0]
    )
    entry_a.set(unit_a)
    option_b = get_ranked_unit_b_options(
        unit_a,
        session_b,
    )
    entry_b.set_options(option_b)
    preferred_b = int(previous_b) if previous_b else None
    default_b = next(partner for unit, partner in option_a if unit == unit_a)
    unit_b = (
        preferred_b
        if preferred_a == unit_a and preferred_b in option_b
        else default_b
    )
    entry_b.set(unit_b)
    match_idx = next(
        index
        for index, (candidate_a, _) in enumerate(option_a)
        if candidate_a == unit_a
    )
    if unusual_only:
        unusual_displacement_status_label.configure(
            text=_unusual_filter_status(session_a, session_b, len(option_a))
        )
    else:
        unusual_displacement_status_label.configure(
            text=(
                "Filter off"
            )
        )
    color_unit_a_options()
    update(None)


def update_unit_cv():
    """
    When updating the CV we need to do /not do the following:
    - keep the same selected units,
    - update the options in boxes for unitA and unitB as matches and likely matches units can change
    - make it so when scrolling the list #MATCHIDX AUTOMATICALLY UPDATES TO THE CORrECT POINT IN THE NEW CV
    - update the screen to show the new CV
    """
    global entry_a
    global entry_b
    global option_a
    global option_b
    global session_entry_a
    global session_entry_b
    global match_idx

    # selecting the unit
    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    if _unusual_filter_enabled():
        update_unusual_displacement_filter()
        return

    # Keep track of the unit it was before as we dont want to change theunit viewed when changing the CV
    entry_a_tmp = int(entry_a.get())
    entry_b_tmp = int(entry_b.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = UnitATable(
        entry_frame, values=get_unit_a_display_options()
    )
    entry_a.set(entry_a_tmp)
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    select_unit_b(entry_b_tmp)

    entry_a.bind("<<UnitASelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()
    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    tmp_list = [int(entry_a_tmp), int(entry_b_tmp)]
    if tmp_list in option_a:
        match_idx = option_a.index(tmp_list)

    update(None)


def update_unit_entryA(event):
    """
    This will be called when the user changes the sesion A, and the following will happen:
    - Keep the same B unit
    - Set A to the first match of the new session
    - Set the Match IDX at 0, for the new list of match pairs
    - update the options in list A
    - update the options for list B
    - update what is in the screen
    """
    global entry_a
    global entry_b
    global option_a
    global option_b
    global match_idx
    global session_entry_b

    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    if session_a == session_b and param["n_sessions"] > 1:
        session_b = session_a % param["n_sessions"] + 1
        session_entry_b.set(session_b)
    if _unusual_filter_enabled():
        update_unusual_displacement_filter()
        return
    EntryBtmp = int(entry_b.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = UnitATable(
        entry_frame, values=get_unit_a_display_options()
    )
    entry_a.set(option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)
    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    select_unit_b(EntryBtmp if EntryBtmp in option_b else option_b[0])
    entry_a.bind("<<UnitASelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()

    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)
    match_idx = 0

    update(event)


def update_unit_entryB(event):
    """
    This will be called when the user changes the sesion B, and the following will happen:
    - Keep the same A unit
    - update the options in list A
    - update the options for list B
    - Set B to the highest match of unit A for the new session B
    - Set the Match IDX at 0, for the new list of match pairs
    - update what is in the screen
    """
    global entry_a
    global entry_b
    global option_a
    global option_b
    global match_idx
    global session_entry_a

    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    if session_a == session_b and param["n_sessions"] > 1:
        session_a = session_b % param["n_sessions"] + 1
        session_entry_a.set(session_a)
    if _unusual_filter_enabled():
        update_unusual_displacement_filter()
        return
    entry_a_tmp = int(entry_a.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = UnitATable(
        entry_frame, values=get_unit_a_display_options()
    )
    valid_unit_a_ids = {pair[0] for pair in option_a}
    entry_a.set(entry_a_tmp if entry_a_tmp in valid_unit_a_ids else option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    entry_b.current(0)
    entry_a.bind("<<UnitASelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()

    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)
    match_idx = 0

    update(event)


def update_units(event):
    """
    This function is called when a row in the Unit A table is selected.
    - split the list of units into two pairs for unit labels for UnitA and UnitB
    - Update the unit B dropdown box to reflect matches for the new Unit A
    - Find the new Match Idx for the new pair which is selected
    """
    global match_idx
    global entry_a
    global entry_b
    global option_a
    global option_b
    global session_entry_b

    session_b = int(session_entry_b.get())
    selected = entry_a.get()

    if entry_b.winfo_exists() == 1:
        entry_b.destroy()
    tmpA = selected.split()[0]
    tmpB = next(partner for unit, partner in option_a if unit == int(tmpA))
    entry_a.set(tmpA)

    option_b = get_ranked_unit_b_options(int(tmpA), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    entry_b.set(tmpB)
    bind_unit_b_navigation()

    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    # need to get list index of the selected postion
    tmp_list = [int(tmpA), int(tmpB)]
    match_idx = option_a.index(tmp_list)
    update(event)


def next_pair(event):
    """
    This function is called when moving to the next unit in the mathc list:
    - get the new unit idx for unit A and unit B
    - update the Units
    - update the dropdown box for unit b to reflect the new Unit A
    """
    global match_idx
    global entry_a
    global entry_b
    global option_a
    global option_b
    global session_entry_b

    if not option_a:
        return "break"
    session_b = int(session_entry_b.get())
    match_idx = (match_idx + 1) % len(option_a)
    tmp_a, tmp_b = option_a[match_idx]
    entry_a.set(tmp_a)

    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_b = get_ranked_unit_b_options(int(tmp_a), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    entry_b.set(tmp_b)
    bind_unit_b_navigation()

    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    update(event)
    return "break"


def previous_pair(event):
    """
    This function is called when moving to the previous unit in the mathc list:
    - get the new unit idx for unit A and unit B
    - update the Units
    - update the dropdown box for unit b to reflect the new Unit A
    """
    global match_idx
    global entry_a
    global entry_b
    global option_a
    global option_b
    global session_entry_b

    if not option_a:
        return "break"
    session_b = int(session_entry_b.get())
    match_idx = (match_idx - 1) % len(option_a)
    tmp_a, tmp_b = option_a[match_idx]
    entry_a.set(tmp_a)

    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_b = get_ranked_unit_b_options(int(tmp_a), session_b)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    entry_b.set(tmp_b)
    bind_unit_b_navigation()

    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    update(event)
    return "break"


def swap_units():
    global entry_frame
    global entry_a
    global entry_b
    global session_entry_b
    global session_entry_b
    global match_idx
    global option_a
    global option_b

    if _unusual_filter_enabled():
        session_a_tmp = int(session_entry_a.get())
        session_b_tmp = int(session_entry_b.get())
        session_entry_a.set(session_b_tmp)
        session_entry_b.set(session_a_tmp)
        update_unusual_displacement_filter()
        return

    # get all initial info
    entry_a_tmp = int(entry_a.get())
    entry_b_tmp = int(entry_b.get())
    session_a_tmp = int(session_entry_a.get())
    session_b_tmp = int(session_entry_b.get())

    # delte old box a and b
    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    # swap allpairs
    session_entry_a.set(session_b_tmp)
    session_entry_b.set(session_a_tmp)

    option_a = get_ranked_unit_a_options(session_b_tmp, session_a_tmp)
    entry_a = UnitATable(
        entry_frame, values=get_unit_a_display_options()
    )
    entry_a.set(entry_b_tmp)
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_a_tmp)

    ##################
    entry_b = UnitBTable(entry_frame, values=option_b)
    select_unit_b(entry_a_tmp)

    entry_a.bind("<<UnitASelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()
    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    tmp_list = [int(entry_b_tmp), int(entry_a_tmp)]

    match_idx = option_a.index(tmp_list)
    update(None)


def _get_score_histograms_for_cv(cv_option):
    """Cache immutable histogram data only when its plot is requested."""
    # Transposing both scores and their match mask preserves both histograms.
    cache_key = -1 if cv_option == -1 else 0
    if cache_key not in score_histogram_cache:
        scores = scores_to_include_avg if cache_key == -1 else scores_to_include_GUI[0]
        probabilities = output_avg if cache_key == -1 else output_GUI[0]
        score_histogram_cache[cache_key] = get_score_histograms(
            scores, probabilities > match_threshold
        )
    return score_histogram_cache[cache_key]


def get_score_histograms(scores_to_include, output_threshold):
    """
    Scores2Include is the dictionary of all scores.
    ProbThreshold is a nUnits*nUnits array where each index is 0 (Not Match) 1 (Match)
    """

    # are lsit of length 6 each list item is 2 np arrays(bins, values)
    hist_names = []
    hist = []
    hist_matches = []
    match_mask = np.asarray(output_threshold, dtype=bool)
    for key, values in scores_to_include.items():
        hist_names.append(key)
        hist.append(np.histogram(values, bins=100, density=True))
        hist_matches.append(
            np.histogram(values[match_mask], bins=100, density=True)
        )

    return hist_names, hist, hist_matches


def _get_event_psth(unit_index, event_name, before_s, after_s, bin_size_s):
    global event_view_settings
    settings = (before_s, after_s, bin_size_s)
    if settings != event_view_settings:
        event_view_settings = settings
        for name in ("entry_a", "entry_b"):
            table = globals().get(name)
            if isinstance(table, _DiagnosticUnitTable) and _widget_exists(table):
                table.refresh_event_metrics()
    return _cached_event_psth(
        unit_index, event_name, settings,
        event_data, clus_info["session_indices"],
        event_spike_times_cache, event_psth_cache,
    )


def _cached_event_psth(unit_index, event_name, settings, data, session_indices, spike_cache, psth_cache):
    before_s, after_s, bin_size_s = settings
    session_index = int(session_indices[unit_index])
    event_times = data["event_times_by_session"][session_index][event_name]
    if unit_index not in spike_cache:
        spike_times = data["get_spike_times"](unit_index)
        spike_cache[unit_index] = np.asarray(
            spike_times,
            dtype=float,
        )

    cache_key = (
        unit_index,
        event_name,
        before_s,
        after_s,
        bin_size_s,
    )
    if cache_key not in psth_cache:
        psth_cache[cache_key] = _compute_event_psth(
            spike_cache[unit_index],
            event_times,
            before_s=before_s,
            after_s=after_s,
            bin_size_s=bin_size_s,
        )
    return psth_cache[cache_key]


def _event_response_correlation(unit_a, unit_b, settings, context):
    data, sessions, spike_cache, psth_cache, correlation_cache = context
    key = (*sorted((unit_a, unit_b)), *settings)
    if key in correlation_cache:
        return correlation_cache[key]
    events_a = data["event_times_by_session"][int(sessions[unit_a])]
    events_b = data["event_times_by_session"][int(sessions[unit_b])]
    correlations = []
    for name in sorted(events_a.keys() & events_b.keys()):
        if min(len(events_a[name]), len(events_b[name])) < 2:
            continue
        profiles = [
            _cached_event_psth(unit, name, settings, data, sessions, spike_cache, psth_cache)[1]
            for unit in (unit_a, unit_b)
        ]
        if any(
            not np.all(np.isfinite(profile))
            or np.ptp(profile) <= np.finfo(float).eps * max(1, np.max(np.abs(profile))) * 32
            for profile in profiles
        ):
            continue
        correlations.append(float(np.corrcoef(*profiles)[0, 1]))
    result = float(np.mean(correlations)) if correlations else None
    correlation_cache[key] = result
    return result


def _compute_candidate_event_correlations(pairs, settings, context, cancelled, results):
    for row, unit_a, unit_b in pairs:
        if cancelled.is_set():
            return
        try:
            correlation = _event_response_correlation(unit_a, unit_b, settings, context)
        except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
            results.put((row, None, str(error)))
        else:
            results.put((row, correlation, None))


def _event_unit_title(label, unit_index):
    session_index = int(clus_info["session_indices"][unit_index])
    session_names = event_data.get("session_names")
    session_name = (
        session_names[session_index]
        if session_names is not None
        else f"session {session_index + 1}"
    )
    title = f"{label}: row {unit_index}, {session_name}"
    if {"probe_numbers", "analyzer_unit_ids"}.issubset(clus_info):
        title += (
            f", probe {clus_info['probe_numbers'][unit_index]}, "
            f"unit {clus_info['analyzer_unit_ids'][unit_index]}"
        )
    return title


def open_event_view():
    """Open a lazily computed event-aligned PSTH viewer."""
    global event_view_refresh
    global event_view_window

    if event_view_window is not None and _widget_exists(event_view_window):
        event_view_window.lift()
        event_view_window.focus_force()
        event_view_refresh()
        return

    event_view_window = Toplevel(root)
    event_view_window.title("Event Viewer")
    event_view_window.geometry(
        f"{round(900 * gui_scale)}x{round(700 * gui_scale)}"
    )

    controls = ttk.Frame(event_view_window, padding=8)
    controls.pack(fill=X)
    before_var = StringVar(master=event_view_window, value=str(event_view_settings[0]))
    after_var = StringVar(master=event_view_window, value=str(event_view_settings[1]))
    bin_size_var = StringVar(master=event_view_window, value=str(event_view_settings[2]))
    for column, (label, variable) in enumerate(
        (
            ("Before (s)", before_var),
            ("After (s)", after_var),
            ("Bin (s)", bin_size_var),
        )
    ):
        ttk.Label(controls, text=label).grid(
            row=0,
            column=column * 2,
            padx=(0, 4),
        )
        ttk.Entry(controls, textvariable=variable, width=7).grid(
            row=0,
            column=column * 2 + 1,
            padx=(0, 10),
        )

    status_label = ttk.Label(controls)
    status_label.grid(row=0, column=7, sticky="W", padx=10)

    figure = Figure(figsize=_scaled_figsize(8, 6), dpi=100)
    figure.patch.set_facecolor("#33393b")
    axes = figure.subplots(2, 1, sharex=True)
    canvas = FigureCanvasTkAgg(figure, master=event_view_window)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def refresh():
        if not _widget_exists(event_view_window):
            return
        try:
            before_s = float(before_var.get())
            after_s = float(after_var.get())
            bin_size_s = float(bin_size_var.get())
            if before_s <= 0 or after_s <= 0 or bin_size_s <= 0:
                raise ValueError("Windows and bin size must be positive.")

            unit_indices = (int(entry_a.get()), int(entry_b.get()))
            for label, unit_index, axis in zip(
                ("Unit A", "Unit B"),
                unit_indices,
                axes,
            ):
                axis.clear()
                axis.set_facecolor("#2d2d2d")
                session_index = int(clus_info["session_indices"][unit_index])
                events = event_data["event_times_by_session"][session_index]
                for event_name in events:
                    centers, firing_rate, event_count = _get_event_psth(
                        unit_index,
                        event_name,
                        before_s,
                        after_s,
                        bin_size_s,
                    )
                    axis.plot(
                        centers,
                        firing_rate,
                        label=f"{event_name} (n={event_count})",
                    )
                axis.axvline(0, color="#DDDDDD", linewidth=0.8, alpha=0.7)
                axis.set_title(
                    _event_unit_title(label, unit_index),
                    fontsize=_scaled_font_size(10),
                )
                axis.set_ylabel("Firing rate (Hz)")
                axis.grid(alpha=0.2)
                if events:
                    legend = axis.legend(
                        fontsize=_scaled_font_size(8),
                        ncol=2,
                    )
                    legend.get_frame().set_facecolor("#33393b")
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No event times available for this session",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
            axes[-1].set_xlabel("Time from event (s)")
            figure.tight_layout()
            status_label.configure(text="")
            canvas.draw_idle()
        except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
            status_label.configure(text=str(error))

    def close():
        global event_view_refresh
        global event_view_window
        event_view_refresh = None
        event_view_window.destroy()
        event_view_window = None

    ttk.Button(controls, text="Refresh", command=refresh).grid(
        row=0,
        column=6,
        padx=4,
    )
    event_view_window.protocol("WM_DELETE_WINDOW", close)
    event_view_refresh = refresh
    refresh()


def open_pair_lookup():
    """Open a selectable unit-identity probability lookup window."""
    global pair_lookup_window

    existing_window = globals().get("pair_lookup_window")
    if existing_window is not None and _widget_exists(existing_window):
        pair_lookup_window.lift()
        pair_lookup_window.focus_force()
        return

    if "session_indices" not in clus_info:
        clus_info["session_indices"] = np.searchsorted(
            np.asarray(clus_info["session_switch"])[1:],
            np.arange(len(clus_info["original_ids"])),
            side="right",
        )

    required_metadata = {"probe_numbers", "analyzer_unit_ids"}
    missing_metadata = required_metadata.difference(clus_info)
    if missing_metadata:
        pair_lookup_window = Toplevel(root)
        pair_lookup_window.title("Inspect UnitMatch Pair")
        ttk.Label(
            pair_lookup_window,
            text=(
                "Pair lookup metadata is unavailable. Rerun the Section 4 "
                "scoring cell and the Section 5 GUI initialization cell.\n"
                f"Missing: {sorted(missing_metadata)}"
            ),
            justify="left",
            padding=12,
        ).grid()
        return

    pair_lookup_window = Toplevel(root)
    pair_lookup_window.title("Inspect UnitMatch Pair")
    pair_lookup_window.resizable(False, False)

    session_values = list(range(1, param["n_sessions"] + 1))
    probe_values = sorted(
        {int(probe_n) for probe_n in clus_info["probe_numbers"]}
    )

    try:
        selected_row_a = int(entry_a.get().split()[0])
        selected_row_b = int(entry_b.get())
    except ValueError:
        selected_row_a = 0
        selected_row_b = min(1, len(clus_info["session_indices"]) - 1)

    selected_rows = (selected_row_a, selected_row_b)
    session_variables = []
    probe_variables = []
    unit_variables = []
    for column, (unit_name, row_index) in enumerate(
        zip(("Unit A", "Unit B"), selected_rows),
        start=1,
    ):
        ttk.Label(
            pair_lookup_window,
            text=unit_name,
            font=("DejaVu Sans", 11, "bold"),
        ).grid(row=0, column=column, padx=8, pady=(8, 4))

        session_variable = IntVar(
            master=pair_lookup_window,
            value=int(clus_info["session_indices"][row_index]) + 1
        )
        probe_variable = IntVar(
            master=pair_lookup_window,
            value=int(clus_info["probe_numbers"][row_index])
        )
        unit_variable = StringVar(
            master=pair_lookup_window,
            value=str(clus_info["analyzer_unit_ids"][row_index])
        )
        session_variables.append(session_variable)
        probe_variables.append(probe_variable)
        unit_variables.append(unit_variable)

        ttk.Combobox(
            pair_lookup_window,
            values=session_values,
            textvariable=session_variable,
            state="readonly",
            width=12,
        ).grid(row=1, column=column, padx=8, pady=3)
        ttk.Combobox(
            pair_lookup_window,
            values=probe_values,
            textvariable=probe_variable,
            state="readonly",
            width=12,
        ).grid(row=2, column=column, padx=8, pady=3)
        ttk.Entry(
            pair_lookup_window,
            textvariable=unit_variable,
            width=15,
        ).grid(row=3, column=column, padx=8, pady=3)

    for row, label_text in enumerate(
        ("Session", "Probe", "Analyzer unit ID"),
        start=1,
    ):
        ttk.Label(pair_lookup_window, text=label_text).grid(
            row=row,
            column=0,
            sticky="e",
            padx=(8, 4),
            pady=3,
        )

    result_text = Text(
        pair_lookup_window,
        width=68,
        height=9,
        wrap="none",
        borderwidth=2,
        relief="groove",
    )
    result_text.grid(
        row=5,
        column=0,
        columnspan=3,
        padx=8,
        pady=(6, 8),
    )

    def show_lookup_result(event=None):
        result_text.configure(state="normal")
        result_text.delete("1.0", END)
        try:
            sessions = [variable.get() - 1 for variable in session_variables]
            probes = [variable.get() for variable in probe_variables]
            unit_ids = [int(variable.get()) for variable in unit_variables]
            if probes[0] != probes[1]:
                raise ValueError(
                    "Units on different probes are not valid UnitMatch pairs."
                )

            matrix_rows = []
            for session_index, probe_n, unit_id in zip(
                sessions,
                probes,
                unit_ids,
            ):
                matches = np.flatnonzero(
                    (clus_info["session_indices"] == session_index)
                    & (clus_info["probe_numbers"] == probe_n)
                    & (clus_info["analyzer_unit_ids"] == unit_id)
                )
                if matches.size == 0:
                    raise ValueError(
                        f"Unit {unit_id} was not exported from "
                        f"Session {session_index + 1}, probe{probe_n}."
                    )
                if matches.size > 1:
                    raise RuntimeError(
                        f"Unit {unit_id} maps to multiple matrix rows: "
                        f"{matches.tolist()}."
                    )
                matrix_rows.append(int(matches[0]))

            row_a, row_b = matrix_rows
            probability_12 = raw_output[row_a, row_b]
            probability_21 = raw_output[row_b, row_a]
            average_probability = (probability_12 + probability_21) / 2
            result = (
                f"Matrix rows: A={row_a}, B={row_b}\n"
                f"CV (A1, B2): {probability_12:.5f}\n"
                f"CV (A2, B1): {probability_21:.5f}\n"
                f"Average: {average_probability:.5f}\n"
                f"Either direction above {match_threshold:.5f}: "
                f"{max(probability_12, probability_21) > match_threshold}\n"
                f"Both directions above {match_threshold:.5f}: "
                f"{min(probability_12, probability_21) > match_threshold}"
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            result = f"Cannot inspect pair: {error}"

        result_text.insert("1.0", result)
        result_text.configure(state="disabled")
        return "break"

    ttk.Button(
        pair_lookup_window,
        text="Show probabilities",
        command=show_lookup_result,
    ).grid(row=4, column=0, columnspan=3, pady=(7, 3))
    pair_lookup_window.bind("<Return>", show_lookup_result)
    show_lookup_result()


def add_original_ID(UnitA, UnitB):
    global original_id_label
    global root
    global clus_info

    if original_id_label.winfo_exists():
        original_id_label.destroy()

    try:
        export_id_a = int(clus_info["original_ids"][UnitA].squeeze())
        export_id_b = int(clus_info["original_ids"][UnitB].squeeze())
        if "probe_numbers" in clus_info and "analyzer_unit_ids" in clus_info:
            probe_a = int(clus_info["probe_numbers"][UnitA])
            probe_b = int(clus_info["probe_numbers"][UnitB])
            analyzer_id_a = clus_info["analyzer_unit_ids"][UnitA]
            analyzer_id_b = clus_info["analyzer_unit_ids"][UnitB]
            identity_rows = (
                f"Unit A: row {UnitA}, probe{probe_a}, "
                f"analyzer unit {analyzer_id_a} "
                f"(export ID {export_id_a})",
                f"Unit B: row {UnitB}, probe{probe_b}, "
                f"analyzer unit {analyzer_id_b} "
                f"(export ID {export_id_b})",
            )
        else:
            identity_rows = (
                f"Unit A: row {UnitA}, export ID {export_id_a}",
                f"Unit B: row {UnitB}, export ID {export_id_b}",
            )
        original_id_label = ttk.LabelFrame(review_info_frame, text="Unit identities")
        for row, identity_text in enumerate(identity_rows):
            identity_entry = ttk.Entry(original_id_label, width=34)
            identity_entry.insert(0, identity_text)
            identity_entry.configure(state="readonly")
            identity_entry.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        original_id_label.columnconfigure(0, weight=1)
    except IndexError as e:
        print(f"Error: {e}")
        original_id_label = ttk.Label(
            review_info_frame, text="Error: Unit ID out of bounds", borderwidth=2, relief="groove"
        )

    original_id_label.grid(
        row=0,
        column=0,
        sticky="ew",
        pady=5,
    )


def add_probability_label(UnitA, UnitB, CVoption):
    global bayes_label

    if bayes_label.winfo_exists():
        bayes_label.destroy()

    if CVoption == -1:
        probability = output_avg[UnitA, UnitB]
        probability_text = (
            f"Average probability: {probability:.5f}\n"
            f"CV (1,2): {output_GUI[0][UnitA, UnitB]:.5f}\n"
            f"CV (2,1): {output_GUI[1][UnitA, UnitB]:.5f}"
        )
    else:
        probability = output_GUI[CVoption][UnitA, UnitB]
        probability_text = f"UnitMatch probability: {probability:.5f}"

    is_automatic_match = (UnitA, UnitB) in automatic_match_pairs
    status = "MATCH" if is_automatic_match else "NOT MATCH"
    status_color = APPROVED_MATCH_COLOR if is_automatic_match else "red"
    pair = [UnitA, UnitB]
    if pair in not_match:
        manual_status = "NON MATCH"
        label_color = "red"
    elif pair in is_match:
        manual_status = "MATCH"
        label_color = APPROVED_MATCH_COLOR
    else:
        manual_status = "UNREVIEWED"
        label_color = status_color
    bayes_label = ttk.Label(
        review_info_frame,
        text=(
            f"{probability_text}\n"
            f"Threshold: {match_threshold:.5f}\n"
            f"Automatic {automatic_match_mode.upper()} decision: {status}\n"
            f"Manual decision: {manual_status}"
        ),
        foreground=label_color,
        borderwidth=2,
        relief="groove",
        justify="center",
        anchor="center",
        width=34,
    )
    bayes_label.grid(
        row=1,
        column=0,
        sticky="ew",
        pady=5,
    )


def _accept_pair(unit_a, unit_b):
    """Replace accepted competitors at either endpoint within this session pair."""
    sessions = clus_info["session_id"]
    session_pair = tuple(sorted((sessions[unit_a], sessions[unit_b])))
    selected_units = {unit_a, unit_b}
    conflicts = [
        (first, second)
        for first, second in _curated_accepted_pairs(
            automatic_match_pairs, is_match, not_match,
        )
        if {first, second} != selected_units
        and (first in selected_units or second in selected_units)
        and tuple(sorted((sessions[first], sessions[second]))) == session_pair
    ]
    rejected_pairs = {
        pair
        for first, second in conflicts
        for pair in ((first, second), (second, first))
    }
    is_match[:] = [pair for pair in is_match if tuple(pair) not in rejected_pairs]
    for pair in sorted(rejected_pairs):
        if list(pair) not in not_match:
            not_match.append(list(pair))
    pairs = [[unit_a, unit_b], [unit_b, unit_a]]
    not_match[:] = [pair for pair in not_match if pair not in pairs]
    for pair in pairs:
        if pair not in is_match:
            is_match.append(pair)


def set_match(event=None):
    if not entry_a.get() or not entry_b.get():
        warnings.warn("Select a pair before accepting a match.", RuntimeWarning)
        return
    unit_a = int(entry_a.get())
    unit_b = int(entry_b.get())
    _accept_pair(unit_a, unit_b)
    add_probability_label(unit_a, unit_b, CV_tkinter.get() - 1)
    _refresh_raw_displacement_overlay()
    color_unit_a_options()
    _refresh_displacement_consistency()


def set_not_match(event=None):
    global is_match
    global not_match
    if not entry_a.get() or not entry_b.get():
        warnings.warn("Select a pair before rejecting a match.", RuntimeWarning)
        return
    unit_a = int(entry_a.get())
    unit_b = int(entry_b.get())

    pairs = [[unit_a, unit_b], [unit_b, unit_a]]
    is_match[:] = [pair for pair in is_match if pair not in pairs]
    for pair in pairs:
        if pair not in not_match:
            not_match.append(pair)
    add_probability_label(unit_a, unit_b, CV_tkinter.get() - 1)
    _refresh_raw_displacement_overlay()
    color_unit_a_options()
    _refresh_displacement_consistency()


def _fit_review_tables(event=None):
    if unit_tables_frame is None or not _widget_exists(unit_tables_frame):
        return
    minimum_width = max(
        unit_tables_frame._minimum_column_width,
        sum(frame.winfo_reqwidth() for frame in unit_tables_frame.winfo_children()) + 28,
    )
    if session_summary_label is not None and _widget_exists(session_summary_label):
        minimum_width = max(minimum_width, session_summary_label.master.winfo_reqwidth() + 20)
    if root.grid_columnconfigure(0)["minsize"] != minimum_width:
        root.columnconfigure(0, minsize=minimum_width)


def _make_review_table(table, title, column):
    """Keep both read-only tables together with content-sized columns."""
    global unit_tables_frame
    if unit_tables_frame is None or not _widget_exists(unit_tables_frame):
        unit_tables_frame = ttk.Frame(root)
        unit_tables_frame.grid(
            row=4, column=0, rowspan=2, padx=10, pady=5, sticky="new",
        )
        unit_tables_frame._minimum_column_width = root.grid_columnconfigure(0)["minsize"]
    frame = ttk.LabelFrame(unit_tables_frame, text=title)
    font = tkfont.Font(root=root, family="DejaVu Sans", size=_scaled_font_size(10))
    values = [[" ".join(str(value).split()) for value in row] for row in table]
    widths = [
        int(np.ceil(max(font.measure(value) for value in cells) / font.measure("0"))) + 2
        for cells in zip(*values)
    ]
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            entry = ttk.Entry(frame, width=widths[column_index], font=font)
            entry.insert(END, value)
            entry.configure(state="readonly")
            entry.grid(row=row_index, column=column_index, sticky="ew")
    frame._value_font = font
    frame.grid(row=0, column=column, padx=(0, 8) if column == 0 else 0, sticky="nw")
    frame.bind("<Configure>", _fit_review_tables)
    return frame


def MakeTable(table):
    global frame_table
    if frame_table.winfo_exists() == 1:
        frame_table.destroy()
    frame_table = _make_review_table(table, "UnitData", 0)


# get table data #ADD STABILTY - prob of unit with itself accros cv
def get_table_data(UnitA, UnitB, CV):
    template = [
        ["Unit", "A", "B"],
        ["Avg Centroid", "tmp", "tmp"],
        ["Amplitude", "tmp", "tmp"],
        ["Spatial Decay", "tmp", "tmp"],
        ["Units Matches", "tmp", "tmp"],
        ["Stability", "tmp", "tmp"],
    ]

    total_rows = len(template)
    total_columns = len(template[0])
    unit_idx_tmp = [0, UnitA, UnitB]
    table = template

    if CV == "Avg":
        for i in range(total_rows):
            for j in range(total_columns):
                if j == 0:
                    continue
                if i == 0:
                    table[i][j] = str(unit_idx_tmp[j])
                if i == 1:
                    table[i][j] = str(np.round(avg_centroid_avg[:, unit_idx_tmp[j]], 2))
                if i == 2:
                    table[i][j] = str(np.round(amplitude_avg[unit_idx_tmp[j]], 2))
                if i == 3:
                    table[i][j] = str(np.round(spatial_decay_avg[unit_idx_tmp[j]], 3))
                if i == 4:
                    table[i][j] = (
                        str(
                            np.argwhere(
                                (
                                    within_session[unit_idx_tmp[j], :]
                                    * output_avg[unit_idx_tmp[j], :]
                                )
                                > match_threshold
                            )
                        )
                        .replace("[", "")
                        .replace("]", "")
                    )
                if i == 5:
                    table[i][j] = str(
                        np.round(output_GUI[0][unit_idx_tmp[j], unit_idx_tmp[j]], 3)
                    )

    else:
        for i in range(total_rows):
            for j in range(total_columns):
                if j == 0:
                    continue
                if i == 0:
                    table[i][j] = str(unit_idx_tmp[j])
                if i == 1:
                    table[i][j] = str(
                        np.round(avg_centroid[:, unit_idx_tmp[j], CV[j - 1]], 2)
                    )
                if i == 2:
                    table[i][j] = str(
                        np.round(amplitude[unit_idx_tmp[j], CV[j - 1]], 2)
                    )
                if i == 3:
                    table[i][j] = str(
                        np.round(spatial_decay[unit_idx_tmp[j], CV[j - 1]], 3)
                    )
                if i == 4:
                    table[i][j] = (
                        str(
                            np.argwhere(
                                (
                                    within_session[unit_idx_tmp[j], :]
                                    * output_GUI[CV[0]][unit_idx_tmp[j], :]
                                )
                                > match_threshold
                            )
                        )
                        .replace("[", "")
                        .replace("]", "")
                    )
                if i == 5:
                    table[i][j] = str(
                        np.round(output_GUI[0][unit_idx_tmp[j], unit_idx_tmp[j]], 3)
                    )

    return table


def get_unit_score_table(UnitA, UnitB, CVoption):

    table = [["tmp"] * 2 for i in range((len(scores_to_include_avg) + 1))]

    table[0] = ["Score", f"{UnitA} and {UnitB}"]

    if CVoption == -1:
        for i in range(len(scores_to_include_avg)):
            for j in range(2):
                if j == 0:
                    table[i + 1][j] = list(scores_to_include_avg.keys())[i]
                else:
                    table[i + 1][j] = str(
                        np.round(
                            scores_to_include_avg[
                                list(scores_to_include_avg.keys())[i]
                            ][UnitA, UnitB],
                            3,
                        )
                    )

    else:
        for i in range(len(scores_to_include_avg)):
            for j in range(2):
                if j == 0:
                    table[i + 1][j] = list(scores_to_include_GUI[CVoption].keys())[i]
                else:
                    table[i + 1][j] = str(
                        np.round(
                            scores_to_include_GUI[CVoption][
                                list(scores_to_include_GUI[CVoption].keys())[i]
                            ][UnitA, UnitB],
                            3,
                        )
                    )

    return table


def make_unit_score_table(table):
    global score_table
    if score_table.winfo_exists() == 1:
        score_table.destroy()
    score_table = _make_review_table(table, "UM Scores", 1)


def plot_avg_waveforms(UnitA, UnitB, CV):
    global avg_waveform_plot
    if avg_waveform_plot.winfo_exists() == 1:
        avg_waveform_plot.destroy()

    fig = Figure(figsize=_scaled_figsize(5.5, 2.5), dpi=100)
    fig.patch.set_facecolor("#33393b")

    plt1 = fig.add_subplot(111)
    # plt1.spines[["left", "bottom"]].set_position(("data", 0))
    plt1.spines[["bottom"]].set_position(("data", 0))
    plt1.spines[["top", "right"]].set_visible(False)
    plt1.patch.set_facecolor("#2d2d2d")
    plt1.xaxis.set_label_coords(0.9, 0)

    if CV == "Avg":
        plt1.plot(
            avg_waveform_avg[:, UnitA],
            color=UNIT_A_COLOR,
            label=f"Unit A ({UnitA})",
        )
        plt1.plot(
            avg_waveform_avg[:, UnitB],
            color=UNIT_B_COLOR,
            label=f"Unit B ({UnitB})",
        )
        plt1.set_xlabel("Time (ms)")
        plt1.set_ylabel("Amplitude (µV)")
        # plt1.set_xlim(left = 0)
        # plt1.set_xticks([])

    else:
        plt1.plot(
            avg_waveform[:, UnitA, CV[0]],
            color=UNIT_A_COLOR,
            label=f"Unit A ({UnitA})",
        )
        plt1.plot(
            avg_waveform[:, UnitB, CV[1]],
            color=UNIT_B_COLOR,
            label=f"Unit B ({UnitB})",
        )
        plt1.set_xlabel("Time (ms)")
        plt1.set_ylabel("Amplitude (µV)")
        # plt1.set_xlim(left = 0)

    avg_waveform_plot = FigureCanvasTkAgg(fig, master=root)
    avg_waveform_plot.draw()
    avg_waveform_plot = avg_waveform_plot.get_tk_widget()
    avg_waveform_plot.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)


def plot_trajectories(UnitA, UnitB, CV):
    global trajectory_plot
    if trajectory_plot.winfo_exists() == 1:
        trajectory_plot.destroy()

    fig = Figure(
        figsize=_scaled_figsize(4, 4),
        dpi=100,
        layout="constrained",
    )
    fig.patch.set_facecolor("#33393b")

    plt2 = fig.add_subplot(111)
    plt2.patch.set_facecolor("#2d2d2d")
    plt2.set_aspect("auto")
    plt2.spines[["right", "top"]].set_visible(False)

    if CV == "Avg":
        # AM not doing a time averaged WaveIDX (where you fins goodtimepoints), will just uses CV 0 for both
        plt2.plot(
            avg_waveform_per_tp_avg[1, UnitA, wave_idx[UnitA, :, 0].astype(bool)],
            avg_waveform_per_tp_avg[2, UnitA, wave_idx[UnitA, :, 0].astype(bool)],
            color=UNIT_A_COLOR,
            label=f"Unit A ({UnitA})",
        )
        plt2.scatter(
            avg_centroid_avg[1, UnitA],
            avg_centroid_avg[2, UnitA],
            c=UNIT_A_COLOR,
        )

        plt2.plot(
            avg_waveform_per_tp_avg[1, UnitB, wave_idx[UnitB, :, 0].astype(bool)],
            avg_waveform_per_tp_avg[2, UnitB, wave_idx[UnitB, :, 0].astype(bool)],
            color=UNIT_B_COLOR,
            label=f"Unit B ({UnitB})",
        )
        plt2.scatter(
            avg_centroid_avg[1, UnitB],
            avg_centroid_avg[2, UnitB],
            c=UNIT_B_COLOR,
        )

        plt2.set_xlabel(r"X position ($\mu$m)")
        plt2.set_ylabel(r"Y position ($\mu$m)")

    else:
        plt2.plot(
            avg_waveform_per_tp[
                1, UnitA, wave_idx[UnitA, :, CV[0]].astype(bool), CV[0]
            ],
            avg_waveform_per_tp[
                2, UnitA, wave_idx[UnitA, :, CV[0]].astype(bool), CV[0]
            ],
            color=UNIT_A_COLOR,
            label=f"Unit A ({UnitA})",
        )
        plt2.scatter(
            avg_centroid[1, UnitA, CV[0]],
            avg_centroid[2, UnitA, CV[0]],
            c=UNIT_A_COLOR,
        )

        plt2.plot(
            avg_waveform_per_tp[
                1, UnitB, wave_idx[UnitB, :, CV[1]].astype(bool), CV[1]
            ],
            avg_waveform_per_tp[
                2, UnitB, wave_idx[UnitB, :, CV[1]].astype(bool), CV[1]
            ],
            color=UNIT_B_COLOR,
            label=f"Unit B ({UnitB})",
        )
        plt2.scatter(
            avg_centroid[1, UnitB, CV[1]],
            avg_centroid[2, UnitB, CV[1]],
            c=UNIT_B_COLOR,
        )

        plt2.set_xlabel(r"X position ($\mu$m)")
        plt2.set_ylabel(r"Y position ($\mu$m)")

    trajectory_plot = FigureCanvasTkAgg(fig, master=root)
    trajectory_plot.draw()
    trajectory_plot = trajectory_plot.get_tk_widget()
    #    TrajectoryPlot.configure(bg = '#33393b')
    trajectory_plot.grid(
        row=3,
        column=1,
        padx=5,
        pady=5,
        sticky="nsew",
    )


def order_good_sites(good_sites, channel_pos, n_sessions):
    good_sites = np.asarray(good_sites).reshape(-1)
    # make it so it goes from biggest to smallest
    reordered_idx = np.argsort(-channel_pos[n_sessions][good_sites, 2])
    reordered_good_sites = good_sites[reordered_idx]

    # re-arange x-axis so it goes (smaller x, bigger x)
    for pair_start in range(0, len(reordered_good_sites) - 1, 2):
        pair = [pair_start, pair_start + 1]
        a, b = channel_pos[n_sessions][reordered_good_sites[pair], 1]

        if a > b:
            # swap order
            reordered_good_sites[pair] = reordered_good_sites[pair[::-1]]
    return reordered_good_sites


def nearest_channels(max_site, max_site_mean, channel_pos, clus_info, unit, CV):

    n_sessions = clus_info["session_id"][unit]
    if CV == "Avg":
        maxsite = max_site_mean[unit].squeeze()
        __, x, y = channel_pos[n_sessions][maxsite, :]

        good_x_sites = np.flatnonzero(
            np.logical_and(
                (x - 50 < channel_pos[n_sessions][:, 1]) == True,
                (channel_pos[n_sessions][:, 1] < x + 50) == True,
            )
        )
        y_values = channel_pos[n_sessions][good_x_sites, 2]

        y_dist_to_max_site = np.abs(y_values - channel_pos[n_sessions][maxsite, 2])
        good_sites = good_x_sites[np.argsort(y_dist_to_max_site)[:18]]
        reordered_good_sites = order_good_sites(good_sites, channel_pos, n_sessions)

    else:
        maxsite = max_site[unit, CV]
        __, x, y = channel_pos[n_sessions][maxsite, :]

        good_x_sites = np.flatnonzero(
            np.logical_and(
                (x - 50 < channel_pos[n_sessions][:, 1]) == True,
                (channel_pos[n_sessions][:, 1] < x + 50) == True,
            )
        )
        y_values = channel_pos[n_sessions][good_x_sites, 2]

        y_dist_to_max_site = np.abs(y_values - channel_pos[n_sessions][maxsite, 2])
        good_sites = good_x_sites[np.argsort(y_dist_to_max_site)[:18]]
        reordered_good_sites = order_good_sites(good_sites, channel_pos, n_sessions)

    return reordered_good_sites


def _add_raw_displacement_overlay(fig, unit_a, unit_b):
    global raw_displacement_axis

    accepted_pairs = _curated_accepted_pairs(
        automatic_match_pairs,
        is_match,
        not_match,
    )
    approved = _approved_displacement(
        unit_a,
        unit_b,
        accepted_pairs,
        raw_avg_centroid,
        clus_info,
    )
    selected = _selected_displacement(
        unit_a,
        unit_b,
        accepted_pairs,
        raw_avg_centroid,
        clus_info,
    )
    raw_displacement_axis = fig.add_axes(
        RAW_DISPLACEMENT_AXES_BOUNDS,
        zorder=100,
        facecolor="#24292b",
    )
    raw_displacement_axis.patch.set_alpha(0.96)

    vectors = [
        result
        for result in (approved, selected)
        if result["available"]
    ]
    scale_limit = max(
        (
            max(abs(result["dx"]), abs(result["dy"]))
            for result in vectors
        ),
        default=0,
    )
    scale_limit = max(scale_limit * 1.25, 1.0)
    raw_displacement_axis.axhline(0, color="#777777", linewidth=0.5)
    raw_displacement_axis.axvline(0, color="#777777", linewidth=0.5)
    reference = _automatic_displacement_reference(unit_a, unit_b)
    if reference is not None:
        heading = np.arctan2(reference[1], reference[0])
        radius = scale_limit * np.sqrt(2)
        for sign in (-1, 1):
            angle = heading + sign * np.deg2rad(DISPLACEMENT_ANGLE_THRESHOLD)
            boundary, = raw_displacement_axis.plot(
                [0, radius * np.cos(angle)], [0, radius * np.sin(angle)],
                linestyle="--", color="#E5BE63", linewidth=1, alpha=0.85,
                label=(
                    f"Automatic mean +/-{DISPLACEMENT_ANGLE_THRESHOLD} deg"
                    if sign == -1 else "_nolegend_"
                ),
            )
            boundary.set_gid(f"displacement_threshold_{sign}")

    status_lines = []
    if approved["available"]:
        raw_displacement_axis.annotate(
            "",
            xy=(approved["dx"], approved["dy"]),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": APPROVED_MATCH_COLOR,
                "linewidth": 2.0,
            },
        )
        mean_status = (
            f"Mean: {approved['magnitude']:.1f} um (n={approved['count']})"
        )
        if approved["excluded_nonfinite"]:
            mean_status += f"; excluded={approved['excluded_nonfinite']}"
        status_lines.append(mean_status)
    else:
        status_lines.append(f"Mean: {approved['message']}")

    if selected["available"]:
        raw_displacement_axis.annotate(
            "",
            xy=(selected["dx"], selected["dy"]),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#42A5F5",
                "linewidth": 1.6,
                "linestyle": "--",
            },
        )
        selected_status = (
            f"Selected: {selected['magnitude']:.1f} um"
            f" ({'approved' if selected['approved'] else 'not approved'})"
        )
        status_lines.append(selected_status)
    else:
        status_lines.append(f"Selected: {selected['message']}")

    raw_displacement_axis.set_xlim(-scale_limit, scale_limit)
    raw_displacement_axis.set_ylim(-scale_limit, scale_limit)
    raw_displacement_axis.set_aspect("equal", adjustable="box")
    raw_displacement_axis.set_title(
        "Raw A -> B (um)",
        fontsize=_scaled_font_size(7, minimum=7),
        color="#DDDDDD",
    )
    raw_displacement_axis.tick_params(
        labelsize=_scaled_font_size(6, minimum=6),
        colors="#DDDDDD",
    )
    raw_displacement_axis.plot(
        [],
        [],
        color=APPROVED_MATCH_COLOR,
        linewidth=2.0,
        label="Mean displacement",
    )
    raw_displacement_axis.plot(
        [],
        [],
        color="#42A5F5",
        linewidth=1.6,
        linestyle="--",
        label="Selected unit displacement",
    )
    legend = raw_displacement_axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.60, 0.96),
        bbox_transform=fig.transFigure,
        fontsize=_scaled_font_size(6, minimum=5),
        framealpha=0.9,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.2,
    )
    legend.get_frame().set_facecolor("#24292b")
    for text in legend.get_texts():
        text.set_color("#DDDDDD")
    status_text = raw_displacement_axis.text(
        0.60,
        0.86,
        "\n".join(status_lines),
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=_scaled_font_size(6, minimum=5),
        color="#DDDDDD",
    )
    status_text.set_gid("displacement_status")
    return raw_displacement_axis


def _refresh_raw_displacement_overlay():
    global raw_displacement_axis

    if raw_waveform_figure is None:
        return
    if raw_displacement_axis is not None:
        raw_displacement_axis.remove()
        raw_displacement_axis = None
    _add_raw_displacement_overlay(
        raw_waveform_figure,
        int(entry_a.get()),
        int(entry_b.get()),
    )
    if raw_waveform_canvas is not None:
        draw_idle = getattr(raw_waveform_canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()
        else:
            raw_waveform_canvas.draw()


def plot_raw_waveforms(unit_a, unit_b, CV):

    session_no_a = clus_info["session_id"][unit_a]
    global raw_waveform_plot
    global raw_waveform_figure
    global raw_waveform_canvas
    global raw_displacement_axis
    global raw_waveform_pair_lines, raw_waveform_view_key
    view_key = (unit_a, str(CV), id(waveform), id(channel_pos), gui_scale)
    if raw_waveform_view_key == view_key and _widget_exists(raw_waveform_plot):
        for channel, line in raw_waveform_pair_lines:
            values = (
                waveform[unit_b, :, channel].mean(axis=-1)
                if CV == "Avg" else waveform[unit_b, :, channel, CV[1]]
            )
            line.set_ydata(values.squeeze())
        if raw_displacement_axis is not None:
            raw_displacement_axis.remove()
        raw_displacement_axis = None
        _add_raw_displacement_overlay(raw_waveform_figure, unit_a, unit_b)
        raw_waveform_canvas.draw_idle()
        return
    if raw_waveform_plot.winfo_exists() == 1:
        raw_waveform_plot.destroy()
    raw_displacement_axis = None
    raw_waveform_pair_lines = []
    raw_waveform_view_key = view_key

    fig = Figure(figsize=_scaled_figsize(4, 8), dpi=100)
    raw_waveform_figure = fig
    fig.set_tight_layout(False)
    fig.patch.set_facecolor("#33393b")

    main_ax = fig.add_axes(RAW_WAVEFORM_AXES_BOUNDS)
    main_ax.set_facecolor("#2d2d2d")
    (
        main_ax_x_offset,
        main_ax_y_offset,
        main_ax_x_scale,
        main_ax_y_scale,
    ) = RAW_WAVEFORM_AXES_BOUNDS

    if CV == "Avg":
        good_channels = nearest_channels(
            max_site, max_site_mean, channel_pos, clus_info, unit_a, CV
        )

        # may want to change so it find this for both units and selects the most extreme arguments
        # however i dont think tis will be necessary
        sub_min_y = np.nanmin(waveform[unit_a, :, good_channels].mean(axis=-1))
        sub_max_y = np.nanmax(waveform[unit_a, :, good_channels].mean(axis=-1))

    else:
        good_channels = nearest_channels(
            max_site, max_site_mean, channel_pos, clus_info, unit_a, CV[0]
        )

        # may want to change so it find this for both units and selects the most extreme arguments
        # however i dont think this will be necessary
        sub_min_y = np.nanmin(waveform[unit_a, :, good_channels, CV[0]])
        sub_max_y = np.nanmax(waveform[unit_a, :, good_channels, CV[0]])

    good_channels = np.asarray(good_channels).reshape(-1)
    num_rows = int(np.ceil(len(good_channels) / 2))
    selected_positions = channel_pos[session_no_a][good_channels][:, [1, 2]]
    min_x, min_y = np.min(selected_positions, axis=0)
    max_x, max_y = np.max(selected_positions, axis=0)
    x_range = max_x - min_x
    y_range = max_y - min_y
    delta_x = x_range / 2 if x_range > 0 else 1
    delta_y = y_range / len(good_channels) if y_range > 0 else 1
    amplitude_range = np.abs(sub_min_y) + np.abs(sub_max_y)
    waveform_y_offset = (
        np.abs(sub_max_y) / amplitude_range / num_rows
        if amplitude_range > 0
        else 0
    )

    # make the main scatter positiose site as scatter with opacity
    main_ax.scatter(
        channel_pos[session_no_a][good_channels, 1],
        channel_pos[session_no_a][good_channels, 2],
        c="grey",
        alpha=0.3,
    )
    main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    main_ax.set_ylim(min_y - delta_y, max_y + delta_y)

    for channel_index, good_channel in enumerate(good_channels):
        i, j = divmod(channel_index, 2)
        waveform_axis_height = main_ax_y_scale / num_rows
        waveform_axis_bottom = (
            main_ax_y_offset
            + main_ax_y_scale
            * (i / num_rows - 1 / (2 * num_rows) + waveform_y_offset)
        )
        waveform_axis_bottom = np.clip(
            waveform_axis_bottom,
            main_ax_y_offset,
            main_ax_y_offset + main_ax_y_scale - waveform_axis_height,
        )
        # may need to change this positioning if units sizes are irregular
        if j == 0:
            # The peak in the waveform is not half way, so maths says the x axis should be starting at
            # 0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
            ax = fig.add_axes(
                [
                    main_ax_x_offset + main_ax_x_scale * 0.25,
                    waveform_axis_bottom,
                    main_ax_x_scale * 0.25,
                    waveform_axis_height,
                ]
            )
        else:
            ax = fig.add_axes(
                [
                    main_ax_x_offset + main_ax_x_scale * 0.75,
                    waveform_axis_bottom,
                    main_ax_x_scale * 0.25,
                    waveform_axis_height,
                ]
            )

        if CV == "Avg":
            ax.plot(
                waveform[unit_a, :, good_channel].mean(axis=-1).squeeze(),
                color=UNIT_A_COLOR,
            )
            partner_line, = ax.plot(
                waveform[unit_b, :, good_channel].mean(axis=-1).squeeze(),
                color=UNIT_B_COLOR,
                lw=0.8,
            )
        else:
            ax.plot(
                waveform[unit_a, :, good_channel, CV[0]].squeeze(),
                color=UNIT_A_COLOR,
            )
            partner_line, = ax.plot(
                waveform[unit_b, :, good_channel, CV[1]].squeeze(),
                color=UNIT_B_COLOR,
                lw=0.8,
            )
        raw_waveform_pair_lines.append((good_channel, partner_line))
        ax.set_ylim(sub_min_y, sub_max_y)
        ax.set_axis_off()

    _add_raw_displacement_overlay(fig, unit_a, unit_b)

    main_ax.spines.right.set_visible(False)
    main_ax.spines.top.set_visible(False)
    main_ax.set_xticks([min_x, max_x])
    main_ax.set_xlabel(
        "X position ($\mu$m)",
        size=_scaled_font_size(14),
    )
    main_ax.set_ylabel(
        "Y position ($\mu$m)",
        size=_scaled_font_size(14),
    )

    raw_waveform_canvas = FigureCanvasTkAgg(fig, master=root)
    raw_waveform_canvas.draw()
    raw_waveform_plot = raw_waveform_canvas.get_tk_widget()
    # RawWaveformPlot.configure(bg = '#33393b')

    raw_waveform_plot.grid(
        row=2, column=2, rowspan=4, padx=5, pady=5, sticky="nsew"
    )


def _set_histogram_y_limits(axis, *histograms):
    """Keep zero and every peak visible, expanding compressed distributions."""
    positive_curves = [
        values[np.isfinite(values) & (values > 0)]
        for values in (np.asarray(histogram[0]) for histogram in histograms)
    ]
    positive_curves = [values for values in positive_curves if values.size]
    if not positive_curves:
        axis.set_ylim(0, 1)
        return
    peak = max(values.max() for values in positive_curves)
    typical_peaks = [np.percentile(values, 90) for values in positive_curves]
    if any(values.max() > 10 * typical for values, typical in zip(
        positive_curves, typical_peaks
    )):
        axis.set_yscale("symlog", linthresh=min(typical_peaks), linscale=1)
        axis.set_ylabel("Density (symlog)", fontsize=_scaled_font_size(10))
    axis.set_ylim(0, peak * 1.08)


class _HistogramPanel:
    """Keep static histograms rasterized; redraw only current-pair markers."""

    def __init__(self, canvas, markers, key):
        self.canvas = canvas
        self.markers = markers
        self.key = key
        self.background = None
        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("resize_event", self._on_resize)

    def _on_resize(self, event):
        self.background = None

    def _on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        for marker in self.markers:
            marker.axes.draw_artist(marker)

    def update(self, values):
        for marker, value in zip(self.markers, values):
            marker.set_xdata([value, value])
        if self.background is None:
            self.canvas.draw_idle()
        else:
            self.canvas.restore_region(self.background)
            for marker in self.markers:
                marker.axes.draw_artist(marker)
            self.canvas.blit(self.canvas.figure.bbox)


def plot_histograms(hist_names, hist, hist_matched, scores_to_include, unit_a, unit_b):
    global hist_plot
    global histogram_panel
    key = (id(scores_to_include), tuple(hist_names), id(hist), id(hist_matched))
    values = [scores_to_include[name][unit_a, unit_b] for name in hist_names]
    if histogram_panel is not None and histogram_panel.key == key and _widget_exists(hist_plot):
        histogram_panel.update(values)
        return
    if hist_plot.winfo_exists() == 1:
        hist_plot.destroy()

    fig = Figure(
        figsize=_scaled_figsize(6, 8),
        dpi=100,
        layout="constrained",
    )
    fig.patch.set_facecolor("#33393b")
    axs = fig.subplots(3, 2, sharex="col")
    axs = axs.flat
    markers = []

    # Create title mapping
    title_mapping = {
        "amp_score": "Amplitude score",
        "spatial_decay_score": "Spatial decay score",
        "centroid_overlord_score": "C'oid overlord score",
        "centroid_dist": "C'oid distance score",
        "waveform_score": "Waveform score",
        "trajectory_score": "Trajectory score",
    }

    # loop over indexes..
    for i in range(len(hist)):
        axs[i].step(
            hist[i][1][:-1],
            hist[i][0],
            color=ALL_SCORES_COLOR,
            label="All scores" if i == 0 else "",
        )
        axs[i].step(
            hist_matched[i][1][:-1],
            hist_matched[i][0],
            color=APPROVED_MATCH_COLOR,
            label="Expected matches" if i == 0 else "",
        )
        # Use improved title from mapping
        plot_title = title_mapping.get(hist_names[i], hist_names[i])
        axs[i].set_title(plot_title, fontsize=_scaled_font_size(12))

        # Add ylabel
        axs[i].set_ylabel("Density", fontsize=_scaled_font_size(10))
        _set_histogram_y_limits(axs[i], hist[i], hist_matched[i])

        marker = axs[i].axvline(
            values[i],
            ls="--",
            color="white",
            label="Current match pair" if i == 0 else "",
            animated=True,
        )
        markers.append(marker)
        axs[i].set_facecolor("#2d2d2d")

    canvas = FigureCanvasTkAgg(fig, master=root)
    histogram_panel = _HistogramPanel(canvas, markers, key)
    canvas.draw()
    hist_plot = canvas.get_tk_widget()

    hist_plot.grid(
        row=2,
        column=3,
        rowspan=4,
        padx=5,
        pady=5,
        sticky="nsew",
    )
