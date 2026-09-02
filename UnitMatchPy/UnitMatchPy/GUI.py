from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from matplotlib import rcParams
import os
import pickle


UNIT_A_COLOR = "#EB6534"
UNIT_B_COLOR = "#FBFAF8"
APPROVED_MATCH_COLOR = "#43A047"
ALL_SCORES_COLOR = "#C0E6DE"


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


def run_GUI():
    """
    This function runs the GUI, allowing the user to look at the result and manually curate the matches.

    Returns
    -------
    List
        lists for the manually curated matches and non-matches
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

    # Try to load pre-calculated ACG cache
    acg_cache = load_acg_cache()

    rcParams.update({"figure.autolayout": True})
    rcParams.update({"font.size": 14})
    rcParams.update({"font.family": "DejaVu Sans"})
    color = "white"
    rcParams["text.color"] = color
    rcParams["axes.labelcolor"] = color
    rcParams["xtick.color"] = color
    rcParams["ytick.color"] = color

    np.set_printoptions(suppress=True)
    is_match = []
    not_match = []
    root = Tk()

    def close_gui():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_gui)
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to fit the screen with some padding
    window_width = screen_width - 100
    window_height = screen_height - 100
    root.geometry(f"{window_width}x{window_height}+50+50")

    # Configure column weights to prevent plot squashing
    for i in range(9):  # Configure columns 0-8
        root.columnconfigure(i, weight=1)
    # Configure rows to expand properly
    for i in range(10):  # Configure rows 0-9
        root.rowconfigure(i, weight=1)

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
    s.configure(".", font=("DejaVu Sans", 12))
    s.configure("TLabel", font=("DejaVu Sans", 12))
    s.configure("TButton", font=("DejaVu Sans", 12))
    s.configure("TEntry", font=("DejaVu Sans", 10))
    s.configure("TCombobox", font=("DejaVu Sans", 10))
    s.configure("TCheckbutton", font=("DejaVu Sans", 12))
    s.configure("TRadiobutton", font=("DejaVu Sans", 12))
    s.configure("TLabelFrame.Label", font=("DejaVu Sans", 12, "bold"))

    root.title("UMPy - Manual Curation")
    # root.geometry('800x800')

    # Construct the file path to the icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GUI_icon.png")

    # Debugging print statements
    print(f"Icon path: {icon_path}")
    print(f"File exists: {os.path.exists(icon_path)}")

    # Load the icon
    try:
        icon = PhotoImage(file=icon_path)
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
    CV_tkinter = IntVar()
    CV_tkinter.set(0)
    label_cv = ttk.Label(entry_frame, text="Select the cv option")
    for i, option in enumerate(CV_options):
        RadioCV = ttk.Radiobutton(
            entry_frame,
            text=option[0],
            value=option[1],
            variable=CV_tkinter,
            command=update_unit_cv,
        ).grid(row=i + 1, column=0)

    # selecting the unit
    session_a = int(session_entry_a.get())
    session_b = int(session_entry_b.get())
    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = ttk.Combobox(
        entry_frame, values=get_unit_a_display_options(), width=18
    )
    entry_a.set(option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    entry_b.current(0)
    entry_a.bind("<<ComboboxSelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()

    session_entry_a.bind("<<ComboboxSelected>>", update_unit_entryA)
    session_entry_b.bind("<<ComboboxSelected>>", update_unit_entryB)

    # adding a button which swaps unit A and B
    swap_button = ttk.Button(entry_frame, text="Swap Units", command=swap_units)

    # Calculate the score histograms
    # for each CV pair
    # make global variable so the functions can access the histograms.
    global hist_names_avg
    global hist_names_12
    global hist_names_21
    global hist_avg
    global hist_12
    global hist_21
    global hist_matches_avg
    global hist_matches_12
    global hist_matches_21

    hist_names_avg, hist_avg, hist_matches_avg = get_score_histograms(
        scores_to_include_avg, (output_avg > match_threshold)
    )
    hist_names_12, hist_12, hist_matches_12 = get_score_histograms(
        scores_to_include_GUI[0], (output_GUI[0] > match_threshold)
    )
    hist_names_21, hist_21, hist_matches_21 = get_score_histograms(
        scores_to_include_GUI[1], (output_GUI[1] > match_threshold)
    )

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
    swap_button.grid(row=3, column=2, columnspan=2, sticky="WE")
    ######################################################################################

    # MatchButtons
    ######################################################################################
    match_button = ttk.Button(root, text="Set as Match", command=set_match)
    non_match_button = ttk.Button(root, text="Set as Non Match", command=set_not_match)
    pair_lookup_button = ttk.Button(
        root,
        text="Inspect Pair",
        command=open_pair_lookup,
    )

    # Toggle Plots
    ######################################################################################
    toggle_raw_val = BooleanVar()
    toggle_UM_score_val = BooleanVar()
    toggle_acg_val = BooleanVar()
    toggle_raw_val.set(False)
    toggle_UM_score_val.set(False)
    toggle_acg_val.set(False)
    toggle_raw_plot = ttk.Checkbutton(
        root, text="Hide Raw Data", variable=toggle_raw_val
    )
    toggle_UM_score_plot = ttk.Checkbutton(
        root, text="Hide UM Score Histograms", variable=toggle_UM_score_val
    )
    toggle_acg_plot = ttk.Checkbutton(
        root, text="Hide Autocorrelograms (ACGs)", variable=toggle_acg_val
    )

    # Set up Key-Board shortcuts
    root.bind_all("u", update)
    root.bind_all("<Return>", update)
    root.bind_all("q", set_match)
    root.bind_all("m", set_match)
    root.bind_all("e", set_not_match)
    root.bind_all("n", set_not_match)

    # Grid the units
    entry_frame.grid(row=2, column=0, pady=5, padx=5)
    match_button.grid(row=1, column=0, sticky="W", padx=10, pady=5)
    non_match_button.grid(row=1, column=1, sticky="W", padx=10, pady=5)
    toggle_UM_score_plot.grid(row=1, column=2, sticky="W", padx=10, pady=5)
    toggle_raw_plot.grid(row=1, column=3, sticky="W", padx=10, pady=5)
    toggle_acg_plot.grid(row=1, column=4, sticky="W", padx=10, pady=5)
    pair_lookup_button.grid(row=1, column=5, sticky="W", padx=10, pady=5)

    # Create visual legend plots outside plots
    global unit_legend_plot
    global hist_legend_plot
    create_unit_legend()
    create_hist_legend()

    unit_legend_plot.grid(row=0, column=1, columnspan=2, sticky="W", padx=10, pady=5)
    hist_legend_plot.grid(row=0, column=3, columnspan=4, sticky="W", padx=10, pady=5)

    # Configure grid weights to auto-adjust subpanels
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    entry_frame.grid_rowconfigure(0, weight=1)
    entry_frame.grid_columnconfigure(0, weight=1)

    update(None)
    match_idx = 0

    root.mainloop()

    # set default plot color back to black
    color = "black"
    rcParams["text.color"] = color
    rcParams["axes.labelcolor"] = color
    rcParams["xtick.color"] = color
    rcParams["ytick.color"] = color

    return is_match, not_match, matches_GUI


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
):
    """
    This function:
    1 - passes data to the GUI
    2 - processes the data so it is in a better form for the GUI
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

    output_threshold = np.zeros_like(output)
    output_threshold[output > match_threshold] = 1

    matches = np.argwhere(
        output_threshold == 1
    )  # need all matches including same session for GUI

    # Matrix [A, B] compares Unit A half 1 with Unit B half 2.
    # Its transpose is the reciprocal comparison, A half 2 with B half 1.
    output_GUI = [output, output.T]
    matches_GUI = [
        np.argwhere(output_GUI[0] > match_threshold),
        np.argwhere(output_GUI[1] > match_threshold),
    ]
    scores_to_include_GUI = [
        scores_to_include,
        {key: value.T for key, value in scores_to_include.items()},
    ]

    # getting avg CV data
    # for the Scores where can do (X + X.T)/2 and take upper triangular part
    total_score_avg = np.triu((total_score + total_score.T) / 2)

    scores_to_include_avg = {}
    for key, value in scores_to_include.items():
        scores_to_include_avg[key] = (value + value.T) / 2

    output_avg = (output_GUI[0] + output_GUI[1]) / 2
    cv_12_matches = output_GUI[0] > match_threshold
    cv_21_matches = output_GUI[1] > match_threshold
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

    matches_avg_part_1 = np.argwhere(output_avg > match_threshold)
    matches_avg_part_2 = np.argwhere(output_avg.T > match_threshold)
    matches_avg = np.unique(
        np.concatenate((matches_avg_part_1, matches_avg_part_2)), axis=0
    )

    # or an simply average over both CV
    amplitude_avg = np.mean(amplitude, axis=-1)
    spatial_decay_avg = np.mean(spatial_decay, axis=-1)
    avg_centroid_avg = np.mean(avg_centroid, axis=-1)
    avg_waveform_avg = np.mean(avg_waveform, axis=-1)
    avg_waveform_per_tp_avg = np.mean(avg_waveform_per_tp, axis=-1)


def get_ranked_unit_a_options(session_a, session_b):
    """Return one best-average partner per union-eligible Unit A."""
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
    if not options:
        raise ValueError(
            f"No above-threshold candidates connect sessions "
            f"{session_a} and {session_b}"
        )
    return options


def get_ranked_unit_b_options(unit_a, session_b):
    session_start = session_switch[session_b - 1]
    session_stop = session_switch[session_b]
    relative_order = np.argsort(
        -output_avg[unit_a, session_start:session_stop],
        kind="stable",
    )
    return (relative_order + session_start).tolist()


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


def color_unit_a_options(event=None):
    """Color Unit A candidates by the displayed automatic decision."""
    try:
        popdown = entry_a.tk.call("ttk::combobox::PopdownWindow", entry_a)
        listbox = f"{popdown}.f.l"
        for index, pair in enumerate(option_a):
            unit_a, unit_b = pair
            is_automatic_match = (unit_a, unit_b) in automatic_match_pairs
            foreground = APPROVED_MATCH_COLOR if is_automatic_match else "#E53935"
            entry_a.tk.call(
                listbox,
                "itemconfigure",
                index,
                "-foreground",
                foreground,
            )
    except TclError:
        pass

    try:
        unit_a = int(entry_a.get().split()[0])
        unit_b = int(entry_b.get())
        is_automatic_match = (unit_a, unit_b) in automatic_match_pairs
        style_name = (
            "AutomaticMatch.TCombobox"
            if is_automatic_match
            else "AutomaticNonMatch.TCombobox"
        )
        entry_a.configure(style=style_name)
    except (TclError, ValueError):
        pass


def enable_unit_a_review_colors():
    style = ttk.Style(root)
    style.configure("AutomaticMatch.TCombobox", foreground=APPROVED_MATCH_COLOR)
    style.configure("AutomaticNonMatch.TCombobox", foreground="#E53935")
    entry_a.configure(postcommand=color_unit_a_options)
    root.after_idle(color_unit_a_options)


def create_unit_legend():
    """Create visual legend for unit colors"""
    global unit_legend_plot

    fig = Figure(figsize=(6, 0.5), dpi=100)
    fig.patch.set_facecolor("#33393b")

    ax = fig.add_subplot(111)
    ax.set_facecolor("#33393b")

    # Draw legend lines and text
    ax.plot([0, 0.15], [0.5, 0.5], color=UNIT_A_COLOR, lw=3)
    ax.text(0.18, 0.5, "Unit A", color="white", fontsize=12, va="center")

    ax.plot([0.6, 0.75], [0.5, 0.5], color=UNIT_B_COLOR, lw=3)
    ax.text(0.78, 0.5, "Unit B", color="white", fontsize=12, va="center")

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis("off")

    unit_legend_plot = FigureCanvasTkAgg(fig, master=root)
    unit_legend_plot.draw()
    unit_legend_plot = unit_legend_plot.get_tk_widget()


def create_hist_legend():
    """Create visual legend for histogram colors"""
    global hist_legend_plot

    fig = Figure(figsize=(8, 0.5), dpi=100)
    fig.patch.set_facecolor("#33393b")

    ax = fig.add_subplot(111)
    ax.set_facecolor("#33393b")

    # Draw legend elements
    ax.plot([0, 0.1], [0.5, 0.5], color=ALL_SCORES_COLOR, lw=3)
    ax.text(0.12, 0.5, "All scores", color="white", fontsize=10, va="center")

    ax.plot(
        [0.35, 0.45],
        [0.5, 0.5],
        color=APPROVED_MATCH_COLOR,
        lw=3,
    )
    ax.text(0.47, 0.5, "Expected matches", color="white", fontsize=10, va="center")

    ax.plot([0.75, 0.85], [0.5, 0.5], "white", lw=2, linestyle="--")
    ax.text(0.87, 0.5, "Current pair", color="white", fontsize=10, va="center")

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis("off")

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
    fig = Figure(figsize=(3, 3), dpi=100)
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
        ax.set_xlabel("Time lag (ms)", fontsize=10, color="white")
        ax.set_ylabel("Rate (Hz)", fontsize=10, color="white")
        ax.set_title("Autocorrelograms", fontsize=12, color="white")
        ax.set_xlim(0, 50)  # 50ms
        if max_rate > 0:
            ax.set_ylim(0, max_rate * 1.1)

        # Add legend
        legend = ax.legend(fontsize=8, loc="upper right")
        legend.get_frame().set_facecolor("#33393b")
        legend.get_frame().set_alpha(0.8)
        for text in legend.get_texts():
            text.set_color("white")

        # Style the plot
        ax.tick_params(colors="white", labelsize=8)
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
            fontsize=12,
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
        columnspan=2,
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
    """
    Updates the GUI.
    """

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
        # plot histograms based of off the CV
        if CV_option == -1:
            plot_histograms(
                hist_names_avg,
                hist_avg,
                hist_matches_avg,
                scores_to_include_avg,
                unit_a,
                unit_b,
            )
        if CV_option == 0:
            plot_histograms(
                hist_names_12,
                hist_12,
                hist_matches_12,
                scores_to_include_GUI[CV_option],
                unit_a,
                unit_b,
            )
        if CV_option == 1:
            plot_histograms(
                hist_names_21,
                hist_21,
                hist_matches_21,
                scores_to_include_GUI[CV_option],
                unit_a,
                unit_b,
            )
    else:
        if hist_plot.winfo_exists() == 1:
            hist_plot.destroy()

    # Plot ACGs for both units if not hidden
    if toggle_acg_val.get() is False:
        plot_acgs(unit_a, unit_b)
    else:
        if "acg_plot" in globals() and acg_plot.winfo_exists() == 1:
            acg_plot.destroy()
    install_navigation_bindtags(root)


def _install_unit_b_popdown_navigation():
    try:
        popdown = str(root.tk.call("ttk::combobox::PopdownWindow", entry_b))
        listbox = f"{popdown}.f.l"
        bindtags = tuple(root.tk.splitlist(root.tk.call("bindtags", listbox)))
    except TclError:
        return
    if "UnitMatchNavigation" not in bindtags:
        root.tk.call(
            "bindtags",
            listbox,
            ("UnitMatchNavigation", *bindtags),
        )


def _close_unit_b_dropdown():
    try:
        root.tk.call("ttk::combobox::Unpost", entry_b)
    except TclError:
        pass


def select_unit_b(unit_id):
    entry_b.current(option_b.index(int(unit_id)))


def up_options_b_list(event):
    """
    moves up the option B list, chose the unit with th next highest probabilty of being a match with unit A.
    """
    global entry_frame
    global option_b
    global entry_b

    _close_unit_b_dropdown()
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

    _close_unit_b_dropdown()
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
    entry_b.configure(postcommand=_install_unit_b_popdown_navigation)
    entry_b.bind("<<ComboboxSelected>>", update)
    install_navigation_bindtags(root)


def install_navigation_bindtags(widget):
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

    # Keep track of the unit it was before as we dont want to change theunit viewed when changing the CV
    entry_a_tmp = int(entry_a.get())
    entry_b_tmp = int(entry_b.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = ttk.Combobox(
        entry_frame, values=get_unit_a_display_options(), width=18
    )
    entry_a.set(entry_a_tmp)
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    select_unit_b(entry_b_tmp)

    entry_a.bind("<<ComboboxSelected>>", update_units)
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
    EntryBtmp = int(entry_b.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = ttk.Combobox(
        entry_frame, values=get_unit_a_display_options(), width=18
    )
    entry_a.set(option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)
    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    select_unit_b(EntryBtmp if EntryBtmp in option_b else option_b[0])
    entry_a.bind("<<ComboboxSelected>>", update_units)
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
    entry_a_tmp = int(entry_a.get())

    if entry_a.winfo_exists() == 1:
        entry_a.destroy()
    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_a = get_ranked_unit_a_options(session_a, session_b)
    entry_a = ttk.Combobox(
        entry_frame, values=get_unit_a_display_options(), width=18
    )
    valid_unit_a_ids = {pair[0] for pair in option_a}
    entry_a.set(entry_a_tmp if entry_a_tmp in valid_unit_a_ids else option_a[0][0])
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    entry_b.current(0)
    entry_a.bind("<<ComboboxSelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()

    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)
    match_idx = 0

    update(event)


def update_units(event):
    """
    This function is called when a option in the Unit A dropdown is selected (is a pair of units)
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
    # The dropdown value is "UnitA (automatic match count) best-UnitB".
    tmpA = selected.split()[0]
    tmpB = selected.split()[-1]
    entry_a.set(tmpA)

    option_b = get_ranked_unit_b_options(int(tmpA), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    entry_b.current(0)
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

    session_b = int(session_entry_b.get())
    match_idx = (match_idx + 1) % len(option_a)
    tmp_a, tmp_b = option_a[match_idx]
    entry_a.set(tmp_a)

    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_b = get_ranked_unit_b_options(int(tmp_a), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    entry_b.current(0)
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

    session_b = int(session_entry_b.get())
    match_idx = (match_idx - 1) % len(option_a)
    tmp_a, tmp_b = option_a[match_idx]
    entry_a.set(tmp_a)

    if entry_b.winfo_exists() == 1:
        entry_b.destroy()

    option_b = get_ranked_unit_b_options(int(tmp_a), session_b)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    entry_b.current(0)
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
    entry_a = ttk.Combobox(
        entry_frame, values=get_unit_a_display_options(), width=18
    )
    entry_a.set(entry_b_tmp)
    option_b = get_ranked_unit_b_options(int(entry_a.get()), session_a_tmp)

    ##################
    entry_b = ttk.Combobox(entry_frame, values=option_b, width=10)
    select_unit_b(entry_a_tmp)

    entry_a.bind("<<ComboboxSelected>>", update_units)
    enable_unit_a_review_colors()
    bind_unit_b_navigation()
    entry_a.grid(row=2, column=1, columnspan=2, stick="WE", padx=5)
    entry_b.grid(row=2, column=3, columnspan=2, sticky="WE", padx=5)

    tmp_list = [int(entry_b_tmp), int(entry_a_tmp)]

    match_idx = option_a.index(tmp_list)
    update(None)


def get_score_histograms(scores_to_include, output_threshold):
    """
    Scores2Include is the dictionary of all scores.
    ProbThreshold is a nUnits*nUnits array where each index is 0 (Not Match) 1 (Match)
    """

    # are lsit of length 6 each list item is 2 np arrays(bins, values)
    hist_names = []
    hist = []
    hist_matches = []
    for key, values in scores_to_include.items():
        hist_names.append(key)
        hist.append(np.histogram(values, bins=100, density=True))
        hist_matches.append(
            np.histogram(values[output_threshold.astype(bool)], bins=100, density=True)
        )

    return hist_names, hist, hist_matches


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
            value=int(clus_info["session_indices"][row_index]) + 1
        )
        probe_variable = IntVar(
            value=int(clus_info["probe_numbers"][row_index])
        )
        unit_variable = StringVar(
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
                f"Unit A: probe{probe_a}, analyzer unit {analyzer_id_a} "
                f"(export ID {export_id_a})",
                f"Unit B: probe{probe_b}, analyzer unit {analyzer_id_b} "
                f"(export ID {export_id_b})",
            )
        else:
            identity_rows = (
                f"Unit A: export ID {export_id_a}",
                f"Unit B: export ID {export_id_b}",
            )
        original_id_label = ttk.LabelFrame(root, text="Unit identities")
        for row, identity_text in enumerate(identity_rows):
            identity_entry = ttk.Entry(original_id_label, width=58)
            identity_entry.insert(0, identity_text)
            identity_entry.configure(state="readonly")
            identity_entry.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        original_id_label.columnconfigure(0, weight=1)
    except IndexError as e:
        print(f"Error: {e}")
        original_id_label = ttk.Label(
            root, text="Error: Unit ID out of bounds", borderwidth=2, relief="groove"
        )

    original_id_label.grid(
        row=2,
        column=1,
        sticky="w",
        padx=(5, 15),
        ipadx=5,
        ipady=5,
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
        root,
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
        width=42,
    )
    bayes_label.grid(
        row=2,
        column=2,
        sticky="w",
        padx=(15, 5),
        ipadx=10,
        ipady=10,
    )


def set_match(event=None):
    global is_match
    global not_match
    unit_a = int(entry_a.get())
    unit_b = int(entry_b.get())

    pairs = [[unit_a, unit_b], [unit_b, unit_a]]
    not_match[:] = [pair for pair in not_match if pair not in pairs]
    for pair in pairs:
        if pair not in is_match:
            is_match.append(pair)
    add_probability_label(unit_a, unit_b, CV_tkinter.get() - 1)


def set_not_match(event=None):
    global is_match
    global not_match
    unit_a = int(entry_a.get())
    unit_b = int(entry_b.get())

    pairs = [[unit_a, unit_b], [unit_b, unit_a]]
    is_match[:] = [pair for pair in is_match if pair not in pairs]
    for pair in pairs:
        if pair not in not_match:
            not_match.append(pair)
    add_probability_label(unit_a, unit_b, CV_tkinter.get() - 1)


def MakeTable(table):
    global frame_table

    if frame_table.winfo_exists() == 1:
        frame_table.destroy()

    total_rows = len(table)
    total_columns = len(table[0])

    colors = ["black", UNIT_A_COLOR, UNIT_B_COLOR]
    frame_table = ttk.LabelFrame(root, text="UnitData")
    for i in range(total_rows):
        for j in range(total_columns):
            e = ttk.Entry(frame_table, width=20)
            e.insert(END, table[i][j])
            e.configure(state="readonly")
            e.grid(row=i, column=j)
    frame_table.grid(row=4, column=0, padx=10, pady=10, sticky="nw")


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
                                (within_session * output_avg)[unit_idx_tmp[j], :]
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
                                (within_session * output_GUI[CV[0]])[unit_idx_tmp[j], :]
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

    total_rows = len(table)
    total_columns = len(table[0])

    colors = ["black", "Purple"]
    score_table = ttk.LabelFrame(root, text="UM Scores")
    for i in range(total_rows):
        for j in range(total_columns):
            e = ttk.Entry(score_table, width=30)
            e.insert(END, table[i][j])
            e.configure(state="readonly")
            e.grid(row=i, column=j)

    score_table.grid(row=5, column=0, padx=10, pady=10, sticky="nw")


def plot_avg_waveforms(UnitA, UnitB, CV):
    global avg_waveform_plot
    if avg_waveform_plot.winfo_exists() == 1:
        avg_waveform_plot.destroy()

    fig = Figure(figsize=(3, 3), dpi=100)
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
    avg_waveform_plot.grid(row=3, column=0)


def plot_trajectories(UnitA, UnitB, CV):
    global trajectory_plot
    if trajectory_plot.winfo_exists() == 1:
        trajectory_plot.destroy()

    fig = Figure(figsize=(4, 4), dpi=100, layout="constrained")
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
        columnspan=2,
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


def plot_raw_waveforms(unit_a, unit_b, CV):

    session_no_a = clus_info["session_id"][unit_a]
    global raw_waveform_plot
    if raw_waveform_plot.winfo_exists() == 1:
        raw_waveform_plot.destroy()

    fig = Figure(figsize=(4, 6), dpi=100)
    fig.set_tight_layout(False)
    fig.patch.set_facecolor("#33393b")

    main_ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    main_ax.set_facecolor("#2d2d2d")
    main_ax_offset = 0.2
    main_ax_scale = 0.8

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
        # may need to change this positioning if units sizes are irregular
        if j == 0:
            # The peak in the waveform is not half way, so maths says the x axis should be starting at
            # 0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
            ax = fig.add_axes(
                [
                    main_ax_offset + main_ax_scale * 0.25,
                    main_ax_offset
                    + main_ax_scale
                    * (i / num_rows - 1 / (2 * num_rows) + waveform_y_offset),
                    main_ax_scale * 0.25,
                    main_ax_scale / num_rows,
                ]
            )
        else:
            ax = fig.add_axes(
                [
                    main_ax_offset + main_ax_scale * 0.75,
                    main_ax_offset
                    + main_ax_scale
                    * (i / num_rows - 1 / (2 * num_rows) + waveform_y_offset),
                    main_ax_scale * 0.25,
                    main_ax_scale / num_rows,
                ]
            )

        if CV == "Avg":
            ax.plot(
                waveform[unit_a, :, good_channel].mean(axis=-1).squeeze(),
                color=UNIT_A_COLOR,
            )
            ax.plot(
                waveform[unit_b, :, good_channel].mean(axis=-1).squeeze(),
                color=UNIT_B_COLOR,
                lw=0.8,
            )
        else:
            ax.plot(
                waveform[unit_a, :, good_channel, CV[0]].squeeze(),
                color=UNIT_A_COLOR,
            )
            ax.plot(
                waveform[unit_b, :, good_channel, CV[1]].squeeze(),
                color=UNIT_B_COLOR,
                lw=0.8,
            )
        ax.set_ylim(sub_min_y, sub_max_y)
        ax.set_axis_off()

    main_ax.spines.right.set_visible(False)
    main_ax.spines.top.set_visible(False)
    main_ax.set_xticks([min_x, max_x])
    main_ax.set_xlabel("X position ($\mu$m)", size=14)
    main_ax.set_ylabel("Y position ($\mu$m)", size=14)

    raw_waveform_plot = FigureCanvasTkAgg(fig, master=root)
    raw_waveform_plot.draw()
    raw_waveform_plot = raw_waveform_plot.get_tk_widget()
    # RawWaveformPlot.configure(bg = '#33393b')

    raw_waveform_plot.grid(
        row=3, column=3, columnspan=2, rowspan=4, padx=15, pady=25, ipadx=15
    )


def plot_histograms(hist_names, hist, hist_matched, scores_to_include, unit_a, unit_b):

    global hist_plot
    if hist_plot.winfo_exists() == 1:
        hist_plot.destroy()

    fig = Figure(figsize=(7, 6), dpi=100, layout="constrained")
    fig.patch.set_facecolor("#33393b")
    axs = fig.subplots(3, 2, sharex="col")
    axs = axs.flat

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
        axs[i].set_ylim(bottom=0)

        # Use improved title from mapping
        plot_title = title_mapping.get(hist_names[i], hist_names[i])
        axs[i].set_title(plot_title, fontsize=12)

        # Add ylabel
        axs[i].set_ylabel("% units", fontsize=10)

        axs[i].axvline(
            scores_to_include[hist_names[i]][unit_a, unit_b],
            ls="--",
            color="white",
            label="Current match pair" if i == 0 else "",
        )
        axs[i].set_facecolor("#2d2d2d")

    hist_plot = FigureCanvasTkAgg(fig, master=root)
    hist_plot.draw()
    hist_plot = hist_plot.get_tk_widget()

    hist_plot.grid(
        row=3,
        column=5,
        columnspan=4,
        rowspan=4,
        padx=5,
        pady=20,
        sticky="nsew",
    )
