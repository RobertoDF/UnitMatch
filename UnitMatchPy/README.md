# UnitMatchPy

This repository contains:
- **UnitMatchPy**: automatic matching of neurons across sessions (Python package).

## References

- **UnitMatch:** https://www.nature.com/articles/s41592-024-02440-1

## Versions

- **UnitMatchPy version:** `3.5.0` (from `pyproject.toml`)

## System requirements

UnitMatchPy can run on a standard computer with sufficient RAM (>32GB for large datasets).
This software is supported for Windows and macOS, and has been tested on Windows 11.

## Installation

`pip install` (including `pip install -e .`) installs into whatever Python environment your `pip` points to (system Python, a conda env, or a virtualenv). `pip` does **not** create or name environments.

We recommend using Anaconda/Miniconda (conda) to create an isolated environment first:

```bash
# Create a new environment (pick any name you like; example: UMPy)
conda create -n UMPy python=3.11 pip #(press y when prompted)
conda activate UMPy
```

### Option A: Install the released package (PyPI)

```bash
pip install UnitMatchPy
```

Optional extras (heavier dependencies used by some notebooks and integration with SpikeInterface):

```bash
pip install "UnitMatchPy[full,notebooks]"
```

### Option B: Install a local, editable copy (for development / modified code)

First, open a terminal and navigate to this folder (the one containing `pyproject.toml`). The `pip install -e` command must be run from here:

```bash
# Windows (PowerShell) - note; you may have to give writing access to the specific CONDA environment via Windows Security → Virus & threat protection → Ransomware protection → Manage ransomware protection. For example, if the below steps result in "could not create 'UnitMatchPy.egg-info'"

Look at Controlled folder access:
cd $HOME\Documents\GitHub\UnitMatch\UnitMatchPy

# macOS / Linux
cd ~/Documents/GitHub/UnitMatch/UnitMatchPy
```

```bash
pip install -e .
```

Optional extras (e.g. if you'd like to run the notebooks or integrate with SpikeInterface):

```bash
pip install -e ".[full,notebooks]"
```

## Demo notebooks

All demo notebooks are in `Demo Notebooks/`.

### Run UnitMatchPy 

To run UnitMatchPy, standard spike sorting data is needed (channel positions and extracted raw waveforms for each unit). Waveforms can be extracted externally (e.g. [BombCell](https://github.com/Julie-Fabre/bombcell)) or using the demo notebooks:
- `Demo Notebooks/extract_raw_data_demo.ipynb` (compressed `.cbin`/`.ch` or raw)
- `Demo Notebooks/extract_raw_data_demo_open_ephys.ipynb` (Open Ephys)
- `Demo Notebooks/UMPy_spike_interface_demo.ipynb` ([SpikeInterface](https://spikeinterface.readthedocs.io/en/latest/) workflow)

Example notebooks:
- `Demo Notebooks/UMPy_example.ipynb` (recommended starting point)
- `Demo Notebooks/UMPy_example_detailed.ipynb` (more modular / advanced)

The GUI is an optional step to curate and explore UnitMatch outputs; see `Demo Notebooks/GUI_Reference_Guide.md` for usage tips and shortcuts.

Both Tk unit tables show raw displacement magnitude in micrometers, the angle
from the automatic-match mean for the same probe and session pair, and event
response correlation. Unit A metrics refer to each row's listed Best B partner;
Unit B metrics refer to its pairing with the currently selected Unit A. Unit B
also shows threshold eligibility and both CV probabilities. Dashed rays show a
60-degree visual reference around the automatic mean, not the filter threshold.
Zero-length displacement or an undefined reference direction has no angle.

Both tables also show **Disp. consistency**, a post-hoc 0-100 displacement
similarity score based on the currently accepted reference, excluding pairs
sharing either candidate endpoint. It does not change UM probability, ranking,
or automatic matches. Hover over a value for the residual, normalized distance,
reference count, and scope; `n/a` includes an explanation. Fits are cached and
computed in a separate background worker, even without event data, and refresh
after manual decisions.

The **Low displacement consistency** toggle filters Unit A by the score of its
listed Best B, strictly below the adjustable cutoff (initially 20). It excludes
`n/a` scores and leaves every Unit B alternative accessible. Filtering is
asynchronous and refreshes after decisions. Column definitions and color meanings
are available from **Columns and review help**, rather than filling the main
window. Histogram backgrounds and raw-channel panels are reused during partner
navigation, and queued redraws are coalesced.

The default minimum is 20 usable reference pairs after exclusions; set
`displacement_min_references_in=20` in `gui.process_info_for_GUI(...)` to
configure it. Reference groups use session pair and probe, plus per-unit
`clus_info["shank_ids"]` when supplied. Without those IDs the tooltip explicitly
labels the reference as probe-only. See the
[method and limitations](../README.md#displacement-consistency).

`Event r` is the equal-weight mean of Pearson correlations between the two
units' trial-averaged PSTHs for shared event types, using the Event Viewer's
window and bin settings (initially 1 second before, 2 seconds after, 10 ms bins).
An event type needs at least two occurrences in each session and nonconstant,
finite profiles; `n/a` means no usable shared profiles, not zero correlation.
The values are computed asynchronously and cached; `...` means pending and
`error` reports a data-access failure in the notebook output.
These are review diagnostics, not additional matching criteria. Response
similarity alone does not establish identity, and using it to select matches
can bias subsequent analyses of functional stability or plasticity.

![Tk review and Event Viewer](../docs/images/unitmatch-review-with-events.png)

For a nonblocking IPython/Jupyter review, call
`gui.run_GUI(block=False, preserve_decisions=True)` after preparing the GUI
inputs. This enables the Tk event loop while keeping the kernel available.
Reopening an already open review focuses that window rather than creating a
duplicate. Saving curated analyzers remains a separate, explicit step.
`preserve_decisions=True` is the default: closing/reopening keeps decisions in
memory, not on disk. Use `preserve_decisions=False` only to deliberately start
a fresh review after closing the window. Accepting a new partner rejects
accepted competitors at either endpoint within the same session pair, not links
to other sessions.
