"""Figure 0 — one SVG per panel from the files written by analysis.py.

    uv run python fig00-teacher-activity/figures.py

    fig00-a-coding-schematic        how an odourant is built: one group up, the rest down
    fig00-b-rates-odour-repeat      per-neuron rate, odourant 1 against the same
                                    odourant with a different Poisson seed
    fig00-c-raster                  spike raster of ten neurons of one trial
    fig00-d-neuron-traces           one neuron's voltage, its spikes and its currents
    fig00-e-conductances            the same neuron's conductance per synapse type
    fig00-f-synaptic-drive          feedforward against recurrent excitatory drive
    fig00-g-variance-explained      cumulative variance explained per PCA component
    fig00-h-variance-spectrum       variance fraction per PCA component

Panels run in the order the talk needs them: what the input codes (a), how the network
responds to it (b), its dynamics and synaptic budget (c-f), and the dimensionality of the
activity every other figure is fitted to (g, h).

Four panels of the first build were cut on 2026-09-21 as redundant: the input-rate
histogram (the schematic says it), the odourant-against-baseline scatter (the repeat
scatter carries the point), the assembly heatmaps, and the per-neuron integrated
conductance bars. Their analysis steps went with them.

Colour follows COLORSCHEME.txt. Synaptic pathways take the presynaptic population's
colour, as the scaling-factor panels do -- red from excitatory, blue from inhibitory, grey
from mitral -- and AMPA and NMDA are separated by line style, not by a new colour. Style
is the archived paper figures (``common/style.py``), sized to drop into the talk at 100%.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


from common.style import (
    EXCITATORY,
    INHIBITORY,
    INK,
    PAIR,
    REFERENCE_GREY,
    SINGLE,
    TEACHER,
    TICK_SIZE,
    apply_style,
    clear_panels,
    save,
)

FIGURE = "fig00"
#: Seconds of the traced neuron to draw, and the sub-bin the window is chosen by. A fast
#: cell packs a hundred spikes into 5 s, where they read as bands rather than as spikes,
#: and a cell that fires in bursts leaves most of a fixed window empty -- so the window is
#: the one whose spikes are spread over the most sub-bins (ties going to the most spikes).
TRACE_WINDOW_S = 2.0
#: Ticks every half second across the two-second window.
TRACE_TICK_S = 0.5
#: The current panel's fixed range, in pA.
CURRENT_LIMIT_PA = 250
TRACE_WINDOW_BIN_S = 0.5
#: Assembly panels tick every 5 s, so the 15 s trial ends on a tick.
ASSEMBLY_TICK_S = 5.0
#: The deviation panel ticks every 2 Hz and always reaches -2.
ASSEMBLY_RATE_TICK_HZ = 2.0
#: Each synapse type's colour is its presynaptic population's; AMPA and NMDA of one
#: population share it and are separated by line style.
SYNAPSE_STYLES = {
    "Recurrent AMPA": (EXCITATORY, "-"),
    "Recurrent NMDA": (EXCITATORY, "--"),
    "Recurrent GABA_A": (INHIBITORY, "-"),
    "Feedforward AMPA": (REFERENCE_GREY, "-"),
    "Feedforward NMDA": (REFERENCE_GREY, "--"),
}
#: The three pathways the current panel and the drive panel contrast, by presynaptic
#: population, and the synapse types each pools.
PATHWAYS = (
    ("Recurrent Excitatory", EXCITATORY, ("Recurrent AMPA", "Recurrent NMDA")),
    ("Recurrent Inhibitory", INHIBITORY, ("Recurrent GABA_A",)),
    ("Feedforward", REFERENCE_GREY, ("Feedforward AMPA", "Feedforward NMDA")),
)
#: The raster's rows, by population: the dashboard's grey feedforward, red excitatory and
#: blue inhibitory, in its own bottom-to-top order, with our colours.
POPULATION_COLORS = {
    "feedforward": REFERENCE_GREY,
    "excitatory": EXCITATORY,
    "inhibitory": INHIBITORY,
}
#: The scree plot stops here: beyond it the spectrum is a long unremarkable tail, and the
#: cumulative panel already carries where the variance actually runs out.
SPECTRUM_COMPONENTS = 100


# ==========
# Input coding
# ==========


def nice_limit(value):
    """The library's axis rule: round up to 1, 2 or 5 times a power of ten."""
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for step in (1.0, 2.0, 5.0, 10.0):
        limit = step * 10**exponent
        if limit >= value:
            return float(limit)
    return float(10 ** (exponent + 1))


def symmetric_limit(values, percentile=99.5):
    """Current limits from a percentile, so one transient cannot own the axis."""
    limit = nice_limit(np.percentile(np.abs(values), percentile))
    return -limit, limit


def assembly_colors(n):
    """Twenty assemblies need twenty colours, which the talk palette does not have, so
    both assembly panels use the same categorical map the library's dashboard uses."""
    return plt.colormaps["tab20"](np.linspace(0, 1, n))


def thick_legend(ax, **kwargs):
    """Legend whose line swatches are readable even though the traces are hairlines."""
    legend = ax.legend(**kwargs)
    for handle in legend.legend_handles:
        handle.set_linewidth(2.0)
    return legend


# ==========
# Dynamics
# ==========


def trace_legend(ax):
    """The trace panels' legend: same anchor and padding on every row, so they align."""
    return thick_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,  # flush with the top of the axes it belongs to
        frameon=True,
        fontsize=TICK_SIZE * 0.85,
        borderpad=0.35,
        labelspacing=0.3,
        handletextpad=0.5,
        handlelength=1.6,
    )


def trace_window_start(spike_times, duration_s):
    """The start of the TRACE_WINDOW_S window whose spikes are the most spread out.

    Scored by how many TRACE_WINDOW_BIN_S sub-bins contain a spike, ties going to the
    window with more spikes, so a bursting cell is not drawn over a silent stretch.
    """
    bins = round(TRACE_WINDOW_S / TRACE_WINDOW_BIN_S)
    best, chosen = (-1, -1), 0.0
    for start in np.arange(
        0.0, max(duration_s - TRACE_WINDOW_S, 0.0) + 1e-9, TRACE_WINDOW_BIN_S
    ):
        inside = spike_times[
            (spike_times >= start) & (spike_times < start + TRACE_WINDOW_S)
        ]
        counts = np.histogram(inside, bins=bins, range=(start, start + TRACE_WINDOW_S))[
            0
        ]
        score = (int((counts > 0).sum()), int(counts.sum()))
        if score > best:
            best, chosen = score, float(start)
    return chosen


def neuron_traces(data):
    """(c) One neuron: membrane potential with its spikes, input currents, conductances.

    The conductance panel was its own figure until 2026-09-23; it is the bottom row here,
    with every synapse type on one axes rather than split into excitatory / inhibitory /
    feedforward, so the three rows read as one neuron's story.

    Spikes are drawn as vertical lines from threshold to 0 mV (the simulator resets the
    voltage, so the trace has no spike peak), and current is inward-positive, so
    excitatory input goes up: the stored values are g(V - E_syn), negative for an
    excitatory synapse, hence the sign flip.
    """
    start = trace_window_start(data["spike_time_s"], float(data["time_s"][-1]))
    window = (data["time_s"] >= start) & (data["time_s"] <= start + TRACE_WINDOW_S)
    # The window is chosen for spike density, then re-zeroed: the axis reads 0 to 2 s
    # rather than 1.5 to 3.5, which looked like an arbitrary slice of a longer recording.
    time_s = data["time_s"][window] - start
    synapses = [str(name) for name in data["synapse"]]
    current = -data["current_pa"][window]
    conductance = data["conductance_ns"][window]
    threshold = float(data["threshold_mv"])
    fig, axes = plt.subplots(3, 1, figsize=(PAIR[0], PAIR[1] * 1.35), sharex=True)

    axes[0].plot(
        time_s,
        data["voltage_mv"][window],
        color=INK,
        linewidth=0.8,
        zorder=3,
        rasterized=True,
    )
    spikes = data["spike_time_s"]
    spikes = spikes[(spikes >= start) & (spikes <= start + TRACE_WINDOW_S)] - start
    if spikes.size:
        axes[0].vlines(
            spikes, threshold, 0.0, color=INK, linewidth=0.6, alpha=0.9, zorder=1
        )
    axes[0].axhline(
        threshold,
        color=REFERENCE_GREY,
        linestyle="--",
        linewidth=1.4,
        zorder=2,
        label=f"Threshold ({threshold:.0f} mV)",
    )
    axes[0].axhline(
        float(data["rest_mv"]),
        color=REFERENCE_GREY,
        linestyle=":",
        linewidth=1.4,
        zorder=2,
        label=f"Rest ({float(data['rest_mv']):.0f} mV)",
    )
    axes[0].set_ylim(top=0.0)  # spikes are drawn up to 0 mV, so that is the ceiling
    axes[0].set_ylabel("Membrane Potential\n(mV)")
    trace_legend(axes[0])
    axes[0].set_title(f"Neuron {int(data['neuron_id'])} ({data['cell_type']!s})")

    traces = []
    for label, color, members in PATHWAYS:
        columns = [synapses.index(name) for name in members]
        trace = current[:, columns].sum(axis=1)
        traces.append(trace)
        axes[1].plot(
            time_s, trace, color=color, linewidth=0.8, label=label, rasterized=True
        )
    axes[1].axhline(0, color=REFERENCE_GREY, linewidth=1, zorder=0)
    axes[1].set_ylabel("Input Current\n(pA)")
    # +-250 pA: the pathways run to about 200 and the axis keeps a little headroom.
    axes[1].set_ylim(-CURRENT_LIMIT_PA, CURRENT_LIMIT_PA)
    trace_legend(axes[1])

    for name in synapses:
        color, linestyle = SYNAPSE_STYLES[name]
        axes[2].plot(
            time_s,
            conductance[:, synapses.index(name)],
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            label=name,
            rasterized=True,
        )
    axes[2].set_ylabel("Conductance\n(nS)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, TRACE_WINDOW_S)
    axes[2].xaxis.set_major_locator(MultipleLocator(TRACE_TICK_S))
    # The inhibitory conductance is an order of magnitude above the rest; the axis is cut
    # at the 98th percentile of everything, as the library's dashboard cuts it.
    axes[2].set_ylim(0, nice_limit(np.percentile(conductance, 98)))
    trace_legend(axes[2])
    fig.tight_layout()
    fig.align_ylabels(axes)  # one left edge for all three y labels
    return fig


def ou_trajectories(data):
    """(a) The Ornstein-Uhlenbeck mixing coefficient of each odourant over one trial."""
    return assembly_series(
        data,
        "mixing_weight",
        "Mixing Weight",
        "Ornstein-Uhlenbeck Mixing Coefficient",
        legend_loc="center right",  # the top right is where a dominant input sits at ~1.0
    )


def assembly_series(data, key, y_label, title, label="Input", legend_loc="upper right"):
    """One line per assembly, in the shared assembly colours.

    The dominant odourant's own line is drawn heavier in both assembly panels, so the eye
    can follow one colour from the stimulus to the response.
    """
    fig, ax = plt.subplots(figsize=SINGLE)
    colors = assembly_colors(data["assembly"].size)
    dominant = [int(k) for k in np.atleast_1d(data["dominant"])]
    for column in range(data["assembly"].size):
        leading = column in dominant
        ax.plot(
            data["time_s"],
            data[key][:, column],
            color=colors[column],
            linewidth=1.8 if leading else 0.8,
            alpha=1.0 if leading else 0.65,
            zorder=3 if leading else 2,
            label=f"{label} {column}" if leading else None,
            rasterized=True,
        )
    thick_legend(
        ax,
        loc=legend_loc,
        frameon=True,
        fontsize=TICK_SIZE * 0.8,
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.4,
    )
    # To the whole trial rather than the last sample (14.995 s), so 15 s carries a tick.
    ax.set_xlim(0, float(np.ceil(data["time_s"][-1])))
    ax.xaxis.set_major_locator(MultipleLocator(ASSEMBLY_TICK_S))
    if data[key].min() >= 0:
        ax.set_ylim(bottom=0)  # a rate starts at zero
    else:
        # A deviation must show its negatives: ticks every 2 Hz, reaching at least -2.
        ax.yaxis.set_major_locator(MultipleLocator(ASSEMBLY_RATE_TICK_HZ))
        ax.set_ylim(bottom=min(-2.0, float(np.floor(data[key].min()))))
        # The zero line is added after this, and an axhline re-autoscales the axis unless
        # autoscaling is off -- which silently pulled the bottom back to -1.75.
        ax.autoscale(enable=False, axis="y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def assembly_rates(data):
    """(b) Each assembly's rate over the same trial, against its own all-trial mean.

    Assemblies differ in intrinsic rate by much more than a stimulus moves them (spread
    SD 1.13 Hz against a deviation SD of 0.41 Hz), so the raw rates would show which
    assembly is fastest rather than which odourant is on. Each assembly's mean over all
    50 trials is subtracted, which leaves the stimulus-driven part: across those trials
    the mixing coefficient correlates +0.51 with its own assembly's deviation, and the
    dominant odourant's assembly is the largest deviator in 62% of them.
    """
    deviation = {
        "time_s": data["time_s"],
        "assembly": data["assembly"],
        "dominant": data["dominant"],
        "smoothing_ms": data["smoothing_ms"],
        "deviation_hz": data["rate_hz"] - data["baseline_hz"][None, :],
    }
    fig = assembly_series(
        deviation,
        "deviation_hz",
        "Assembly Mean Deviation (Hz)",
        "Assembly Population Activity",
        label="Assembly",
    )
    fig.axes[0].axhline(0.0, color=REFERENCE_GREY, linewidth=1, zorder=1)
    return fig


def synaptic_drive(drive):
    """(d) The share of excitatory synaptic drive each pathway delivers, across trials.

    A bar per pathway with a whisker for the spread over the 50 trials. The whisker is
    the SD, and it is worth having: it shows the 2.4:1 split is not one trial's accident.
    (Unlike the performance panels, where three seeds are shown as points, there are 50
    trials here and the points would be a smear.)
    """
    colors = {label: color for label, color, _ in PATHWAYS}
    pathways = list(dict.fromkeys(drive["pathway"]))
    means = [
        drive.loc[drive["pathway"] == p, "drive_fraction"].mean() for p in pathways
    ]
    spreads = [
        drive.loc[drive["pathway"] == p, "drive_fraction"].std() for p in pathways
    ]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.bar(
        pathways,
        means,
        width=0.5,
        color=[colors.get(name, REFERENCE_GREY) for name in pathways],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.errorbar(
        pathways, means, yerr=spreads, fmt="none", ecolor=INK, capsize=5, linewidth=1.2
    )
    ax.set_ylabel("Fraction of Excitatory Synaptic Drive")
    ax.set_ylim(0, 1)
    ax.set_title("Excitatory Synaptic Drive by Pathway")
    fig.tight_layout()
    return fig


# ==========
# Dimensionality
# ==========


def variance_spectrum(spectrum, summary):
    """(i) Variance fraction per component, the first SPECTRUM_COMPONENTS of them.

    A scree plot on linear axes, as scree plots are drawn: the elbow is the point, and it
    is inside the first twenty components.
    """
    pr = float(summary["participation_ratio"])
    shown = spectrum[spectrum["component"] <= SPECTRUM_COMPONENTS]
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(shown["component"], shown["variance_fraction"], color=TEACHER)
    ax.axvline(pr, color=REFERENCE_GREY, linewidth=1.2, label=f"PR = {pr:.1f}")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Variance Fraction")
    ax.set_xlim(0, SPECTRUM_COMPONENTS)
    ax.set_ylim(0, float(shown["variance_fraction"].max()) * 1.1)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Variance Spectrum of the Teacher's PCs")
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)

    def output(fig, letter, slug, raster=False):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate, raster)

    assemblies = np.load(data_dir / "fig00_assemblies.npz")
    output(ou_trajectories(assemblies), "a", "ou-trajectories", raster=True)
    output(assembly_rates(assemblies), "b", "assembly-rates", raster=True)

    traces = np.load(data_dir / "fig00_traces.npz")
    output(neuron_traces(traces), "c", "neuron-traces", raster=True)
    output(
        synaptic_drive(pd.read_csv(data_dir / "fig00_drive.csv")), "d", "synaptic-drive"
    )

    spectrum = pd.read_csv(data_dir / "fig00_pca_spectrum.csv")
    summary = pd.read_csv(data_dir / "fig00_dimensionality.csv").iloc[0]
    output(variance_spectrum(spectrum, summary), "e", "variance-spectrum")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
