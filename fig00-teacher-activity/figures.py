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
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from connectome_snns.visualization import RASTER_BAND_COLOR

from common.style import (
    EXCITATORY,
    INHIBITORY,
    INK,
    LEGEND_GREY,
    MARKERS,
    PAIR,
    RATE_MARKER_SIZE,
    RATE_MAX_HZ,
    RATE_TICK_HZ,
    REFERENCE_GREY,
    SINGLE,
    TEACHER,
    TICK_SIZE,
    WIDE,
    apply_style,
    clear_panels,
    save,
    scatter_legend,
)

FIGURE = "fig00"
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


def coding_schematic(schematic):
    """(a) The arithmetic of one odourant: baseline beside odourant 1."""
    baseline = float(schematic["baseline_hz"].iloc[0])
    odourants = schematic["odourant"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=PAIR, sharey=True)
    for ax, heights, title in (
        (axes[0], np.full(odourants.size, baseline), "Baseline"),
        (axes[1], schematic["odour_hz"].to_numpy(), "Odourant 1"),
    ):
        colors = [
            TEACHER if height > baseline else REFERENCE_GREY for height in heights
        ]
        ax.bar(odourants, heights, color=colors, width=0.7)
        ax.axhline(
            baseline,
            color=INK,
            linestyle="--",
            linewidth=1,
            label=f"Mean = {baseline:.0f} Hz",
        )
        ax.set_xticks(odourants)
        ax.set_xticklabels(odourants, fontsize=TICK_SIZE - 4)
        ax.set_xlabel("Odourant")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Input Firing Rate (Hz)")
    axes[0].set_ylim(0, schematic["odour_hz"].max() * 1.15)
    axes[1].legend(loc="upper right", frameon=True, fontsize=TICK_SIZE)
    fig.suptitle("Odourant Input Rates, Constructed at a Fixed Mean")
    fig.tight_layout()
    return fig


# ==========
# Response to the input conditions
# ==========


def condition_scatter(rates, x, y, x_label, y_label, title):
    """One rate scatter, linear over 0-RATE_MAX_HZ, coloured by cell type.

    Cells outside the window are left off rather than counted on the axis or piled on
    its edge, as in ``common.style.rate_scatter``.
    """
    fig, ax = plt.subplots(figsize=SINGLE)
    for cell_type, color, name in (
        ("inhibitory", INHIBITORY, "Inhibitory"),
        ("excitatory", EXCITATORY, "Excitatory"),
    ):
        subset = rates[rates["cell_type"] == cell_type]
        inside = (subset[x] <= RATE_MAX_HZ) & (subset[y] <= RATE_MAX_HZ)
        ax.scatter(
            subset.loc[inside, x],
            subset.loc[inside, y],
            s=RATE_MARKER_SIZE,
            alpha=0.45,
            color=color,
            marker=MARKERS[cell_type],
            label=name,
            rasterized=True,
        )
    ax.plot([0, RATE_MAX_HZ], [0, RATE_MAX_HZ], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, RATE_MAX_HZ)
    ax.set_ylim(0, RATE_MAX_HZ)
    ax.xaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.yaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    scatter_legend(ax, RATE_MARKER_SIZE)
    fig.tight_layout()
    return fig


# ==========
# Dynamics
# ==========


def raster(data):
    """(c) One trial's spikes, feedforward and recurrent, as the dashboard draws them.

    Laid out like ``plot_spike_trains`` inside ``create_activity_dashboard``
    (``connectome_snns/visualization``): feedforward rows at the bottom, then excitatory,
    then inhibitory; one eventplot with 0.6-long, 0.8-wide ticks; a pale band behind every
    other row; no y ticks, since a row's identity is its colour, not its index; and a
    framed patch legend in the upper right, populations in reverse order (the
    dashboard leaves it unframed; every legend in the talk is boxed). Colours are ours
    (2026-09-21): the dashboard's own red/blue for cell type agree with the scheme, and
    its feedforward grey becomes REFERENCE_GREY.
    """
    population = data["population"]
    n = population.size
    fig, ax = plt.subplots(figsize=WIDE)
    # Alternating single-row bands, as the dashboard shades its raster.
    for row in range(0, n, 2):
        ax.axhspan(row - 0.5, row + 0.5, color=RASTER_BAND_COLOR, zorder=0)
    ax.eventplot(
        [data["time_s"][data["row"] == row] for row in range(n)],
        lineoffsets=np.arange(n),
        linelengths=0.6,
        linewidths=0.8,
        colors=[POPULATION_COLORS[str(name)] for name in population],
        rasterized=True,
    )
    ax.set_xlim(0, float(data["duration_s"]))
    # The dashboard's own ylim is (-0.5, rows - 0.5) with the legend inside the upper
    # right, which here covered the top rows' spikes: two rows of headroom keep its
    # placement without the legend sitting on the data.
    ax.set_ylim(-0.5, n + 1.5)
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Network Spike Trains, {n} Neurons of One Trial (Feedforward + Recurrent)"
    )
    ax.legend(
        handles=[
            Patch(facecolor=POPULATION_COLORS[name], label=name.capitalize())
            for name in POPULATION_COLORS
            if name in set(population)
        ][::-1],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def neuron_traces(data):
    """(f) One neuron's membrane potential, its spikes, and the currents that move it.

    Follows the library's activity dashboard: spikes are drawn as vertical lines from
    threshold to 0 mV (the simulator resets the voltage, so the trace itself has no spike
    peak), and current is plotted **inward-positive**, so excitatory input goes up. The
    stored values are g(V - E_syn), which is negative for an excitatory synapse, hence
    the sign flip here.
    """
    time_s = data["time_s"]
    synapses = list(data["synapse"])
    current = -data["current_pa"]
    leak = -data["leak_current_pa"]
    threshold = float(data["threshold_mv"])
    fig, axes = plt.subplots(2, 1, figsize=PAIR, sharex=True)
    # Five thousand samples per line: the traces are rasterized so the SVG stays small,
    # as the dense scatters are elsewhere. Text and axes remain vector.
    axes[0].plot(time_s, data["voltage_mv"], color=INK, linewidth=0.8, rasterized=True)
    spike_times = data["spike_time_s"]
    if spike_times.size:
        axes[0].vlines(
            spike_times,
            threshold,
            0.0,
            color=INK,
            linewidth=0.8,
            rasterized=True,
        )
    axes[0].axhline(
        threshold, color=REFERENCE_GREY, linestyle="--", linewidth=1, label="Threshold"
    )
    axes[0].axhline(
        float(data["rest_mv"]),
        color=REFERENCE_GREY,
        linestyle=":",
        linewidth=1,
        label="Rest",
    )
    axes[0].set_ylabel("Membrane Potential (mV)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=TICK_SIZE)
    axes[0].set_title(
        f"Neuron {int(data['neuron_id'])} ({data['cell_type']!s}): "
        "Membrane Potential and Input Currents"
    )
    traces = []
    for label, color, members in PATHWAYS:
        columns = [synapses.index(name) for name in members]
        trace = current[:, columns].sum(axis=1)
        traces.append(trace)
        axes[1].plot(
            time_s, trace, color=color, linewidth=0.8, label=label, rasterized=True
        )
    axes[1].plot(
        time_s,
        leak,
        color=INK,
        linestyle=":",
        linewidth=0.8,
        label="Leak",
        rasterized=True,
    )
    axes[1].set_ylabel("Input Current (pA)\npositive = depolarising")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(time_s[0], time_s[-1])
    axes[1].set_ylim(*symmetric_limit(np.concatenate([*traces, leak])))
    thick_legend(
        axes[1],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=TICK_SIZE,
    )
    fig.tight_layout()
    return fig


def conductances(data):
    """(g) The same neuron's conductance, split excitatory / inhibitory / feedforward.

    The three groups and their y limits follow the library's
    ``plot_synaptic_conductances``: excitatory and feedforward share a limit taken from
    their 98th percentile and inhibition gets ten times it, so a rare transient cannot
    flatten every other trace. Peaks above the limit are therefore clipped, exactly as
    the dashboard clips them.
    """
    time_s = data["time_s"]
    names = [str(name) for name in data["synapse"]]
    conductance = data["conductance_ns"]
    groups = (
        (
            "Excitatory",
            [n for n in names if n.startswith("Recurrent") and "GABA" not in n],
        ),
        ("Inhibitory", [n for n in names if "GABA" in n]),
        ("Feedforward", [n for n in names if n.startswith("Feedforward")]),
    )
    shared = np.concatenate(
        [
            conductance[:, names.index(n)]
            for label, members in groups
            for n in members
            if label != "Inhibitory"
        ]
    )
    limit = nice_limit(np.percentile(shared, 98))
    limits = {"Excitatory": limit, "Feedforward": limit, "Inhibitory": limit * 10}
    fig, axes = plt.subplots(3, 1, figsize=PAIR, sharex=True)
    for ax, (label, members) in zip(axes, groups, strict=True):
        for name in members:
            color, linestyle = SYNAPSE_STYLES[name]
            ax.plot(
                time_s,
                conductance[:, names.index(name)],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                label=name,
                rasterized=True,
            )
        ax.set_xlim(time_s[0], time_s[-1])
        ax.set_ylim(0, limits[label])
        ax.set_ylabel(label)
        thick_legend(
            ax,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=TICK_SIZE,
        )
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(
        f"Synaptic Conductance (nS) of Neuron {int(data['neuron_id'])} "
        f"({data['cell_type']!s})"
    )
    fig.tight_layout()
    return fig


def ou_trajectories(data):
    """(b) The Ornstein-Uhlenbeck mixing coefficient of each odourant over one trial."""
    fig, ax = plt.subplots(figsize=SINGLE)
    colors = assembly_colors(data["assembly"].size)
    for column in range(data["assembly"].size):
        ax.plot(
            data["time_s"],
            data["mixing_weight"][:, column],
            color=colors[column],
            linewidth=0.8,
            alpha=0.85,
            rasterized=True,
        )
    ax.set_xlim(data["time_s"][0], data["time_s"][-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mixing Weight")
    ax.set_title("Odourant Mixing Coefficient of One Trial")
    fig.tight_layout()
    return fig


def assembly_series(data, key, y_label, title):
    """One line per assembly, in the shared assembly colours."""
    fig, ax = plt.subplots(figsize=SINGLE)
    colors = assembly_colors(data["assembly"].size)
    for column in range(data["assembly"].size):
        ax.plot(
            data["time_s"],
            data[key][:, column],
            color=colors[column],
            linewidth=0.8,
            alpha=0.85,
            rasterized=True,
        )
    ax.set_xlim(data["time_s"][0], data["time_s"][-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} (Gaussian σ = {float(data['smoothing_ms']):.0f} ms)")
    fig.tight_layout()
    return fig


def assembly_rates(data):
    """(c) Each assembly's excitatory population rate over the same trial."""
    return assembly_series(
        data,
        "rate_hz",
        "Firing Rate (Hz)",
        "Assembly Population Activity, Excitatory Cells",
    )


def synaptic_drive(drive):
    """(f) The share of excitatory synaptic drive each pathway delivers."""
    colors = {label: color for label, color, _ in PATHWAYS}
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.bar(
        drive["pathway"],
        drive["drive_fraction"],
        width=0.5,
        color=[colors.get(name, REFERENCE_GREY) for name in drive["pathway"]],
    )
    ax.set_ylabel("Fraction of Excitatory Synaptic Drive")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Excitatory Synaptic Drive by Pathway\n"
        f"({int(drive['n_trials'].iloc[0])} Trials x "
        f"{drive['seconds_per_trial'].iloc[0]:.0f} s)"
    )
    fig.tight_layout()
    return fig


# ==========
# Dimensionality
# ==========


def variance_explained(spectrum, summary):
    """(g) Cumulative variance explained, with the participation ratio marked."""
    pr = float(summary["participation_ratio"])
    n90 = int(summary["n_pcs_90pct_var"])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(spectrum["component"], spectrum["cumulative_fraction"], color=TEACHER)
    ax.axvline(pr, color=REFERENCE_GREY, linewidth=1.2, label=f"PR = {pr:.1f}")
    ax.axhline(0.9, color=LEGEND_GREY, linestyle="--", linewidth=1)
    ax.axvline(
        n90,
        color=LEGEND_GREY,
        linestyle="--",
        linewidth=1,
        label=f"90% at {n90} PCs",
    )
    ax.set_xscale("log")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(
        f"Variance Explained by the Teacher's PCs\n({int(summary['n_neurons'])} Neurons, "
        f"Gaussian σ = {summary['smoothing_sigma_ms']:.0f} ms)"
    )
    fig.tight_layout()
    return fig


def variance_spectrum(spectrum, summary):
    """(j) Variance fraction per component, the first SPECTRUM_COMPONENTS of them.

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
    ax.set_title(
        f"Variance Spectrum of the Teacher's PCs\n({int(summary['n_trials'])} Trials x "
        f"{summary['seconds_per_trial']:.0f} s)"
    )
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    output(
        coding_schematic(pd.read_csv(data_dir / "fig00_coding_schematic.csv")),
        "a",
        "coding-schematic",
    )
    assemblies = np.load(data_dir / "fig00_assemblies.npz")
    output(ou_trajectories(assemblies), "b", "ou-trajectories")
    output(assembly_rates(assemblies), "c", "assembly-rates")

    output(raster(np.load(data_dir / "fig00_raster.npz")), "d", "raster")
    output(
        condition_scatter(
            pd.read_csv(data_dir / "fig00_condition_rates.csv"),
            "odour_hz",
            "odour_repeat_hz",
            "Odourant 1 Firing Rate (Hz)",
            "Odourant 1, Repeated (Hz)",
            "Firing Rate per Neuron, One Odourant and Two Input Noise Seeds",
        ),
        "e",
        "rates-odour-repeat",
    )

    traces = np.load(data_dir / "fig00_traces.npz")
    output(neuron_traces(traces), "f", "neuron-traces")
    output(conductances(traces), "g", "conductances")
    output(
        synaptic_drive(pd.read_csv(data_dir / "fig00_drive.csv")), "h", "synaptic-drive"
    )

    spectrum = pd.read_csv(data_dir / "fig00_pca_spectrum.csv")
    summary = pd.read_csv(data_dir / "fig00_dimensionality.csv").iloc[0]
    output(variance_explained(spectrum, summary), "i", "variance-explained")
    output(variance_spectrum(spectrum, summary), "j", "variance-spectrum")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
