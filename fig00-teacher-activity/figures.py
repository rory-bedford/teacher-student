"""Figure 0 — one SVG per panel from the files written by analysis.py.

    uv run python fig00-teacher-activity/figures.py

    fig00-a-coding-schematic        how an odourant is built: one group up, the rest down
    fig00-b-input-rates             feedforward input rates across the 20 odourants
    fig00-c-rates-odour-baseline    per-neuron rate, odourant 1 against baseline
    fig00-d-rates-odour-repeat      per-neuron rate, odourant 1 against the same
                                    odourant with a different Poisson seed
    fig00-e-raster                  spike raster of one trial, ordered by assembly
    fig00-f-assemblies              odourant mixing weights and assembly rates
    fig00-g-neuron-traces           one neuron's voltage and its synaptic currents
    fig00-h-conductances            the same neuron's conductance per synapse type
    fig00-i-integrated-conductance  integrated conductance per synapse type, per neuron
    fig00-j-synaptic-drive          feedforward against recurrent excitatory drive
    fig00-k-variance-explained      cumulative variance explained per PCA component
    fig00-l-variance-spectrum       variance fraction per PCA component

Panels run in the order the talk needs them: what the input codes (a, b), how the network
responds to it (c, d), its dynamics and synaptic budget (e-j), and the dimensionality of
the activity every other figure is fitted to (k, l).

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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
    TRUTH,
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
#: Assembly activity is 20 series, which no categorical palette of six colours can carry,
#: so both assembly panels are heatmaps on one sequential map.
#: Heatmaps need a sequential ramp, which the palette does not have: build one from the
#: scheme's own steel blue (white -> TRUTH) rather than importing an off-scheme colormap.
ASSEMBLY_CMAP = LinearSegmentedColormap.from_list(
    "teacher_blue", ["#ffffff", TRUTH, INK]
)


# ==========
# Input coding
# ==========


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


def input_rates(data):
    """(b) Distribution of the feedforward input rates, pooled over the 20 odourants."""
    odour = data["odour_hz"]
    baseline = float(data["baseline_hz"][0])
    n_patterns, n_neurons = odour.shape
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.hist(odour.ravel(), bins=30, color=TEACHER, label="Odourants")
    ax.axvline(
        baseline,
        color=REFERENCE_GREY,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline = {baseline:.0f} Hz",
    )
    ax.set_xlabel("Input Firing Rate (Hz)")
    ax.set_ylabel("Input Neurons")
    ax.set_title(
        f"Feedforward Input Rates\n({n_neurons} Neurons x {n_patterns} Odourants)"
    )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


# ==========
# Response to the three input conditions
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
    """(e) One trial's spikes for a sample of neurons, in assembly order."""
    assembly = data["assembly"]
    cell_type = data["cell_type"]
    fig, ax = plt.subplots(figsize=WIDE)
    # A band per alternate assembly, so the assembly blocks are readable without ticks.
    for index, value in enumerate(np.unique(assembly)):
        rows = np.flatnonzero(assembly == value)
        if index % 2 == 0:
            ax.axhspan(
                rows.min() - 0.5, rows.max() + 0.5, color=RASTER_BAND_COLOR, zorder=0
            )
    for name, color, label in (
        ("excitatory", EXCITATORY, "Excitatory"),
        ("inhibitory", INHIBITORY, "Inhibitory"),
    ):
        selected = np.isin(data["row"], np.flatnonzero(cell_type == name))
        ax.scatter(
            data["time_s"][selected],
            data["row"][selected],
            s=1.5,
            marker="|",
            color=color,
            label=label,
            rasterized=True,
        )
    ax.set_xlim(0, float(data["duration_s"]))
    ax.set_ylim(-0.5, cell_type.size - 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron (Assembly Order)")
    ax.grid(visible=False)
    ax.set_title(f"Teacher Spikes, {cell_type.size} Neurons of One Trial")
    ax.legend(
        handles=[
            Line2D([], [], color=color, linewidth=4, label=label)
            for _, color, label in (
                ("excitatory", EXCITATORY, "Excitatory"),
                ("inhibitory", INHIBITORY, "Inhibitory"),
            )
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    fig.tight_layout()
    return fig


def assemblies(data):
    """(f) The Ornstein-Uhlenbeck mixing weights above the assembly rates they drive."""
    time_s = data["time_s"]
    extent = (0.0, float(time_s[-1] + time_s[1]), data["assembly"].size - 0.5, -0.5)
    fig, axes = plt.subplots(2, 1, figsize=PAIR, sharex=True)
    for ax, values, label in (
        (axes[0], data["ou_weight"], "Mixing Weight"),
        (axes[1], data["rate_hz"], "Rate (Hz)"),
    ):
        image = ax.imshow(
            values.T, aspect="auto", origin="upper", extent=extent, cmap=ASSEMBLY_CMAP
        )
        ax.set_ylabel("Assembly")
        ax.grid(visible=False)
        fig.colorbar(image, ax=ax, label=label, pad=0.01)
    axes[0].set_title("Odourant Mixing Weights and Assembly Firing Rates of One Trial")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def neuron_traces(data):
    """(g) One neuron's membrane potential and the currents that move it."""
    time_s = data["time_s"]
    synapses = list(data["synapse"])
    current = data["current_pa"]
    fig, axes = plt.subplots(2, 1, figsize=PAIR, sharex=True)
    # Five thousand samples per line: the traces are rasterized so the SVG stays small,
    # as the dense scatters are elsewhere. Text and axes remain vector.
    axes[0].plot(time_s, data["voltage_mv"], color=INK, linewidth=0.8, rasterized=True)
    axes[0].axhline(
        float(data["threshold_mv"]),
        color=REFERENCE_GREY,
        linestyle="--",
        linewidth=1,
        label="Threshold",
    )
    axes[0].set_ylabel("Membrane Potential (mV)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=TICK_SIZE)
    axes[0].set_title(
        f"Neuron {int(data['neuron_id'])} ({data['cell_type']!s}): "
        "Membrane Potential and Input Currents"
    )
    for label, color, members in PATHWAYS:
        columns = [synapses.index(name) for name in members]
        axes[1].plot(
            time_s,
            current[:, columns].sum(axis=1),
            color=color,
            linewidth=0.8,
            label=label,
            rasterized=True,
        )
    axes[1].plot(
        time_s,
        data["leak_current_pa"],
        color=INK,
        linestyle=":",
        linewidth=0.8,
        label="Leak",
        rasterized=True,
    )
    axes[1].set_ylabel("Input Current (pA)\nnegative = inward")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(time_s[0], time_s[-1])
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
    """(h) The same neuron's synaptic conductance, one line per synapse type."""
    time_s = data["time_s"]
    fig, ax = plt.subplots(figsize=WIDE)
    for column, name in enumerate(data["synapse"]):
        color, linestyle = SYNAPSE_STYLES[str(name)]
        ax.plot(
            time_s,
            data["conductance_ns"][:, column],
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            label=str(name),
            rasterized=True,
        )
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conductance (nS)")
    ax.set_title(
        f"Synaptic Conductance of Neuron {int(data['neuron_id'])} "
        f"({data['cell_type']!s}) by Synapse Type"
    )
    thick_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=TICK_SIZE,
    )
    fig.tight_layout()
    return fig


def integrated_conductance(table):
    """(i) Integrated conductance per synapse type, for a sample of neurons."""
    synapses = [name for name in SYNAPSE_STYLES if name in table]
    positions = np.arange(len(table))
    width = 0.8 / len(synapses)
    fig, ax = plt.subplots(figsize=WIDE)
    for index, name in enumerate(synapses):
        color, linestyle = SYNAPSE_STYLES[name]
        ax.bar(
            positions + (index - (len(synapses) - 1) / 2) * width,
            table[name],
            width=width,
            color=color,
            # NMDA is the dashed line of the trace panels; in bars it is the open face.
            alpha=1.0 if linestyle == "-" else 0.45,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{neuron}\n{cell_type[0].upper()}"
            for neuron, cell_type in zip(table["neuron_id"], table["cell_type"])
        ],
        fontsize=TICK_SIZE - 2,
    )
    ax.set_xlabel("Neuron")
    ax.set_ylabel(f"Integrated Conductance (nS s, {table['seconds'].iloc[0]:.0f} s)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Integrated Conductance per Synapse Type, Sampled Neurons")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=TICK_SIZE
    )
    fig.tight_layout()
    return fig


def synaptic_drive(drive):
    """(j) The share of excitatory synaptic drive each pathway delivers."""
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
    """(k) Cumulative variance explained, with the participation ratio marked."""
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
    """(l) Variance fraction per component, log-log."""
    pr = float(summary["participation_ratio"])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.loglog(spectrum["component"], spectrum["variance_fraction"], color=TEACHER)
    ax.axvline(pr, color=REFERENCE_GREY, linewidth=1.2, label=f"PR = {pr:.1f}")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Variance Fraction")
    # Below ~1e-6 the spectrum is numerical noise of the eigensolver.
    ax.set_ylim(1e-6, 1)
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
    output(input_rates(np.load(data_dir / "fig00_input_rates.npz")), "b", "input-rates")

    rates = pd.read_csv(data_dir / "fig00_condition_rates.csv")
    output(
        condition_scatter(
            rates,
            "odour_hz",
            "baseline_hz",
            "Odourant 1 Firing Rate (Hz)",
            "Baseline Firing Rate (Hz)",
            "Firing Rate per Neuron, Odourant 1 against Baseline",
        ),
        "c",
        "rates-odour-baseline",
    )
    output(
        condition_scatter(
            rates,
            "odour_hz",
            "odour_repeat_hz",
            "Odourant 1 Firing Rate (Hz)",
            "Odourant 1, Repeated (Hz)",
            "Firing Rate per Neuron, One Odourant and Two Input Noise Seeds",
        ),
        "d",
        "rates-odour-repeat",
    )

    output(raster(np.load(data_dir / "fig00_raster.npz")), "e", "raster")
    output(assemblies(np.load(data_dir / "fig00_assemblies.npz")), "f", "assemblies")

    traces = np.load(data_dir / "fig00_traces.npz")
    output(neuron_traces(traces), "g", "neuron-traces")
    output(conductances(traces), "h", "conductances")
    output(
        integrated_conductance(
            pd.read_csv(data_dir / "fig00_conductance_integral.csv")
        ),
        "i",
        "integrated-conductance",
    )
    output(
        synaptic_drive(pd.read_csv(data_dir / "fig00_drive.csv")), "j", "synaptic-drive"
    )

    spectrum = pd.read_csv(data_dir / "fig00_pca_spectrum.csv")
    summary = pd.read_csv(data_dir / "fig00_dimensionality.csv").iloc[0]
    output(variance_explained(spectrum, summary), "k", "variance-explained")
    output(variance_spectrum(spectrum, summary), "l", "variance-spectrum")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
