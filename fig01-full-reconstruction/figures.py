"""Figure 1 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig01-full-reconstruction/figures.py

    fig01-a-raster          held-out stimulus, observed and unobserved neurons
    fig01-b-scatter         firing rates, observed | unobserved
    fig01-c-delta-scatter   perturbation: teacher vs student Δrate per neuron
    fig01-d-scaling-factors the six tied scaling factors against their true value

The per-neuron Fluctuation R² histogram was dropped on 2026-09-18: the raster and the
rate scatters already show how well individual neurons are matched.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. ``placeholder_figures/fig01-full-reconstruction/figures.py`` calls ``main`` here
with a watermark and fake CSVs, so content edits show up in both.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    delta_mean_inset,
    delta_rate_scatter,
    r2_title,
)
from common.style import (
    EXCITATORY,
    INHIBITORY,
    LEGEND_GREY,
    PAIR,
    SINGLE,
    TICK_SIZE,
    WIDE,
    apply_style,
    nice_max,
    rate_scatter,
    save,
    spike_raster,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig01"
RASTER_SECONDS = 3.0
PERTURBATION_CSV = "fig01_perturbation.csv"
GROUPS = {"observed": "Observed", "unobserved": "Unobserved"}


def raster(spikes):
    """(a) teacher and student spikes of a few neurons on a held-out stimulus."""
    neurons = (
        spikes[["neuron_id", "observed"]]
        .drop_duplicates()
        .sort_values(["observed", "neuron_id"], ascending=[False, True])
    )
    fig, ax = plt.subplots(figsize=WIDE)
    spike_raster(
        ax,
        spikes[spikes["time_s"] <= RASTER_SECONDS],
        list(neurons["neuron_id"]),
        RASTER_SECONDS,
        [
            GROUPS["observed"] if o else GROUPS["unobserved"]
            for o in neurons["observed"]
        ],
    )
    ax.set_title("Student vs Teacher Raster (Held-Out Stimulus)")
    fig.tight_layout()
    return fig


def scatters(rates, held_out):
    """(b) teacher vs student firing rate, observed beside unobserved."""
    max_rate = nice_max(rates[["teacher_rate_hz", "student_rate_hz"]].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=PAIR)
    for ax, (group, label) in zip(axes, GROUPS.items()):
        subset = rates[rates["observed"] == int(group == "observed")]
        rate_scatter(ax, subset, r2_title(label, held_out, group), max_rate)
        ax.title.set_fontsize(TICK_SIZE)
    fig.suptitle("Student vs Teacher Activity")
    fig.tight_layout()
    return fig


def scaling_factors(factors):
    """(d) The six tied scaling factors the student actually fits, against the truth.

    The true model is scaling factor 1.0 for every projection (the student's physiology
    is the teacher's), so the target is a line rather than a second axis, and the panel
    reads as recovery: how close six numbers get, and which ones stay degenerate.
    """
    order = sorted(factors["scaling_factor"].unique())
    fig, ax = plt.subplots(figsize=SINGLE)
    for x, name in enumerate(order):
        rows = factors[factors["scaling_factor"] == name]
        ratio = rows["value"] / rows["target"]
        presynaptic = name.split("_to_")[0]
        color = {"excitatory": EXCITATORY, "inhibitory": INHIBITORY}.get(
            presynaptic, LEGEND_GREY
        )
        ax.scatter(np.full(len(ratio), x), ratio, s=45, color=color, zorder=3)
        if len(ratio) > 1:
            ax.errorbar(
                x, ratio.mean(), yerr=ratio.std(), color="k", capsize=3, linewidth=1
            )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # The log scale's minor labels (3 x 10^0, 6 x 10^-1) collide with the ticks above.
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [name.replace("_to_", "\u2192").replace("_", " ") for name in order],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Learnt / True Scaling Factor")
    ax.set_title("Six Parameters, Against the Truth (--)")
    fig.tight_layout()
    return fig


def delta_scatter(summary, deltas):
    """(d) the intervention's effect: teacher vs student Δrate, one point per neuron."""
    fig, ax = plt.subplots(figsize=(SINGLE[0], SINGLE[0]))
    delta_rate_scatter(ax, deltas)
    inset = delta_mean_inset(ax, deltas)
    inset.tick_params(labelsize=TICK_SIZE - 5)
    inset.set_xticklabels(
        [label.get_text() for label in inset.get_xticklabels()],
        fontsize=TICK_SIZE - 5,
    )
    inset.set_title("Mean Δ: Teacher / Student", fontsize=TICK_SIZE - 5)
    ax.legend(loc="upper left", markerscale=3, frameon=True)
    rows = summary[summary["evaluation"] == "perturbation"]
    scores = []
    for metric, short in (
        ("delta_activity_r2", "ΔAct"),
        ("delta_fluctuation_r2", "ΔFlu"),
    ):
        values = []
        for cell_type, label in (("excitatory", "E"), ("inhibitory", "I")):
            m = rows[
                (rows["metric"] == metric)
                & (rows["group"] == "unobserved")
                & (rows["cell_type"] == cell_type)
            ]
            if not m.empty:
                values.append(
                    f"{label} {m['value'].mean():.2f} [{m['ceiling_value'].mean():.2f}]"
                )
        if values:
            scores.append(f"{short} R²: " + ", ".join(values))
    ax.set_title(
        "Inhibiting 25% of Unobserved I Cells\n" + "\n".join(scores),
        fontsize=TICK_SIZE,
    )
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    summary = pd.read_csv(data_dir / "fig01_summary.csv")
    rates = pd.read_csv(data_dir / "fig01_rates.csv")
    spikes = pd.read_csv(data_dir / "fig01_spikes.csv")
    # Written since 2026-09-18; absent from older CSV sets.
    factor_path = data_dir / "fig01_scaling_factors.csv"
    factors = pd.read_csv(factor_path) if factor_path.exists() else None
    seed = int(spikes["seed"].iloc[0])
    rates_seed = rates[rates["seed"] == seed]
    if "evaluation" in summary:
        summary["evaluation"] = summary["evaluation"].fillna("held_out")
    else:  # CSVs written before the perturbation panel
        summary["evaluation"] = "held_out"
    held_out = summary[summary["evaluation"] == "held_out"]
    # The perturbation CSV is absent when the teacher has no calibrated current.
    perturbation = data_dir / PERTURBATION_CSV
    deltas = pd.read_csv(perturbation) if perturbation.exists() else None
    if deltas is not None:
        deltas = deltas[deltas["seed"] == seed]
        if deltas.empty:
            deltas = None

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    output(raster(spikes), "a", "raster")
    output(scatters(rates_seed, held_out), "b", "scatter")
    if factors is not None:
        output(scaling_factors(factors), "d", "scaling-factors")

    # The perturbation panel is a separate file, so dropping it from the talk is
    # dropping one SVG.
    if deltas is not None:
        output(delta_scatter(summary, deltas), "c", "delta-scatter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
