"""Figure 1 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig01-full-reconstruction/figures.py

    fig01-a-raster          held-out stimulus, observed and unobserved neurons
    fig01-b-scatter         firing rates, observed | unobserved
    fig01-c-delta-scatter   perturbation: teacher vs student Δrate per neuron
    fig01-d-delta-means     perturbation: mean Δrate per population, teacher vs student
    fig01-e-scaling-factors the six tied scaling factors, learnt / true, this figure's
                            runs beside the fully observed ones (recovery)

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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    PERTURBATION_TITLE,
    delta_rate_scatter,
    r2_title,
)
from common.style import (
    EXCITATORY,
    INHIBITORY,
    LEGEND_GREY,
    PAIR,
    REFERENCE_GREY,
    SINGLE,
    STUDENT,
    TEACHER,
    TICK_SIZE,
    WIDE,
    apply_style,
    clear_panels,
    rate_scatter,
    save,
    spike_raster,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig01"
RASTER_SECONDS = 3.0
PERTURBATION_CSV = "fig01_perturbation.csv"
GROUPS = {"observed": "Observed", "unobserved": "Unobserved"}
#: (value of the ``observed`` column, filled marker) per condition for panel (e). The
#: label carries the run's own observed fraction, read from the CSV, so it cannot drift
#: from the runs. "full" = Figure 2's fully observed runs, where every neuron is
#: teacher-forced and the six parameters are recoverable exactly.
CONDITIONS = (("full", True), ("partial", False))


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
    ax.set_title("Observed and Unobserved Spikes, Held-Out Stimulus")
    fig.tight_layout()
    return fig


def scatters(rates, held_out):
    """(b) teacher vs student firing rate, observed beside unobserved."""
    fig, axes = plt.subplots(1, 2, figsize=PAIR)
    for ax, (group, label) in zip(axes, GROUPS.items()):
        subset = rates[rates["observed"] == int(group == "observed")]
        rate_scatter(ax, subset, r2_title(label, held_out, group))
        ax.title.set_fontsize(TICK_SIZE)
    fig.suptitle("Firing Rates, Student vs Teacher, Held-Out Stimulus")
    fig.tight_layout()
    return fig


def condition_label(factors, condition):
    """ "Fully observed" / "10% observed", from the runs' own observed fraction."""
    rows = factors[factors["observed"] == condition]
    if rows.empty:
        return condition
    fractions = sorted(rows["observed_fraction"].unique())
    if fractions == [1.0]:
        return "Fully observed"
    return ", ".join(f"{fraction:.0%} observed" for fraction in fractions)


def scaling_factors(factors):
    """Learnt / true scaling factor per projection, one condition beside the other.

    The true model is scaling factor 1.0 for every projection (the student's physiology
    is the teacher's), so truth is a line rather than a second axis.
    """
    order = sorted(factors["scaling_factor"].unique())
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.15, SINGLE[1]))
    offsets = np.linspace(-0.12, 0.12, len(CONDITIONS))
    for (condition, filled), offset in zip(CONDITIONS, offsets):
        rows = factors[factors["observed"] == condition]
        for x, name in enumerate(order):
            values = rows[rows["scaling_factor"] == name]
            if values.empty:
                continue
            ratio = values["value"] / values["target"]
            # Coloured by the presynaptic population, so the three factors onto
            # inhibitory cells (all low) are visibly a group.
            color = {"excitatory": EXCITATORY, "inhibitory": INHIBITORY}.get(
                name.split("_to_")[0], REFERENCE_GREY
            )
            ax.scatter(
                np.full(len(ratio), x + offset),
                ratio,
                s=45,
                zorder=3,
                color=color if filled else "none",
                edgecolors=color,
                linewidths=1.4,
            )
            if len(ratio) > 1:
                ax.errorbar(
                    x + offset,
                    ratio.mean(),
                    yerr=ratio.std(),
                    color="k",
                    capsize=3,
                    linewidth=1,
                )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # The log scale's minor labels (3 x 10^0, 6 x 10^-1) collide with the ticks above.
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_ylim(0.2, 5.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [name.replace("_to_", "→").replace("_", " ") for name in order],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Learnt / True Scaling Factor")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                color=LEGEND_GREY,
                markerfacecolor=LEGEND_GREY if filled else "none",
                label=condition_label(factors, condition),
            )
            for condition, filled in CONDITIONS
        ],
        loc="upper left",
        frameon=True,
    )
    ax.set_title("Scaling Factors, Learnt vs True")
    fig.tight_layout()
    return fig


def delta_scores(summary):
    """ΔR² of the non-targeted unobserved populations, for a panel title."""
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
    return scores


def delta_scatter(summary, deltas):
    """(c) the intervention's effect: teacher vs student Δrate, one point per neuron."""
    fig, ax = plt.subplots(figsize=(PAIR[0] / 2, PAIR[1]))
    delta_rate_scatter(ax, deltas)
    ax.legend(loc="upper left", markerscale=4, frameon=True)
    ax.set_title(
        PERTURBATION_TITLE + "\n" + "\n".join(delta_scores(summary)),
        fontsize=TICK_SIZE,
    )
    fig.tight_layout()
    return fig


def delta_means(deltas):
    """(d) Mean Δrate per population, teacher beside student, on its own axes.

    Was an inset on panel (c), where it was unclear what the bars were; each population
    is now named on the axis and the teacher/student pairing is in the legend.
    """
    populations = (
        ("Targeted I", deltas["targeted"] == 1),
        (
            "Non-targeted I",
            (deltas["targeted"] == 0) & (deltas["cell_type"] == "inhibitory"),
        ),
        (
            "Non-targeted E",
            (deltas["targeted"] == 0) & (deltas["cell_type"] == "excitatory"),
        ),
    )
    fig, ax = plt.subplots(figsize=(SINGLE[0], SINGLE[1] * 0.8))
    width = 0.38
    for position, (label, mask) in enumerate(populations):
        subset = deltas[mask]
        for offset, column, color, name in (
            (-width / 2, "teacher_delta_rate_hz", TEACHER, "Teacher"),
            (width / 2, "student_delta_rate_hz", STUDENT, "Student"),
        ):
            ax.bar(
                position + offset,
                subset[column].mean(),
                width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=name if position == 0 else None,
            )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(range(len(populations)))
    ax.set_xticklabels([label for label, _ in populations])
    ax.set_ylabel("Mean Δrate (Hz)")
    ax.set_title("Mean Rate Change per Population, Student vs Teacher")
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig01_summary.csv")
    rates = pd.read_csv(data_dir / "fig01_rates.csv")
    spikes = pd.read_csv(data_dir / "fig01_spikes.csv")
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

    # The perturbation panel is a separate file, so dropping it from the talk is
    # dropping one SVG.
    if deltas is not None:
        output(delta_scatter(summary, deltas), "c", "delta-scatter")
        output(delta_means(deltas), "d", "delta-means")
    if factors is not None:
        output(scaling_factors(factors), "e", "scaling-factors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
