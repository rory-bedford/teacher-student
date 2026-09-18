"""Figure 3 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig03-observed-fraction/figures.py
    uv run python fig03-observed-fraction/figures.py --scatter-fractions 0.25 0.05 0.02

    fig03-a-curve                Fluctuation R² vs observed fraction, observed / unobserved
    fig03-b-scatter              unobserved firing rates at three observed fractions
    fig03-c-delta-fluctuation    perturbation: ΔFluctuation R² vs observed fraction

The x-range is whatever analysis.py reported (its ``REPORTED_FRACTIONS``): the sweep
starts at 2% observed, the first point that fails, because below ~5% the fit itself
collapses. Panel (b) shows rate scatters at three observed fractions: by default the
lowest fraction still within 10% of the ceiling (above threshold), the fraction closest
to half the ceiling (near), and the lowest fraction run (below). Pass
--scatter-fractions to choose them by hand once the curve is known.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panel instead.
``placeholder_figures/fig03-observed-fraction/figures.py`` calls ``main`` here with a
watermark and fake CSVs, so content edits show up in both.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import OBSERVED_COLOR, UNOBSERVED_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS, perturbation_sweep, r2_title
from common.style import (
    LEGEND_GREY,
    SINGLE,
    TICK_SIZE,
    TRIPLE,
    apply_style,
    ceiling,
    clear_panels,
    rate_scatter,
    save,
    sweep_legend,
    sweep_series,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig03"
#: Teacher dimensionality (PCA of smoothed spikes), pooled over all training trials.
#: A figure's own ``fig03_dimensionality.csv`` wins if present (the PLACEHOLDER build
#: estimates one).
TEACHER_DIMENSIONALITY = (
    HERE.parent / "generate-teacher-activity" / "teacher_dimensionality.csv"
)
DIMENSIONALITY_CSV = "fig03_dimensionality.csv"
N_NEURONS = 5000
GROUPS = {
    "observed": ("Observed", OBSERVED_COLOR),
    "unobserved": ("Unobserved", UNOBSERVED_COLOR),
}


def default_scatter_fractions(summary):
    rows = summary[
        (summary["group"] == "unobserved") & (summary["metric"] == "fluctuation_r2")
    ]
    stats = rows.groupby("obs_fraction")[["value", "ceiling_value"]].mean()
    normalised = stats["value"] / stats["ceiling_value"]
    fractions = sorted(stats.index)
    comfortable = [f for f in fractions if normalised[f] >= 0.9]
    above = comfortable[0] if comfortable else fractions[-1]
    near = float((normalised - 0.5).abs().idxmin())
    below = fractions[0]
    return [above, near, below]


def dimensionality_band(ax, dimensionality):
    """The teacher's dominant subspace, as a fraction of the population.

    The band the observed sample stops spanning is where the fit fails. The
    participation ratio (41 neurons, 0.8%) sits well inside the collapsed region and is
    not marked. An estimated CSV may carry only the 90% count, in which case the band
    collapses to that one line.
    """
    high = dimensionality["n_pcs_90pct_var"] / N_NEURONS
    low = (
        dimensionality["n_pcs_80pct_var"] / N_NEURONS
        if "n_pcs_80pct_var" in dimensionality
        else high
    )
    if low < high:
        ax.axvspan(low, high, color=LEGEND_GREY, alpha=0.12, linewidth=0)
        label = (
            f"Teacher PCs for 80-90% of Variance\n"
            f"({dimensionality['n_pcs_80pct_var']:.0f}-"
            f"{dimensionality['n_pcs_90pct_var']:.0f} Neurons)"
        )
    else:
        label = (
            f"Teacher PCs for 90% of Variance\n"
            f"({dimensionality['n_pcs_90pct_var']:.0f} Neurons)"
        )
    ax.axvline(high, color=LEGEND_GREY, linewidth=1, alpha=0.6)
    ax.text(
        high * 1.08,
        0.03,
        label,
        transform=ax.get_xaxis_transform(),
        fontsize=TICK_SIZE - 1,
        color=LEGEND_GREY,
    )


def neuron_axis(ax):
    top = ax.secondary_xaxis(
        "top", functions=(lambda f: f * N_NEURONS, lambda n: n / N_NEURONS)
    )
    top.set_xlabel("Observed Neurons")
    return top


def curve(summary, dimensionality):
    """(a) Fluctuation R² against observed fraction, observed and unobserved."""
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.25, SINGLE[1] * 1.1))
    for group, (_, color) in GROUPS.items():
        rows = summary[
            (summary["group"] == group) & (summary["metric"] == "fluctuation_r2")
        ]
        sweep_series(ax, rows, "obs_fraction", "fluctuation_r2", color)
        ceiling(
            ax,
            summary[
                (summary["group"] == group) & (summary["metric"] == "fluctuation_r2")
            ],
            "obs_fraction",
            color,
        )
    dimensionality_band(ax, dimensionality)
    ax.set_xscale("log")
    ax.set_ylim(min(0.0, summary["value"].min() - 0.05), 1.0)
    ax.set_xlabel("Observed Fraction")
    ax.set_ylabel("Fluctuation R²")
    neuron_axis(ax)
    sweep_legend(ax, {label: color for label, color in GROUPS.values()}, metrics=False)
    ax.set_title("How Few Neurons Need to Be Observed?", pad=12)
    fig.tight_layout()
    return fig


def scatters(summary, rates, fractions, seed):
    """(b) unobserved firing rates at three observed fractions."""
    subsets = [
        rates[
            np.isclose(rates["obs_fraction"], fraction)
            & (rates["seed"] == seed)
            & (rates["observed"] == 0)
        ]
        for fraction in fractions
    ]
    fig, axes = plt.subplots(1, 3, figsize=TRIPLE)
    for ax, fraction, subset in zip(axes, fractions, subsets):
        rows = summary[np.isclose(summary["obs_fraction"], fraction)]
        n_observed = int(rows["n_observed"].iloc[0])
        rate_scatter(
            ax,
            subset,
            r2_title(f"{n_observed} Observed", rows, "unobserved"),
        )
        ax.title.set_fontsize(TICK_SIZE)
    fig.suptitle("Unobserved Neurons, Student vs Teacher Activity")
    fig.tight_layout()
    return fig


def delta_sweep(summary, metric):
    """Perturbation panel: Δ R² of each non-targeted population against observed fraction."""
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.25, SINGLE[1]))
    perturbation_sweep(ax, summary, "obs_fraction", metric)
    # perturbation_sweep draws at the compact talk weights; the archived panels are
    # thicker (see common/style.py sweep_series).
    for line in ax.get_lines():
        line.set_linewidth(1.5)
        if line.get_marker() not in (None, "None", ""):
            line.set_markersize(5)
    ax.set_xscale("log")
    ax.set_xlabel("Observed Fraction")
    ax.set_ylabel(METRIC_LABELS[metric])
    neuron_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_title(
        f"{METRIC_LABELS[metric]}: Inhibiting 25% of Unobserved I Cells", pad=12
    )
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, scatter_fractions=None, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig03_summary.csv")
    rates = pd.read_csv(data_dir / "fig03_rates.csv")
    estimated = data_dir / DIMENSIONALITY_CSV
    dimensionality = pd.read_csv(
        estimated if estimated.exists() else TEACHER_DIMENSIONALITY
    ).iloc[0]

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    held_out = summary
    if "evaluation" in summary:
        held_out = summary[summary["evaluation"] == "held_out"]
    output(curve(held_out, dimensionality), "a", "curve")

    fractions = scatter_fractions or default_scatter_fractions(held_out)
    seed = int(rates["seed"].min())
    output(scatters(held_out, rates, fractions, seed), "b", "scatter")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(delta_sweep(summary, "delta_fluctuation_r2"), "c", "delta-fluctuation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--scatter-fractions", type=float, nargs=3, default=None)
    args = parser.parse_args()
    main(args.data, args.out_dir, args.scatter_fractions)
