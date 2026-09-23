"""Figure 3 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig03-observed-fraction/figures.py
    uv run python fig03-observed-fraction/figures.py --scatter-fractions 0.25 0.05 0.02

    fig03-a-curve                Fluctuation R² vs observed fraction, observed / unobserved
    fig03-b-scatter              unobserved firing rates at three observed fractions
    fig03-c-delta-fluctuation    perturbation: ΔFluctuation R² vs observed fraction

The x-range is whatever analysis.py reported (its ``REPORTED_FRACTIONS``): the sweep starts
at 2% observed, and below ~25% observed a run either trains or collapses depending on the
observed draw, so the per-seed points matter more than the mean. Panel (b) shows rate scatters at three observed fractions: by default the
lowest fraction still within 10% of the ceiling (above threshold), the fraction closest
to half the ceiling (near), and the lowest fraction run (below). Pass
--scatter-fractions to choose them by hand once the curve is known.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panel instead.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    HELD_OUT_TITLE,
    METRIC_LABELS,
    PERTURBATION_LABEL,
    PERTURBATION_TITLE,
    plotted_performance_values,
    pool_populations,
)
from common.style import (
    LEGEND_GREY,
    OBSERVED,
    REFERENCE_GREY,
    SINGLE,
    TICK_SIZE,
    TRIPLE,
    UNOBSERVED,
    apply_style,
    ceiling,
    clear_panels,
    performance_axis,
    performance_limits,
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
    HERE.parent / "fig00-teacher-activity" / "fig00_dimensionality.csv"
)
DIMENSIONALITY_CSV = "fig03_dimensionality.csv"
N_NEURONS = 5000
GROUPS = {
    "observed": ("Observed", OBSERVED),
    "unobserved": ("Unobserved", UNOBSERVED),
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


def dimensionality_markers(ax, dimensionality):
    """Mark the teacher's dimensionality, and return the legend handle.

    The PCs carrying 90% of the variance, as a fraction of the 5000 neurons: it lands
    where the fit breaks down, which is the point of the panel, so it is named in the
    legend rather than annotated in the plot. The participation ratio (41 neurons, 0.8%)
    is not marked -- it sits below everything tested and would imply the opposite.
    """
    fraction = dimensionality["n_pcs_90pct_var"] / N_NEURONS
    ax.axvline(fraction, color=REFERENCE_GREY, linewidth=1.6, alpha=0.9)
    label = (
        f"90% of Teacher Variance\n({dimensionality['n_pcs_90pct_var']:.0f} Neurons)"
    )
    return [Line2D([], [], color=REFERENCE_GREY, linewidth=1.6, label=label)]


def observed_axis(ax, fractions):
    """The sweep's x axis: log, decreasing left to right, ticked in percentages."""
    ax.set_xscale("log")
    fractions = sorted(fractions)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{fraction * 100:g}%" for fraction in fractions])
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.invert_xaxis()
    ax.set_xlabel("Neurons Observed (% of Network)")
    neuron_axis(ax)


def neuron_axis(ax):
    top = ax.secondary_xaxis(
        "top", functions=(lambda f: f * N_NEURONS, lambda n: n / N_NEURONS)
    )
    top.set_xlabel(f"Neurons Observed (of {N_NEURONS:,})")
    return top


def limits(summary):
    """One y range for the sweep and its perturbation panel, as in Figures 4 and 5.

    Both panels report a Fluctuation R², so reading one against the other only works if
    they share a scale (2026-09-23).
    """
    return performance_limits(
        plotted_performance_values(summary, "unobserved", ["obs_fraction", "seed"])
    )


def curve(summary, dimensionality, ylim):
    """(a) Fluctuation R² against observed fraction, observed and unobserved."""
    fig, ax = plt.subplots(figsize=SINGLE)
    for group, (_, color) in GROUPS.items():
        rows = summary[
            (summary["group"] == group) & (summary["metric"] == "fluctuation_r2")
        ]
        sweep_series(
            ax,
            rows,
            "obs_fraction",
            "fluctuation_r2",
            color,
            seeds=True,
            errorbars=False,
        )
        ceiling(
            ax,
            summary[
                (summary["group"] == group) & (summary["metric"] == "fluctuation_r2")
            ],
            "obs_fraction",
            color,
        )
    markers = dimensionality_markers(ax, dimensionality)
    # Decreasing left to right (2026-09-21): the slide reads as neurons being taken away,
    # ending at the hard end of the sweep.
    ax.set_title(HELD_OUT_TITLE)
    observed_axis(ax, summary["obs_fraction"].unique())
    performance_axis(ax, ylim)
    ax.set_ylabel("Fluctuation R²")
    sweep_legend(
        ax,
        {label: color for label, color in GROUPS.values()},
        metrics=False,
        extra=markers,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    # No title (2026-09-21): the slide carries the claim.
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
            f"{fraction * 100:g}% Observed ({n_observed} Neurons)",
        )
        ax.title.set_fontsize(TICK_SIZE)
    fig.suptitle("Unobserved Neurons, Student vs Teacher Activity")
    fig.tight_layout()
    return fig


def delta_sweep(summary, metric, dimensionality, ylim):
    """(c) Perturbation Δ R² against observed fraction, cell types pooled.

    Built like panel (a) -- same reversed percentage axis, individual seeds, no error
    bars, the same 90%-variance marker and an inside legend. The non-targeted unobserved
    E and I populations are pooled (see ``common.plotting.pool_populations``): they
    differ only where both are already near zero.
    """
    rows = summary[
        (summary["evaluation"] == "perturbation")
        & (summary["metric"] == metric)
        & (summary["group"] == "unobserved")
    ]
    pooled = pool_populations(rows, ["obs_fraction", "seed"])
    fig, ax = plt.subplots(figsize=SINGLE)
    sweep_series(
        ax,
        pooled,
        "obs_fraction",
        metric,
        UNOBSERVED,
        seeds=True,
        errorbars=False,
    )
    ceiling(ax, pooled, "obs_fraction", UNOBSERVED)
    markers = dimensionality_markers(ax, dimensionality)
    ax.set_title(PERTURBATION_TITLE)
    observed_axis(ax, pooled["obs_fraction"].unique())
    performance_axis(ax, ylim)  # shared with panel (a)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=UNOBSERVED,
                linewidth=6,
                label=PERTURBATION_LABEL,
            ),
            Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Noise Ceiling"),
        ]
        + markers,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        fontsize=TICK_SIZE - 2,
        handlelength=1.6,
        labelspacing=0.35,
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

    def output(fig, letter, slug, raster=False):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate, raster)

    held_out = summary
    if "evaluation" in summary:
        held_out = summary[summary["evaluation"] == "held_out"]
    ylim = limits(summary)
    output(curve(held_out, dimensionality, ylim), "a", "curve")

    fractions = scatter_fractions or default_scatter_fractions(held_out)
    seed = int(rates["seed"].min())
    output(scatters(held_out, rates, fractions, seed), "b", "scatter", raster=True)

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(
            delta_sweep(summary, "delta_fluctuation_r2", dimensionality, ylim),
            "c",
            "delta-fluctuation",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--scatter-fractions", type=float, nargs=3, default=None)
    args = parser.parse_args()
    main(args.data, args.out_dir, args.scatter_fractions)
