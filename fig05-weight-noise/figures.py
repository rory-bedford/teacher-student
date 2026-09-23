"""Figure 5 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig05-weight-noise/figures.py

    fig05-a-weight-perturbation     what the noise does to a synapse, at 0.1 and 0.5
    fig05-b-curve                   Fluctuation R² vs weight noise, observed / unobserved
    fig05-c-delta-fluctuation       perturbation: ΔFluctuation R² vs weight noise

The contrast panel (weight noise beside Figure 4's neuron removal) was removed on
2026-09-21: it duplicated Figure 4's own curve, and comparing the two error types is a
job for the slide deck rather than a panel.

Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panels of figures 1 and 3.

Style is the shared slide style (``common/style.py``), sized to drop into the talk at
100%.

"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

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
    INK,
    MODEL,
    OBSERVED,
    PAIR,
    REFERENCE_GREY,
    SINGLE,
    TICK_SIZE,
    UNOBSERVED,
    apply_style,
    ceiling,
    clear_panels,
    performance_axis,
    performance_limits,
    save,
    sweep_legend,
    sweep_series,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig05"
GROUPS = {
    "observed": ("Observed", OBSERVED),
    "unobserved": ("Unobserved", UNOBSERVED),
}


#: The upper limit of the weight axes, as a percentile of the sampled synapses: the
#: teacher's weights are log-normal with a long tail (a handful reach 65), so plotting
#: the full range would put every point in one corner.
PERTURBATION_PERCENTILE = 99.0
#: Both weight axes carry the same ticks, so the identity line reads as the diagonal it is.
PERTURBATION_TICK = 0.1


def weight_perturbation(table):
    """(a) Single synapses before and after the noise, one panel per level.

    A recreation of the old repository's ``weight_perturbation.svg``: the perturbed weight
    against the original, the identity dashed, and r², which is what degrades. Lower
    case: it is the squared correlation between two sets of weights, not the variance a
    model explains, which is what R² means on every other panel in the talk. The mean and
    SD are not quoted (2026-09-23): the noise preserves them exactly by construction, so
    they said nothing the panel needed. They remain in fig05_weight_perturbation.csv, over
    the whole non-zero population. R² is over that population too; the points are a random
    sample of it, and the axes stop at the PERTURBATION_PERCENTILE of the weights, since a
    handful of synapses run two orders of magnitude further out.
    """
    levels = sorted(table["weight_noise"].unique())
    limit = float(
        np.percentile(table[["original", "noisy"]].to_numpy(), PERTURBATION_PERCENTILE)
    )
    fig, axes = plt.subplots(1, len(levels), figsize=PAIR, sharex=True, sharey=True)
    for ax, noise in zip(np.atleast_1d(axes), levels, strict=True):
        rows = table[table["weight_noise"] == noise]
        ax.plot(
            [0, limit], [0, limit], color=INK, linestyle="--", linewidth=1, zorder=3
        )
        ax.scatter(
            rows["original"],
            rows["noisy"],
            s=6,
            color=MODEL,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        ax.tick_params(labelleft=True)  # both panels keep their y ticks, despite sharey
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(PERTURBATION_TICK))
        ax.yaxis.set_major_locator(MultipleLocator(PERTURBATION_TICK))
        ax.set_xlabel("Teacher Weight (nS)")
        ax.set_title(f"Weight Noise {noise:g}")
        ax.text(
            0.04,
            0.96,
            f"r² = {rows['correlation'].iloc[0] ** 2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TICK_SIZE - 1,
            color=INK,
            bbox={
                "facecolor": "white",
                "edgecolor": REFERENCE_GREY,
                "boxstyle": "round",
            },
        )
    np.atleast_1d(axes)[0].set_ylabel("Student Weight (nS)")
    fig.tight_layout()
    return fig


def held_out(summary):
    """The held-out rows; CSVs written before the perturbation panel have no column."""
    if "evaluation" in summary:
        return summary[summary["evaluation"] == "held_out"]
    return summary


def group_rows(summary, group, metric, cell_type="all"):
    rows = summary[(summary["group"] == group) & (summary["metric"] == metric)]
    if "cell_type" in rows:
        rows = rows[rows["cell_type"] == cell_type]
    return rows


def limits(summary):
    """One y range for all three panels, as in Figures 3 and 4."""
    return performance_limits(
        plotted_performance_values(summary, "unobserved", ["weight_noise", "seed"])
    )


def curve(summary, clipped, ylim):
    """(a) Fluctuation R² against weight noise, per population, with dotted ceilings."""
    rows = held_out(summary)
    fig, ax = plt.subplots(figsize=SINGLE)
    for group, (_, color) in GROUPS.items():
        sweep_series(
            ax,
            group_rows(rows, group, "fluctuation_r2"),
            "weight_noise",
            "fluctuation_r2",
            color,
            seeds=True,
            errorbars=False,
        )
        ceiling(ax, group_rows(rows, group, "fluctuation_r2"), "weight_noise", color)
    ax.set_xlim(-0.025, rows["weight_noise"].max() + 0.025)
    # Limits from the data, not fixed: the real sweep runs lower than the estimates.
    performance_axis(ax, ylim)
    ax.set_xlabel("Weight Noise Fraction")
    ax.set_ylabel("Fluctuation R²")
    sweep_legend(
        ax,
        {label: color for label, color in GROUPS.values()},
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    # The mean/SD-preserving perturbation clips a few weights at zero (1.5-4.5% across
    # the sweep). That was in the title until 2026-09-21: five numbers nobody reads from
    # the back of a room, and the figure's README carries them instead.
    ax.set_title(HELD_OUT_TITLE)
    fig.tight_layout()
    return fig


def perturbation(summary, metric, ylim):
    """(b) The intervention's effect against weight noise, cell types pooled.

    The non-targeted unobserved E and I populations are pooled (see
    ``common.plotting.pool_populations``): across this sweep they differ by less than
    0.02 R² at every level, so two series were redundant.
    """
    rows = summary[
        (summary["evaluation"] == "perturbation")
        & (summary["metric"] == metric)
        & (summary["group"] == "unobserved")
    ]
    pooled = pool_populations(rows, ["weight_noise", "seed"])
    fig, ax = plt.subplots(figsize=SINGLE)
    # sweep_series looks its marker/linestyle up by the base metric name, so the delta
    # panels keep the archived Activity o- / Fluctuation s-- convention.
    base_metric = metric.replace("delta_", "")
    sweep_series(
        ax,
        pooled,
        "weight_noise",
        base_metric,
        UNOBSERVED,
        seeds=True,
        errorbars=False,
    )
    ceiling(ax, pooled, "weight_noise", UNOBSERVED)
    ax.set_xlabel("Weight Noise Fraction")
    ax.set_ylabel(METRIC_LABELS[metric])
    performance_axis(ax, ylim)
    sweep_legend(
        ax,
        {PERTURBATION_LABEL: UNOBSERVED},
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(PERTURBATION_TITLE)
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig05_summary.csv")
    clipped = summary.groupby("weight_noise")["noise_clipped_fraction"].mean()

    def output(fig, letter, slug, raster=False):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate, raster)

    perturbation_table = data_dir / "fig05_weight_perturbation.csv"
    if perturbation_table.exists():
        output(
            weight_perturbation(pd.read_csv(perturbation_table)),
            "a",
            "weight-perturbation",
            raster=True,
        )

    ylim = limits(summary)
    output(curve(summary, clipped, ylim), "b", "curve")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(
            perturbation(summary, "delta_fluctuation_r2", ylim),
            "c",
            "delta-fluctuation",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
