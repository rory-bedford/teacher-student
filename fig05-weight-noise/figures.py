"""Figure 5 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig05-weight-noise/figures.py

    fig05-a-curve                   R² vs weight noise, observed / unobserved, both metrics
    fig05-b-contrast                unobserved Fluctuation R²: weight noise | neuron removal
    fig05-c-delta-fluctuation       perturbation: ΔFluctuation R² vs weight noise

Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panels of figures 1 and 3.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. ``placeholder_figures/fig05-weight-noise/figures.py`` calls ``main`` here with a
watermark and fake CSVs, so content edits show up in both.

Panel (b) reads Figure 4's neuron-removal curve from
../fig04-reconstruction-errors/fig04_summary.csv, so run Figure 4's analysis first.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from connectome_snns.visualization import (
    NEURON_REMOVAL_COLOR,
    OBSERVED_COLOR,
    UNOBSERVED_COLOR,
    WEIGHT_NOISE_COLOR,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS, PERTURBATION_SERIES
from common.style import (
    PAIR,
    SINGLE,
    TICK_SIZE,
    apply_style,
    ceiling,
    clear_panels,
    save,
    sweep_legend,
    sweep_series,
)

HERE = Path(__file__).resolve().parent
FIG04_SUMMARY = HERE.parent / "fig04-reconstruction-errors" / "fig04_summary.csv"
FIGURE = "fig05"
GROUPS = {
    "observed": ("Observed", OBSERVED_COLOR),
    "unobserved": ("Unobserved", UNOBSERVED_COLOR),
}


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
    rows = summary[
        summary["metric"].isin(["fluctuation_r2", "delta_fluctuation_r2"])
        & (summary["group"] == "unobserved")
    ]
    return min(0.0, float(rows["value"].min()) - 0.05), 1.02


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
    ax.set_ylim(*ylim)
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
    ax.set_title("Observed and Unobserved Neurons vs Weight Noise")
    fig.tight_layout()
    return fig


def contrast(summary, fig04_summary, ylim):
    """(b) The contrast panel: imprecise weights beside missing connections, shared y."""
    rows = group_rows(held_out(summary), "unobserved", "fluctuation_r2")
    fig, (left, right) = plt.subplots(1, 2, figsize=PAIR, sharey=True)
    sweep_series(
        left,
        rows,
        "weight_noise",
        "fluctuation_r2",
        WEIGHT_NOISE_COLOR,
        seeds=True,
        errorbars=False,
    )
    left.set_xlabel("Weight Noise Fraction")
    left.set_ylabel("Fluctuation R² (Unobserved)")
    left.set_title("Imprecise Weights (Figure 5)")
    if Path(fig04_summary).exists():
        fig04 = held_out(pd.read_csv(fig04_summary))
        removal = fig04[
            (fig04["error_model"] == "neuron_removal")
            & (fig04["group"] == "unobserved")
            & (fig04["metric"] == "fluctuation_r2")
        ]
        if "cell_type" in removal:
            removal = removal[removal["cell_type"] == "all"]
        # Snapped to the nominal grid and grouped by level, as Figure 4 does.
        removal = removal.assign(
            kappa_snapped=(removal["mean_kappa_lost"] * 10).round() / 10
        )
        sweep_series(
            right,
            removal,
            "kappa_snapped",
            "fluctuation_r2",
            NEURON_REMOVAL_COLOR,
            seeds=True,
            errorbars=False,
            x_group="level",
        )
    else:
        right.text(
            0.5, 0.5, "run fig04 analysis.py", transform=right.transAxes, ha="center"
        )
    right.set_xlabel("Fraction of Recurrent Input Lost")
    right.set_title("Missing Connections (Figure 4)")
    left.set_ylim(*ylim)
    fig.suptitle("Unobserved Neurons vs Input Volume Lost, Both Error Models")
    fig.tight_layout()
    return fig


def perturbation(summary, metric, ylim):
    """The intervention's effect against weight noise, one population per colour."""
    rows = summary[summary["evaluation"] == "perturbation"]
    fig, ax = plt.subplots(figsize=SINGLE)
    series = {}
    for group, cell_type, color, label in PERTURBATION_SERIES:
        subset = group_rows(rows, group, metric, cell_type)
        # sweep_series looks its marker/linestyle up by the base metric name, so the
        # delta panels keep the archived Activity o- / Fluctuation s-- convention.
        base_metric = metric.replace("delta_", "")
        sweep_series(
            ax,
            subset,
            "weight_noise",
            base_metric,
            color,
            seeds=True,
            errorbars=False,
        )
        ceiling(ax, subset, "weight_noise", color)
        series[label] = color
    ax.set_xlabel("Weight Noise Fraction")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(*ylim)
    sweep_legend(
        ax,
        series,
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(f"{METRIC_LABELS[metric]}\nInhibiting 25% of Unobserved I Cells")
    fig.tight_layout()
    return fig


def main(data_dir, fig04_summary, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig05_summary.csv")
    clipped = summary.groupby("weight_noise")["noise_clipped_fraction"].mean()

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    output(curve(summary, clipped, ylim), "a", "curve")
    output(contrast(summary, fig04_summary, ylim), "b", "contrast")

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
    parser.add_argument("--fig04-summary", type=Path, default=FIG04_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.fig04_summary, args.out_dir)
