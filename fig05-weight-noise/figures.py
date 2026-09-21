"""Figure 5 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig05-weight-noise/figures.py

    fig05-a-curve                   Fluctuation R² vs weight noise, observed / unobserved
    fig05-b-delta-fluctuation       perturbation: ΔFluctuation R² vs weight noise

The contrast panel (weight noise beside Figure 4's neuron removal) was removed on
2026-09-21: it duplicated Figure 4's own curve, and comparing the two error types is a
job for the slide deck rather than a panel.

Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panels of figures 1 and 3.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. ``placeholder_figures/fig05-weight-noise/figures.py`` calls ``main`` here with a
watermark and fake CSVs, so content edits show up in both.

"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from connectome_snns.visualization import (
    OBSERVED_COLOR,
    UNOBSERVED_COLOR,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS, pool_populations
from common.style import (
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
        UNOBSERVED_COLOR,
        seeds=True,
        errorbars=False,
    )
    ceiling(ax, pooled, "weight_noise", UNOBSERVED_COLOR)
    ax.set_xlabel("Weight Noise Fraction")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(*ylim)
    sweep_legend(
        ax,
        {"Unobserved": UNOBSERVED_COLOR},
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(f"{METRIC_LABELS[metric]}\nInhibiting 25% of Unobserved I Cells")
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig05_summary.csv")
    clipped = summary.groupby("weight_noise")["noise_clipped_fraction"].mean()

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    output(curve(summary, clipped, ylim), "a", "curve")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(
            perturbation(summary, "delta_fluctuation_r2", ylim),
            "b",
            "delta-fluctuation",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
