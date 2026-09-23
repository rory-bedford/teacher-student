"""Figure 6 — one SVG or PNG per panel from the CSVs written by analysis.py.

    uv run python fig06-learnt-feedforward/figures.py

    fig06-a-bars-held-out       Fluctuation R² on the held-out trial, Observed | Unobserved,
                                input given (Figure 1) beside input learnt
    fig06-b-bars-perturbation   the same two conditions under the perturbation, cell types
                                pooled over the unobserved population
    fig06-c-legend              the shared legend, as its own file to drag onto the slide
    fig06-d-scatter             teacher vs student firing rate, Observed | Unobserved, for
                                the learnt-input condition

The claim is the gap between the two bars *within* the unobserved panel, against Figure 1
where both populations are predicted equally well. Style is the shared slide style
(``common/style.py``), sized to drop into the talk at 100%.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    HELD_OUT_TITLE,
    METRIC_LABELS,
    PERTURBATION_LABEL,
    PERTURBATION_TITLE,
    pool_populations,
    rate_scatter,
)
from common.style import (
    LEGEND_GREY,
    MODEL,
    PAIR,
    TICK_SIZE,
    TRUTH,
    apply_style,
    clear_panels,
    performance_axis,
    performance_limits,
    save,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig06"
#: Blue is what you are given, orange what the model infers (COLORSCHEME.txt).
VARIANTS = ["known_input", "learnt_feedforward"]
VARIANT_LABELS = {
    "known_input": "Input Given",
    "learnt_feedforward": "Input Learnt",
}
VARIANT_COLORS = {"known_input": TRUTH, "learnt_feedforward": MODEL}
#: Both bar figures use this subpanel size, so held-out (two populations) and perturbation
#: (one) tile on a slide with every subpanel the same size.
SUBPANEL_SIZE = (4.4, 4.0)
POPULATIONS = [("observed", "Observed Neurons"), ("unobserved", "Unobserved Neurons")]


def held_out(summary):
    return summary[summary["evaluation"] == "held_out"]


def limits(summary):
    """One y range over both bar figures, so the two can be read against each other."""
    rows = summary[summary["metric"].isin(["fluctuation_r2", "delta_fluctuation_r2"])]
    return performance_limits(rows["value"])


def subpanel(ax, summary, metric, group, title):
    """One population: a bar per condition, the three seeds as dots, dotted ceiling.

    No error bars (2026-09-23): with three seeds the points are the distribution.
    """
    for position, variant in enumerate(VARIANTS):
        rows = summary[
            (summary["metric"] == metric)
            & (summary["group"] == group)
            & (summary["variant"] == variant)
        ]
        if rows.empty:
            continue
        ax.bar(
            position,
            rows["value"].mean(),
            0.7,
            color=VARIANT_COLORS[variant],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.scatter(
            np.full(len(rows), position), rows["value"], s=8, color="k", zorder=3
        )
        if rows["ceiling_value"].notna().any():
            ax.hlines(
                rows["ceiling_value"].mean(),
                position - 0.35,
                position + 0.35,
                colors="k",
                linestyles=":",
                linewidth=1.2,
            )
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_title(title, fontsize=TICK_SIZE)


def bars(summary, metric, populations, ylim):
    """One figure, one subpanel per population, sharing a y axis."""
    fig, axes = plt.subplots(
        1,
        len(populations),
        figsize=(SUBPANEL_SIZE[0] * len(populations), SUBPANEL_SIZE[1]),
        sharey=True,
    )
    for ax, (group, title) in zip(np.atleast_1d(axes), populations, strict=True):
        subpanel(ax, summary, metric, group, title)
    for ax in np.atleast_1d(axes):
        performance_axis(ax, ylim)
    fig.supylabel(METRIC_LABELS[metric])
    fig.suptitle(PERTURBATION_TITLE if metric.startswith("delta") else HELD_OUT_TITLE)
    fig.tight_layout()
    return fig


def legend(summary):
    """The shared legend as its own file, stacked vertically."""
    present = set(summary["variant"])
    handles = [
        Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in VARIANTS
        if v in present
    ]
    handles.append(
        Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Noise Ceiling")
    )
    fig = plt.figure(figsize=(3.0, 1.2))
    fig.legend(handles=handles, loc="center", frameon=True, ncol=1)
    return fig


def scatters(rates, summary):
    """Teacher vs student firing rate for the learnt-input condition."""
    subset = rates[rates["variant"] == "learnt_feedforward"]
    seed = int(subset["seed"].min())
    subset = subset[subset["seed"] == seed]
    fig, axes = plt.subplots(1, 2, figsize=PAIR)
    for ax, (group, title) in zip(axes, POPULATIONS, strict=True):
        rows = subset[subset["observed"] == int(group == "observed")]
        rate_scatter(ax, rows, title)
        ax.title.set_fontsize(TICK_SIZE)
    fig.suptitle("Firing Rates, Student vs Teacher, Input Learnt, Held-Out Stimulus")
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig06_summary.csv")
    rates = pd.read_csv(data_dir / "fig06_rates.csv")

    def output(fig, letter, slug, raster=False):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate, raster)

    ylim = limits(summary)
    output(
        bars(held_out(summary), "fluctuation_r2", POPULATIONS, ylim),
        "a",
        "bars-held-out",
    )

    perturbation = summary[summary["evaluation"] == "perturbation"]
    if not perturbation.empty:
        # Pool E and I within each population, as every other perturbation panel does;
        # "group" stays a key so the unobserved population can still be selected.
        pooled = pool_populations(
            perturbation, ["variant", "seed", "metric", "group"]
        ).reset_index()
        output(
            bars(
                pooled,
                "delta_fluctuation_r2",
                [("unobserved", PERTURBATION_LABEL)],
                ylim,
            ),
            "b",
            "bars-perturbation",
        )
        plt.close("all")

    output(legend(summary), "c", "legend")
    output(scatters(rates, held_out(summary)), "d", "scatter", raster=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
