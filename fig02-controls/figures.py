"""Figure 2 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig02-controls/figures.py

    fig02-a-bars-held-out      held-out Fluctuation R²: observed | unobserved
    fig02-b-bars-perturbation  perturbation ΔFluctuation R²: non-targeted E | I | targeted I
    fig02-c-legend             the shared legend, stacked vertically, on its own

Two figures (2026-09-21), each a row of subpanels one population wide. Every subpanel is
``SUBPANEL_SIZE`` in both figures and they share one y range, so the two tile on a single
slide with matching subpanel sizes; neither carries a legend. Activity R² is still
scored and kept in the CSVs but not plotted (rates are reported by the scatter panels of
Figures 1 and 3), and the connectivity schematic was dropped -- it belongs on a slide of
its own, not in this figure.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. ``placeholder_figures/fig02-controls/figures.py`` calls ``main`` here with a
watermark and fake CSVs, so content edits show up in both.

Plots ``PLOTTED_VARIANTS``, a subset of the variants analysis.py scores: the weight
shuffle is trained and scored but not shown (see the constant).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import (
    CONFIGURATION_MODEL_COLOR,
    FULL_CONNECTOME_COLOR,
    LEARNT_RECURRENCE_COLOR,
    SHUFFLE_WEIGHTS_COLOR,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS
from common.style import LEGEND_GREY, apply_style, clear_panels, save

HERE = Path(__file__).resolve().parent
FIGURE = "fig02"
#: Plotted bars — a SUBSET of what analysis.py scores. "shuffle_inputs" (shuffled weights
#: within neuron) is trained on every seed and stays in fig02_summary.csv / fig02_rates.csv,
#: but is not plotted (2026-09-18): it overlaps with Figure 5's weight noise, which makes
#: the same point as a graded curve. Add it back to this list to restore the bar.
PLOTTED_VARIANTS = [
    "full_connectome",
    "learnt_recurrence",
    "configuration_model",
]
VARIANT_LABELS = {
    "full_connectome": "Full Connectome",
    "learnt_recurrence": "Learnt Recurrence",
    "shuffle_inputs": "Shuffled Weights",
    "configuration_model": "Configuration Model",
}
VARIANT_COLORS = {
    "full_connectome": FULL_CONNECTOME_COLOR,
    "learnt_recurrence": LEARNT_RECURRENCE_COLOR,
    "shuffle_inputs": SHUFFLE_WEIGHTS_COLOR,
    "configuration_model": CONFIGURATION_MODEL_COLOR,
}
#: Two figures, each a row of subpanels, one per scored population (2026-09-21):
#: (metric, [(group, cell_type, title), ...]).
FIGURES = (
    (
        "fluctuation_r2",
        (("observed", "all", "Observed"), ("unobserved", "all", "Unobserved")),
    ),
    (
        "delta_fluctuation_r2",
        (
            ("unobserved", "excitatory", "Non-targeted E"),
            ("unobserved", "inhibitory", "Non-targeted I"),
            ("targeted", "inhibitory", "Targeted I"),
        ),
    ),
)
#: Every subpanel is this size in both figures, so they tile on one slide.
SUBPANEL_SIZE = (3.4, 4.0)


def panel_rows(summary, metric, group, cell_type):
    rows = summary[
        (summary["metric"] == metric)
        & (summary["group"] == group)
        & (summary["cell_type"] == cell_type)
    ]
    return [(v, rows[rows["variant"] == v]) for v in PLOTTED_VARIANTS]


def limits(summary):
    """One y range for both figures, so every subpanel is directly comparable."""
    metrics = [metric for metric, _ in FIGURES]
    rows = summary[
        summary["metric"].isin(metrics) & summary["variant"].isin(PLOTTED_VARIANTS)
    ]
    low = min(0.0, float(rows["value"].min()))
    return low - 0.08, 1.08


def subpanel(ax, summary, metric, group, cell_type, title):
    """One population: a bar per variant, per-seed dots, dotted ceiling, no legend."""
    for position, (variant, rows) in enumerate(
        panel_rows(summary, metric, group, cell_type)
    ):
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
        ax.errorbar(
            position,
            rows["value"].mean(),
            yerr=rows["value"].std() if len(rows) > 1 else 0,
            color="k",
            capsize=3,
            linewidth=1,
        )
        ax.scatter(
            np.full(len(rows), position), rows["value"], s=8, color="k", zorder=3
        )
        ax.hlines(
            rows["ceiling_value"].mean(),
            position - 0.35,
            position + 0.35,
            colors="k",
            linestyles=":",
            linewidth=1.2,
        )
    ax.set_xticks(range(len(PLOTTED_VARIANTS)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(PLOTTED_VARIANTS) - 0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title(title)


def bars(summary, metric, populations, ylim):
    """One figure, one subpanel per population.

    Subpanels are ``SUBPANEL_SIZE`` in both figures and share one y axis, so the held-out
    figure (two populations) and the perturbation figure (three) tile on a slide with
    every subpanel the same size. The legend is a separate file.
    """
    fig, axes = plt.subplots(
        1,
        len(populations),
        figsize=(SUBPANEL_SIZE[0] * len(populations), SUBPANEL_SIZE[1]),
        sharey=True,
    )
    for ax, (group, cell_type, title) in zip(np.atleast_1d(axes), populations):
        subpanel(ax, summary, metric, group, cell_type, title)
    np.atleast_1d(axes)[0].set_ylim(*ylim)
    fig.supylabel(METRIC_LABELS[metric])
    fig.tight_layout()
    return fig


def legend(summary):
    """The shared legend as its own file, stacked vertically."""
    present = set(summary["variant"])
    handles = [
        Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in PLOTTED_VARIANTS
        if v in present
    ]
    handles.append(Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Ceiling"))
    fig = plt.figure(figsize=(2.1, 0.42 * len(handles) + 0.2))
    fig.legend(handles=handles, loc="center", ncol=1, frameon=True)
    return fig


def main(data_dir, out_dir, observed_fraction=None, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig02_summary.csv")
    # The grid held two observation levels until 2026-09-18; plot one per figure.
    if "observed_fraction" not in summary:  # CSVs written before 2026-09-18
        summary["observed_fraction"] = np.nan
    if "cell_type" not in summary:  # CSVs written before the perturbation panel
        summary["cell_type"] = "all"
        summary["evaluation"] = "held_out"
    if observed_fraction is None:
        observed_fraction = summary["observed_fraction"].min()
    if np.isnan(observed_fraction):
        summary["observed_fraction"] = observed_fraction = 0.5
    summary = summary[np.isclose(summary["observed_fraction"], observed_fraction)]
    if summary.empty:
        raise SystemExit(f"no runs at observed_fraction {observed_fraction}")

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    for letter, (metric, populations), slug in zip(
        "ab", FIGURES, ("bars-held-out", "bars-perturbation")
    ):
        if not (summary["metric"] == metric).any():
            print(f"  no {metric} rows yet: skipping ({letter})")
            continue
        output(bars(summary, metric, populations, ylim), letter, slug)
    output(legend(summary), "c", "legend")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument(
        "--observed-fraction",
        type=float,
        help="which observation level to plot (default: the lowest present)",
    )
    args = parser.parse_args()
    main(args.data, args.out_dir, args.observed_fraction)
