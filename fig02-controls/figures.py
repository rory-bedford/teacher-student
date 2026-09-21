"""Figure 2 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig02-controls/figures.py

    fig02-a-bars-held-out      held-out Fluctuation R²: observed | unobserved
    fig02-b-bars-perturbation  perturbation ΔFluctuation R²: unobserved E and I pooled,
                               targeted cells excluded
    fig02-c-legend             the shared legend, stacked vertically, on its own

Two figures (2026-09-21), each a row of subpanels one population wide. Every subpanel is
``SUBPANEL_SIZE`` in both figures and they share one y range, so the two tile on a single
slide with matching subpanel sizes; neither carries a legend. Activity R² is still
scored and kept in the CSVs but not plotted (rates are reported by the scatter panels of
Figures 1 and 3), and the connectivity schematic was dropped -- it belongs on a slide of
its own, not in this figure.

Style is the shared slide style (``common/style.py``), sized to drop into the talk at
100%.

Plots ``PLOTTED_VARIANTS``, a subset of the variants analysis.py scores: the weight
shuffle is trained and scored but not shown (see the constant).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectome_snns.visualization import (
    CONFIGURATION_MODEL_COLOR,
    LEARNT_RECURRENCE_COLOR,
    SHUFFLE_WEIGHTS_COLOR,
)

from common.plotting import METRIC_LABELS, PERTURBATION_LABEL, pool_populations
from common.style import (
    LEGEND_GREY,
    TRUTH,
    apply_style,
    clear_panels,
    save,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig02"
#: Epoch budget plotted for the learnt-recurrence bars (2026-09-21). The 50-epoch runs
#: and the 100-epoch reruns coexist in the grid, so this switches the figure between them:
#: set it to 100 once ``learnt-100ep__seed-*`` have finished. Variants with a single
#: budget are unaffected.
LEARNT_EPOCHS = 50
PLOTTED_VARIANTS = [
    "full_connectome",
    "learnt_recurrence",
    "shuffle_weights",
    "configuration_model",
]
VARIANT_LABELS = {
    "full_connectome": "Full Connectome",
    "learnt_recurrence": "Learnt Recurrence",
    "shuffle_weights": "Shuffled Weights",
    "shuffle_weights_global": "Shuffled Weights (Whole Connectome)",
    "configuration_model": "Shuffled Topology",
}
#: One fixed accent per condition, never reassigned. The full connectome takes the
#: teacher's steel blue: it is the teacher's connectivity, and it is the first bar.
VARIANT_COLORS = {
    "full_connectome": TRUTH,
    "learnt_recurrence": LEARNT_RECURRENCE_COLOR,
    "shuffle_weights": SHUFFLE_WEIGHTS_COLOR,
    "shuffle_weights_global": SHUFFLE_WEIGHTS_COLOR,
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
        # One subpanel, E and I pooled, targeted cells excluded -- exactly the population
        # every other figure's perturbation panel reports (2026-09-21). The targeted cells
        # and the separate cell types are still scored, in fig02_summary.csv.
        "delta_fluctuation_r2",
        ((("unobserved"), "pooled", PERTURBATION_LABEL),),
    ),
)
#: Every subpanel is this size in both figures, so they tile on one slide.
SUBPANEL_SIZE = (4.4, 4.0)


def plotted_epochs(summary):
    """Keep one epoch budget per variant: LEARNT_EPOCHS for learnt recurrence."""
    if "total_epochs" not in summary:  # CSVs written before 2026-09-21
        return summary
    learnt = summary["variant"] == "learnt_recurrence"
    return summary[~learnt | (summary["total_epochs"] == LEARNT_EPOCHS)]


def panel_rows(summary, metric, group, cell_type):
    """Rows per variant for one population; ``cell_type="pooled"`` pools E and I."""
    rows = summary[(summary["metric"] == metric) & (summary["group"] == group)]
    if cell_type == "pooled":
        rows = pd.concat(
            [
                pool_populations(
                    rows[rows["variant"] == variant], ["variant", "seed"]
                ).assign(variant=variant)
                for variant in PLOTTED_VARIANTS
                if not rows[rows["variant"] == variant].empty
            ]
        )
    else:
        rows = rows[rows["cell_type"] == cell_type]
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
    handles.append(
        Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Noise Ceiling")
    )
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
    summary = plotted_epochs(summary)
    if summary.empty:
        raise SystemExit(f"no runs at observed_fraction {observed_fraction}")

    def output(fig, letter, slug, raster=False):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate, raster)

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
