"""Figure 2 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig02-controls/figures.py

    fig02-a-bars-fluctuation        Fluctuation R², observed / unobserved, per variant
    fig02-b-bars-delta-fluctuation  perturbation: ΔFluctuation R² per population

Two panels only (2026-09-18): held-out Fluctuation R² and the perturbation's
ΔFluctuation R², both with every plotted variant side by side. Activity R² is still
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
#: (group, cell_type, x label) per group of bars, held-out and perturbation.
HELD_OUT_GROUPS = (
    ("observed", "all", "Observed"),
    ("unobserved", "all", "Unobserved"),
)
PERTURBATION_GROUPS = (
    ("unobserved", "excitatory", "Non-targeted E"),
    ("unobserved", "inhibitory", "Non-targeted I"),
    ("targeted", "inhibitory", "Targeted I"),
)


def bars(summary, metric, groups=HELD_OUT_GROUPS):
    """Archived controls bar chart: populations on x, one bar per variant, legend above."""
    rows = summary[summary["metric"] == metric]
    variants = [v for v in PLOTTED_VARIANTS if v in set(rows["variant"])]
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for g, (group, cell_type, _) in enumerate(groups):
        for i, variant in enumerate(variants):
            sub = rows[
                (rows["variant"] == variant)
                & (rows["group"] == group)
                & (rows["cell_type"] == cell_type)
            ]
            if sub.empty:
                continue
            x = g + (i - (len(variants) - 1) / 2) * width
            ax.bar(
                x,
                sub["value"].mean(),
                width,
                color=VARIANT_COLORS[variant],
                edgecolor="white",
                linewidth=0.5,
            )
            ax.errorbar(
                x,
                sub["value"].mean(),
                yerr=sub["value"].std() if len(sub) > 1 else 0,
                color="k",
                capsize=3,
                linewidth=1,
            )
            ax.scatter(np.full(len(sub), x), sub["value"], s=8, color="k", zorder=3)
            ax.hlines(
                sub["ceiling_value"].mean(),
                x - width / 2,
                x + width / 2,
                colors="k",
                linestyles=":",
                linewidth=1.2,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for _, _, label in groups])
    ax.set_ylabel(METRIC_LABELS[metric])
    ymin = min(0.0, rows["value"].min()) - 0.05 if not rows.empty else -0.05
    ax.set_ylim(ymin, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.axhline(0, color="k", linewidth=0.5)
    handles = [
        Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v]) for v in variants
    ]
    handles.append(Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Ceiling"))
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout()
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

    output(bars(summary, "fluctuation_r2"), "a", "bars-fluctuation")

    # The perturbation panel is a separate file, so dropping it from the talk is
    # dropping one SVG.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        fig = bars(summary, "delta_fluctuation_r2", PERTURBATION_GROUPS)
        output(fig, "b", "bars-delta-fluctuation")


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
