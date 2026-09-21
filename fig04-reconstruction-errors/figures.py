"""Figure 4 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig04-reconstruction-errors/figures.py

    fig04-a-curve                unobserved Fluctuation R² vs input volume lost, both models
    fig04-b-delta-fluctuation    perturbation: ΔFluctuation R² vs input volume lost

The per-neuron panel (per-neuron R² against that neuron's own lost input) was deleted on
2026-09-21: its premise was that neuron removal would spread per-neuron loss much wider
than synapse dropout, and the data says the spreads match (SD 0.173 vs 0.178). Panel (a)
makes the population claim more legibly. It is in the git history if ever wanted.

Activity R² is scored and kept in the CSVs but not plotted (2026-09-18): rates are
reported by the scatter panels of figures 1 and 3.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%: both metrics per error model, Activity ``o-`` and Fluctuation ``s--``.
``placeholder_figures/fig04-reconstruction-errors/figures.py`` calls ``main`` here with a
watermark and fake CSVs, so content edits show up in both.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import NEURON_REMOVAL_COLOR, SYNAPSE_DROPOUT_COLOR
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS
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
FIGURE = "fig04"
MODELS = {
    "neuron_removal": ("Neuron Removal", NEURON_REMOVAL_COLOR),
    "synapse_dropout": ("Synapse Dropout", SYNAPSE_DROPOUT_COLOR),
}
X_LABEL = "Fraction of Recurrent Input Lost"
#: The perturbation's non-targeted unobserved populations (targets scored separately).


def limits(summary):
    """One y range for panels (a) and (b), so the two read on the same scale."""
    rows = summary[
        summary["metric"].isin(["fluctuation_r2", "delta_fluctuation_r2"])
        & (summary["group"] == "unobserved")
    ]
    return min(0.0, float(rows["value"].min()) - 0.05), 1.0


def curve(summary, ylim):
    """(a) unobserved Fluctuation R² against input volume lost, per error model."""
    unobserved = summary[summary["group"] == "unobserved"]
    fig, ax = plt.subplots(figsize=SINGLE)
    for model, (_, color) in MODELS.items():
        rows = unobserved[
            (unobserved["error_model"] == model)
            & (unobserved["metric"] == "fluctuation_r2")
        ]
        if rows.empty:
            continue
        # Plotted against the volume actually lost, not the nominal level: the two error
        # models reach the same fraction at different levels. The measured fraction is
        # snapped to the nearest 0.1 so the two series align -- the realised values sit
        # within 0.7% of the grid, far below anything this panel claims.
        rows = rows.assign(
            kappa_snapped=(rows["mean_kappa_lost"] * 10).round() / 10,
        )
        sweep_series(
            ax,
            rows,
            "kappa_snapped",
            "fluctuation_r2",
            color,
            seeds=True,
            errorbars=False,
            x_group="level",
        )
        ceiling(ax, rows, "kappa_snapped", color, x_group="level")
    ax.set_ylim(*ylim)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Fluctuation R² (Unobserved)")
    sweep_legend(
        ax,
        {label: color for label, color in MODELS.values()},
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title("Unobserved Neurons vs Input Volume Lost")
    fig.tight_layout()
    return fig


def pooled_populations(rows):
    """Non-targeted E and I of one condition as a single number per run.

    Weighted by cell count, so the pooled value is what the two populations' R² would be
    if they were one group of that size. It is an approximation -- a pooled R² computed
    from the traces themselves would need the smoothed traces, which the cache does not
    keep -- and it is close because E and I degrade together here.
    """
    return (
        rows.groupby(["level", "seed"])
        .apply(
            lambda group: pd.Series(
                {
                    "value": np.average(group["value"], weights=group["n_cells"]),
                    "ceiling_value": np.average(
                        group["ceiling_value"], weights=group["n_cells"]
                    ),
                    "mean_kappa_lost": group["mean_kappa_lost"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def delta_sweep(summary, metric, ylim):
    """(b) Δ R² of the intervention against input volume lost, per error model.

    The non-targeted unobserved E and I populations are pooled (see
    :func:`pooled_populations`), so the series are the two error models in panel (a)'s
    colours: four series of E and I per model was clutter, and the populations degrade
    together. Seeds as points, no error bars, panel (a)'s y range.
    """
    rows = summary[
        (summary["evaluation"] == "perturbation")
        & (summary["metric"] == metric)
        & (summary["group"] == "unobserved")
    ]
    fig, ax = plt.subplots(figsize=SINGLE)
    handles = []
    for model, (label, color) in MODELS.items():
        sub = rows[rows["error_model"] == model]
        if sub.empty:
            continue
        pooled = pooled_populations(sub)
        # Snapped to the nominal grid, as in panel (a): the realised fractions sit within
        # 0.7% of it and the two models would otherwise sit side by side.
        pooled = pooled.assign(
            kappa_snapped=(pooled["mean_kappa_lost"] * 10).round() / 10
        )
        sweep_series(
            ax,
            pooled,
            "kappa_snapped",
            metric,
            color,
            seeds=True,
            errorbars=False,
            x_group="level",
        )
        ceiling(ax, pooled, "kappa_snapped", color, x_group="level")
        handles.append(Line2D([], [], color=color, linewidth=6, label=label))
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(*ylim)
    sweep_legend(
        ax,
        {},
        metrics=False,
        extra=handles,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(
        f"{METRIC_LABELS[metric]}: Inhibiting 25% of Unobserved I Cells",
        fontsize=TICK_SIZE + 1,
    )
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig04_summary.csv")
    held_out = summary
    if "evaluation" in summary:
        held_out = summary[summary["evaluation"] == "held_out"]

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    output(curve(held_out, ylim), "a", "curve")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(
            delta_sweep(summary, "delta_fluctuation_r2", ylim), "b", "delta-fluctuation"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
