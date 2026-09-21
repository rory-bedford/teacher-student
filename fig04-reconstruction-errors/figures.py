"""Figure 4 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig04-reconstruction-errors/figures.py

    fig04-a-curve                unobserved Fluctuation R² vs input volume lost, both models
    fig04-b-per-neuron           per-neuron Fluctuation R² vs per-neuron volume lost
    fig04-c-delta-fluctuation    perturbation: ΔFluctuation R² vs input volume lost

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
MAX_POINTS = 20000
X_LABEL = "Mean Input Volume Lost (κ)"
#: The perturbation's non-targeted unobserved populations (targets scored separately).
PERTURBATION_CELL_TYPES = (("excitatory", "E", "-"), ("inhibitory", "I", "--"))


def curve(summary):
    """(a) unobserved Fluctuation R² against input volume lost, per error model."""
    unobserved = summary[summary["group"] == "unobserved"]
    fig, ax = plt.subplots(figsize=SINGLE)
    for model, (_, color) in MODELS.items():
        rows = unobserved[
            (unobserved["error_model"] == model)
            & (unobserved["metric"] == "fluctuation_r2")
        ]
        if not rows.empty:
            # Plotted against the volume actually lost, not the nominal level: the two
            # error models reach the same κ at different levels.
            sweep_series(ax, rows, "mean_kappa_lost", "fluctuation_r2", color)
        ceiling(
            ax,
            unobserved[
                (unobserved["error_model"] == model)
                & (unobserved["metric"] == "fluctuation_r2")
            ],
            "mean_kappa_lost",
            color,
        )
    ax.set_ylim(min(0.0, unobserved["value"].min() - 0.05), 1.0)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Fluctuation R² (Unobserved)")
    sweep_legend(ax, {label: color for label, color in MODELS.values()}, metrics=False)
    ax.set_title("Unobserved Neurons vs Input Volume Lost")
    fig.tight_layout()
    return fig


def per_neuron_panel(per_neuron):
    """(b) per-neuron Fluctuation R² against per-neuron κ, both error models pooled."""
    neurons = per_neuron[
        (per_neuron["observed"] == 0) & (per_neuron["level"] > 0)
    ].dropna(subset=["fluctuation_r2"])
    if len(neurons) > MAX_POINTS:
        neurons = neurons.sample(MAX_POINTS, random_state=0)
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=SINGLE)
    handles = []
    for model, (label, color) in MODELS.items():
        sub = neurons[neurons["error_model"] == model]
        if sub.empty:
            continue
        ax.scatter(
            sub["kappa_lost"],
            sub["fluctuation_r2"].clip(-1, 1),
            s=3,
            color=color,
            alpha=0.15,
            linewidths=0,
            rasterized=True,
        )
        binned = sub.groupby(pd.cut(sub["kappa_lost"], bins), observed=True)[
            "fluctuation_r2"
        ].median()
        ax.plot(
            [interval.mid for interval in binned.index],
            binned.values,
            color=color,
            linewidth=2.5,
        )
        handles.append(
            Line2D([], [], color=color, linewidth=2.5, label=f"{label} (Median)")
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Per-Neuron Input Volume Lost (κ)")
    ax.set_ylabel("Per-Neuron Fluctuation R²")
    ax.legend(handles=handles, loc="lower left", frameon=True)
    ax.set_title("Per-Neuron Prediction vs That Neuron's Input Lost")
    fig.tight_layout()
    return fig


def delta_sweep(summary, metric):
    """Δ R² of the intervention against input volume lost, one panel per metric.

    Same x-axis, colours and dotted ceilings as panel (a); E and I of the non-targeted
    unobserved population are the solid and dashed lines.
    """
    rows = summary[summary["evaluation"] == "perturbation"]
    fig, ax = plt.subplots(figsize=SINGLE)
    handles = []
    for model, (label, color) in MODELS.items():
        for cell_type, short, linestyle in PERTURBATION_CELL_TYPES:
            sub = rows[
                (rows["error_model"] == model)
                & (rows["metric"] == metric)
                & (rows["group"] == "unobserved")
                & (rows["cell_type"] == cell_type)
            ]
            if sub.empty:
                continue
            stats = sub.groupby("level")[
                ["mean_kappa_lost", "value", "ceiling_value"]
            ].mean()
            spread = sub.groupby("level")["value"].std().fillna(0.0)
            ax.errorbar(
                stats["mean_kappa_lost"],
                stats["value"],
                yerr=spread,
                color=color,
                marker="o",
                markersize=5,
                linestyle=linestyle,
                linewidth=1.5,
                capsize=2.5,
            )
            ax.plot(
                stats["mean_kappa_lost"],
                stats["ceiling_value"],
                ":",
                color=color,
                linewidth=1.2,
                alpha=0.7,
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    markersize=5,
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=f"{label}, {short}",
                )
            )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(METRIC_LABELS[metric])
    sweep_legend(ax, {}, metrics=False, extra=handles)
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
    per_neuron = pd.read_csv(data_dir / "fig04_per_neuron.csv")
    held_out = summary
    if "evaluation" in summary:
        held_out = summary[summary["evaluation"] == "held_out"]

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    output(curve(held_out), "a", "curve")
    output(per_neuron_panel(per_neuron), "b", "per-neuron")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(delta_sweep(summary, "delta_fluctuation_r2"), "c", "delta-fluctuation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
