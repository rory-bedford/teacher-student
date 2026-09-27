"""Figure 4 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig04-reconstruction-errors/figures.py

    fig04-a-curve-neuron-removal       held-out Fluctuation R² vs recurrent input lost to
                                       neuron removal, observed / unobserved
    fig04-b-curve-synapse-dropout      the same for synapse dropout
    fig04-c-delta-neuron-removal       perturbation ΔFluctuation R², neuron removal
    fig04-d-delta-synapse-dropout      perturbation ΔFluctuation R², synapse dropout

One panel per error model, each with observed and unobserved neurons in the colours every
other sweep uses, so the figure reads like Figures 3, 5, 7 and 8. The four panels share
their axes, so the two error models can be compared panel against panel.

Style is the shared slide style (``common/style.py``), sized to drop into the talk at 100%.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    HELD_OUT_TITLE,
    METRIC_LABELS,
    PERTURBATION_GROUP_LABELS,
    PERTURBATION_TITLE,
    plotted_performance_values,
    pool_populations,
)
from common.style import (
    OBSERVED,
    TICK_SIZE,
    UNOBSERVED,
    apply_style,
    ceiling,
    clear_panels,
    estimate_band,
    performance_axis,
    performance_limits,
    save,
    sweep_layout,
    sweep_legend,
    sweep_panel,
    sweep_series,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig04"
#: Error model -> (x-axis wording, file slug).
MODELS = {
    "neuron_removal": ("Neuron Removal", "neuron-removal"),
    "synapse_dropout": ("Synapse Dropout", "synapse-dropout"),
}
GROUPS = {
    "observed": ("Observed", OBSERVED),
    "unobserved": ("Unobserved", UNOBSERVED),
}
#: Our estimate for the real dataset (2026-09-27): about 15% of recurrent input lost to
#: reconstruction errors.
REAL_DATA_ESTIMATE = (0.125, 0.175)


def limits(summary):
    """One y range for all four panels, so they read on the same scale."""
    return performance_limits(
        [
            value
            for group in GROUPS
            for value in plotted_performance_values(
                summary, group, ["error_model", "level", "seed"]
            )
        ]
    )


def snapped(rows):
    """x = the input volume actually lost, snapped to the nominal 0.1 grid.

    Plotted against the volume lost rather than the nominal level, since the two error
    models reach the same fraction at different levels; the realised values sit within
    0.015 of the grid, so snapping lines the points up without moving them visibly.
    """
    return rows.assign(kappa_snapped=(rows["mean_kappa_lost"] * 10).round() / 10)


def finish(ax, model, ylim, labels, title):
    ax.set_xlabel(f"Recurrent Input Lost to {MODELS[model][0]}")
    performance_axis(ax, ylim)
    estimate_band(ax, REAL_DATA_ESTIMATE)
    sweep_legend(
        ax,
        {labels[group]: color for group, (_, color) in GROUPS.items()},
        metrics=False,
        estimate=True,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(title)


def curve(summary, model, ylim):
    """(a, b) Held-out Fluctuation R² against input lost to one error model."""
    rows = summary[
        (summary["evaluation"] == "held_out")
        & (summary["metric"] == "fluctuation_r2")
        & (summary["cell_type"] == "all")
        & (summary["error_model"] == model)
    ]
    fig, ax = sweep_panel()
    for group, (_, color) in GROUPS.items():
        series = snapped(rows[rows["group"] == group])
        sweep_series(
            ax,
            series,
            "kappa_snapped",
            "fluctuation_r2",
            color,
            seeds=True,
            errorbars=False,
            x_group="level",
        )
        ceiling(ax, series, "kappa_snapped", color, x_group="level")
    ax.set_ylabel(METRIC_LABELS["fluctuation_r2"])
    finish(
        ax, model, ylim, {g: label for g, (label, _) in GROUPS.items()}, HELD_OUT_TITLE
    )
    sweep_layout(fig)
    return fig


def delta_sweep(summary, model, ylim):
    """(c, d) Perturbation ΔFluctuation R² against input lost to one error model.

    E and I pooled within each population (see ``common.plotting.pool_populations``);
    the unobserved series excludes the targeted cells.
    """
    metric = "delta_fluctuation_r2"
    rows = summary[
        (summary["evaluation"] == "perturbation")
        & (summary["metric"] == metric)
        & (summary["error_model"] == model)
    ]
    fig, ax = sweep_panel()
    for group, (_, color) in GROUPS.items():
        sub = rows[rows["group"] == group]
        if sub.empty:
            continue
        pooled = snapped(
            pool_populations(sub, ["level", "seed"]).merge(
                sub.groupby("level")["mean_kappa_lost"].mean().reset_index(),
                on="level",
            )
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
    ax.set_ylabel(METRIC_LABELS[metric])
    finish(ax, model, ylim, PERTURBATION_GROUP_LABELS, PERTURBATION_TITLE)
    sweep_layout(fig)
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig04_summary.csv")

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    for letter, model in zip("ab", MODELS):
        output(curve(summary, model, ylim), letter, f"curve-{MODELS[model][1]}")
    if (summary["metric"] == "delta_fluctuation_r2").any():
        for letter, model in zip("cd", MODELS):
            output(
                delta_sweep(summary, model, ylim), letter, f"delta-{MODELS[model][1]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
