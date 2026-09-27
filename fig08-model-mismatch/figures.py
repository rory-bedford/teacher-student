"""Figure 8 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig08-model-mismatch/figures.py

    fig08-a-curve               Fluctuation R² vs threshold heterogeneity (SD),
                                observed / unobserved
    fig08-b-delta-fluctuation   perturbation: ΔFluctuation R² vs threshold heterogeneity
    fig08-c-scaling-factors     learnt / true scaling factor vs threshold heterogeneity

Heterogeneity 0 is Figure 1's runs. The ceiling is the perfectly specified student, which
has the teacher's thresholds whatever the student's are, so it is one constant per
population (Figure 1's, averaged over seeds) drawn as a horizontal dotted line rather
than a curve.

Activity R² is scored and kept in the CSVs but not plotted, as in Figures 3-5.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, NullFormatter, ScalarFormatter

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
    EXCITATORY,
    INHIBITORY,
    OBSERVED,
    REFERENCE_GREY,
    TICK_SIZE,
    UNOBSERVED,
    apply_style,
    clear_panels,
    performance_axis,
    performance_limits,
    save,
    sweep_layout,
    sweep_legend,
    sweep_panel,
    sweep_series,
)

HERE = Path(__file__).resolve().parent
FIGURE = "fig08"
GROUPS = {
    "observed": ("Observed", OBSERVED),
    "unobserved": ("Unobserved", UNOBSERVED),
}
X = "threshold_heterogeneity"
X_LABEL = "Threshold Heterogeneity, SD (mV)"
SHORT_POPULATION = {"mitral": "M", "excitatory": "E", "inhibitory": "I"}
#: Each factor coloured by its presynaptic population (as Figure 1's panel), and the
#: target population told apart by marker; every line joining points is dashed.
SOURCE_COLORS = {
    "mitral": REFERENCE_GREY,
    "excitatory": EXCITATORY,
    "inhibitory": INHIBITORY,
}
TARGET_MARKERS = {"excitatory": "o", "inhibitory": "^"}


def group_rows(summary, group, metric, cell_type="all"):
    rows = summary[(summary["group"] == group) & (summary["metric"] == metric)]
    return rows[rows["cell_type"] == cell_type]


def constant_ceiling(ax, rows, color):
    """The ceiling at heterogeneity 0 (Figure 1), as one horizontal dotted line."""
    baseline = rows[np.isclose(rows[X], 0.0)]
    if baseline.empty:
        return
    ax.axhline(
        baseline["ceiling_value"].mean(),
        linestyle=":",
        color=color,
        linewidth=1.2,
        alpha=0.7,
    )


def limits(summary):
    """One y range for both performance panels."""
    values = []
    for group in ("observed", "unobserved"):
        values += plotted_performance_values(summary, group, [X, "seed"])
    return performance_limits(values)


def x_axis(ax, rows):
    ax.set_xlim(-0.2, rows[X].max() + 0.2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel(X_LABEL)


def curve(summary, ylim):
    """(a) Fluctuation R² against threshold heterogeneity, per population."""
    rows = summary[summary["evaluation"] == "held_out"]
    fig, ax = sweep_panel()
    for group, (_, color) in GROUPS.items():
        series = group_rows(rows, group, "fluctuation_r2")
        sweep_series(
            ax, series, X, "fluctuation_r2", color, seeds=True, errorbars=False
        )
        constant_ceiling(ax, series, color)
    x_axis(ax, rows)
    performance_axis(ax, ylim)
    ax.set_ylabel(METRIC_LABELS["fluctuation_r2"])
    sweep_legend(
        ax,
        {label: color for label, color in GROUPS.values()},
        metrics=False,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(HELD_OUT_TITLE)
    sweep_layout(fig)
    return fig


def perturbation(summary, ylim):
    """(b) The intervention's effect against threshold heterogeneity, per population.

    E and I pooled within each; the unobserved series excludes the targeted cells.
    """
    metric = "delta_fluctuation_r2"
    rows = summary[
        (summary["evaluation"] == "perturbation") & (summary["metric"] == metric)
    ]
    fig, ax = sweep_panel()
    for group, (_, color) in GROUPS.items():
        pooled = pool_populations(rows[rows["group"] == group], [X, "seed"])
        if pooled.empty:
            continue
        sweep_series(
            ax, pooled, X, "fluctuation_r2", color, seeds=True, errorbars=False
        )
        constant_ceiling(ax, pooled, color)
    x_axis(ax, rows)
    performance_axis(ax, ylim)
    ax.set_ylabel(METRIC_LABELS[metric])
    sweep_legend(
        ax,
        {
            PERTURBATION_GROUP_LABELS[group]: color
            for group, (_, color) in GROUPS.items()
        },
        metrics=False,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(PERTURBATION_TITLE)
    sweep_layout(fig)
    return fig


def scaling_factors(factors):
    """(c) Learnt / true scaling factor against threshold heterogeneity, one line each.

    Scaling factor 1.0 is the true model at every level, since the weights are the
    teacher's; a factor away from 1 is the six parameters compensating for the wrong
    thresholds.
    """
    rows = factors.assign(ratio=factors["value"] / factors["target"])
    fig, ax = sweep_panel()
    for name in sorted(rows["scaling_factor"].unique()):
        source, target = name.split("_to_")
        stats = rows[rows["scaling_factor"] == name].groupby(X)["ratio"].mean()
        ax.plot(
            stats.index,
            stats.values,
            marker=TARGET_MARKERS[target],
            markersize=5,
            linewidth=1.5,
            color=SOURCE_COLORS[source],
            linestyle="--",
            label=f"{SHORT_POPULATION[source]}→{SHORT_POPULATION[target]}",
        )
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_ylim(0.2, 5.0)
    x_axis(ax, rows)
    ax.set_ylabel("Learnt / True Scaling Factor")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title("Scaling Factors")
    sweep_layout(fig)
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig08_summary.csv")

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(summary)
    output(curve(summary, ylim), "a", "curve")
    if (summary["metric"] == "delta_fluctuation_r2").any():
        output(perturbation(summary, ylim), "b", "delta-fluctuation")
    factor_path = data_dir / "fig08_scaling_factors.csv"
    if factor_path.exists():
        output(scaling_factors(pd.read_csv(factor_path)), "c", "scaling-factors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
