"""Figure 6 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig06-learnt-feedforward/figures.py

    fig06-a-curve               Fluctuation R² vs reconstructed fraction, observed / unobserved
    fig06-b-delta-fluctuation   perturbation: ΔFluctuation R² vs reconstructed fraction

Two panels (2026-09-21). The rate scatters went first -- they were the only panels
quoting Activity R², which no other panel reports -- and then the raster: the sweep
carries the result. Both tables are still written by analysis.py.

Style is the archived paper figures (``common/style.py``), sized to drop into the talk at
100%. ``placeholder_figures/fig06-learnt-feedforward/figures.py`` calls ``main`` here with
a watermark and fake CSVs, so content edits show up in both.

Groups follow analysis.py: "observed" are modelled neurons in the loss, "heldout" are
modelled neurons that are simulated and never in the loss.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from connectome_snns.visualization import OBSERVED_COLOR, UNOBSERVED_COLOR
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    METRIC_LABELS,
    PERTURBATION_LABEL,
    PERTURBATION_TITLE,
    pool_populations,
)
from common.style import (
    LEGEND_GREY,
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
FIGURE = "fig06"
GROUPS = {
    "observed": ("Observed", OBSERVED_COLOR),
    "heldout": ("Held-Out", UNOBSERVED_COLOR),
}
OPERATING_POINT = 0.1


def format_count(n):
    return (
        f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n:,}"
    )


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


def limits(sweep):
    """One y range for the sweep and the perturbation panel, as in Figures 3 to 5."""
    rows = sweep[
        sweep["metric"].isin(["fluctuation_r2", "delta_fluctuation_r2"])
        & sweep["group"].isin(["unobserved", "heldout"])
    ]
    return min(0.0, float(rows["value"].min()) - 0.05), 1.05


def curve(sweep, fully_observed, ylim):
    """(b) The sweep, with the free-parameter count annotated at each level."""
    rows = held_out(sweep)
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.35, SINGLE[1] * 1.15))
    for group, (_, color) in GROUPS.items():
        sweep_series(
            ax,
            group_rows(rows, group, "fluctuation_r2"),
            "reconstructed_fraction",
            "fluctuation_r2",
            color,
            seeds=True,
            errorbars=False,
        )
        ceiling(
            ax,
            group_rows(rows, group, "fluctuation_r2"),
            "reconstructed_fraction",
            color,
        )
    handles_extra = []
    fully_observed = held_out(fully_observed)
    if not fully_observed.empty:
        point = group_rows(fully_observed, "observed", "fluctuation_r2")
        ax.scatter(
            point["reconstructed_fraction"].mean(),
            point["value"].mean(),
            marker="X",
            s=90,
            color="k",
            zorder=4,
        )
        handles_extra.append(
            Line2D(
                [],
                [],
                marker="X",
                color="k",
                linestyle="",
                markersize=9,
                label="Fully Observed (Fluctuation)",
            )
        )
    ax.axvspan(
        OPERATING_POINT - 0.02, OPERATING_POINT + 0.02, color="#dddddd", zorder=0
    )
    ax.text(
        OPERATING_POINT,
        1.0,
        "Our Dataset",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=TICK_SIZE - 1,
        color=LEGEND_GREY,
    )
    ax.set_xlim(1.05, 0.0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Fraction of Units Reconstructed")
    ax.set_ylabel("Fluctuation R²")
    sweep_legend(
        ax,
        {label: color for label, color in GROUPS.values()},
        metrics=False,
        extra=handles_extra,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    # The free-parameter count is the mechanism: it pre-empts the objection that the
    # learnt bucket can fit anything.
    params = rows.groupby("reconstructed_fraction")["n_free_params"].mean()
    for fraction, count in params.items():
        ax.annotate(
            format_count(int(count)),
            (fraction, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=TICK_SIZE - 2,
            color=LEGEND_GREY,
        )
    ax.set_title("Held-Out and Unobserved Neurons vs Reconstructed Fraction", pad=18)
    fig.tight_layout()
    return fig


def perturbation(sweep, metric, ylim):
    """(d) The intervention against reconstructed fraction, cell types pooled.

    The targets are held-out (unobserved) I cells; the unreconstructed units are
    teacher-forced, so they are never targets. At 10% reconstruction the held-out
    population is small, so these points are noisy. E and I are pooled (see
    ``common.plotting.pool_populations``).
    """
    rows = sweep[
        (sweep["evaluation"] == "perturbation")
        & (sweep["metric"] == metric)
        # analysis.py renames the perturbation rows' "unobserved" group to "heldout".
        & (sweep["group"] == "heldout")
    ]
    pooled = pool_populations(rows, ["reconstructed_fraction", "seed"])
    base_metric = metric.replace("delta_", "")
    fig, ax = plt.subplots(figsize=SINGLE)
    sweep_series(
        ax,
        pooled,
        "reconstructed_fraction",
        base_metric,
        UNOBSERVED_COLOR,
        seeds=True,
        errorbars=False,
    )
    ceiling(ax, pooled, "reconstructed_fraction", UNOBSERVED_COLOR)
    ax.set_xlim(1.05, 0.0)
    ax.set_xlabel("Fraction of Units Reconstructed")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(*ylim)
    sweep_legend(
        ax,
        {PERTURBATION_LABEL: UNOBSERVED_COLOR},
        metrics=False,
        loc="lower left",
        bbox_to_anchor=None,
        fontsize=TICK_SIZE - 2,
    )
    ax.set_title(f"{METRIC_LABELS[metric]}\n{PERTURBATION_TITLE}")
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, decorate=None, suffix=""):
    apply_style()
    clear_panels(out_dir, FIGURE, suffix)
    summary = pd.read_csv(data_dir / "fig06_summary.csv")
    sweep = summary[summary["recorded_pool_fraction"] < 1.0]
    fully_observed = summary[summary["recorded_pool_fraction"] >= 1.0]

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    ylim = limits(sweep)
    output(curve(sweep, fully_observed, ylim), "a", "curve")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if "evaluation" in sweep and (sweep["metric"] == "delta_fluctuation_r2").any():
        output(
            perturbation(sweep, "delta_fluctuation_r2", ylim), "b", "delta-fluctuation"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.data, args.out_dir)
