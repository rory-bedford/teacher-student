"""Figure 6 — one SVG per panel from the CSVs written by analysis.py.

    uv run python fig06-learnt-feedforward/figures.py
    uv run python fig06-learnt-feedforward/figures.py --scatter-fractions 0.5 0.1

    fig06-a1-scatter-<pct>      firing rates, observed | held-out, first scatter level
    fig06-a2-scatter-<pct>      same, second scatter level
    fig06-b-curve               R² vs reconstructed fraction, observed / held-out
    fig06-c-raster              observed and held-out neuron at the lowest level
    fig06-d-delta-fluctuation   perturbation: ΔFluctuation R² vs reconstructed fraction

Activity R² is scored and kept in the CSVs but plotted only as the scatter panels' R²
(2026-09-18): the sweeps report Fluctuation R² alone.

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
import numpy as np
import pandas as pd
from connectome_snns.visualization import OBSERVED_COLOR, UNOBSERVED_COLOR
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import METRIC_LABELS, PERTURBATION_SERIES
from common.style import (
    LEGEND_GREY,
    PAIR,
    SINGLE,
    TICK_SIZE,
    WIDE,
    apply_style,
    ceiling,
    nice_max,
    rate_scatter,
    save,
    spike_raster,
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
RASTER_SECONDS = 2.0


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


def scatters(sweep, rates, fraction, seed):
    """(a) Teacher-vs-student rates, observed beside held-out, at one level."""
    level = rates[
        np.isclose(rates["reconstructed_fraction"], fraction)
        & (rates["recorded_pool_fraction"] < 1.0)
        & (rates["seed"] == seed)
    ]
    max_rate = nice_max(level[["teacher_rate_hz", "student_rate_hz"]].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=PAIR)
    for ax, (group, (label, _)) in zip(axes, GROUPS.items()):
        r2 = group_rows(
            sweep[np.isclose(sweep["reconstructed_fraction"], fraction)],
            group,
            "activity_r2",
        )["value"].mean()
        rate_scatter(
            ax, level[level["group"] == group], f"{label} (R² = {r2:.3f})", max_rate
        )
    fig.suptitle(
        f"Degeneracy Under Partial Reconstruction ({fraction:.0%} Reconstructed)"
    )
    fig.tight_layout()
    return fig


def curve(sweep, fully_observed):
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
    ax.set_ylim(min(0.0, rows["value"].min() - 0.05), 1.05)
    ax.set_xlabel("Fraction of Units Reconstructed")
    ax.set_ylabel("Firing Rate R²")
    sweep_legend(
        ax, {label: color for label, color in GROUPS.values()}, extra=handles_extra
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
    ax.set_title("Unreconstructed Inputs Break Prediction", pad=18)
    fig.tight_layout()
    return fig


def raster(spikes, level):
    """(c) One observed and one held-out neuron at the lowest level."""
    level_spikes = spikes[np.isclose(spikes["reconstructed_fraction"], level)]
    neurons = (
        level_spikes[["neuron_id", "group"]]
        .drop_duplicates()
        .sort_values("group", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(WIDE[0], WIDE[1] * 0.7))
    spike_raster(
        ax,
        level_spikes[level_spikes["time_s"] <= RASTER_SECONDS],
        list(neurons["neuron_id"]),
        RASTER_SECONDS,
        [GROUPS[g][0] for g in neurons["group"]],
    )
    ax.set_title(f"Student vs Teacher Raster ({level:.0%} Reconstructed)")
    return fig


def perturbation(sweep, metric):
    """The intervention against reconstructed fraction, one population per colour.

    The targets are held-out (unobserved) I cells; the unreconstructed units are
    teacher-forced, so they are never targets. At 10% reconstruction the held-out
    population is small, so these points are noisy.
    """
    rows = sweep[sweep["evaluation"] == "perturbation"]
    # analysis.py renames the perturbation rows' "unobserved" group to "heldout".
    series = [
        ("heldout", ct, color, label) for _, ct, color, label in PERTURBATION_SERIES
    ]
    base_metric = metric.replace("delta_", "")
    fig, ax = plt.subplots(figsize=SINGLE)
    labels = {}
    for group, cell_type, color, label in series:
        subset = group_rows(rows, group, metric, cell_type)
        sweep_series(ax, subset, "reconstructed_fraction", base_metric, color)
        ceiling(ax, subset, "reconstructed_fraction", color)
        labels[label.replace("Unobserved", "Held-Out")] = color
    ax.set_xlim(1.05, 0.0)
    ax.set_xlabel("Fraction of Units Reconstructed")
    ax.set_ylabel(METRIC_LABELS[metric])
    sweep_legend(ax, labels, metrics=False)
    ax.set_title(f"{METRIC_LABELS[metric]}\nInhibiting 25% of Held-Out I Cells")
    fig.tight_layout()
    return fig


def main(data_dir, out_dir, scatter_fractions=None, decorate=None, suffix=""):
    apply_style()
    summary = pd.read_csv(data_dir / "fig06_summary.csv")
    rates = pd.read_csv(data_dir / "fig06_rates.csv")
    spikes = pd.read_csv(data_dir / "fig06_spikes.csv")
    sweep = summary[summary["recorded_pool_fraction"] < 1.0]
    fully_observed = summary[summary["recorded_pool_fraction"] >= 1.0]
    levels = sorted(held_out(sweep)["reconstructed_fraction"].unique())

    def output(fig, letter, slug):
        save(fig, out_dir, FIGURE, letter, slug, suffix, decorate)

    if scatter_fractions is None:
        scatter_fractions = [levels[len(levels) // 2], min(levels)]
    # A CSV with a single level (the pilot) would otherwise draw the same panel twice.
    scatter_fractions = list(dict.fromkeys(scatter_fractions))
    seed = int(rates["seed"].min())
    for index, fraction in enumerate(scatter_fractions, start=1):
        fig = scatters(sweep, rates, fraction, seed)
        output(fig, f"a{index}", f"scatter-{fraction * 100:.0f}pct")

    output(curve(sweep, fully_observed), "b", "curve")
    output(raster(spikes, min(levels)), "c", "raster")

    # The perturbation panels are separate files, so dropping them from the talk is
    # dropping two SVGs.
    if "evaluation" in sweep and (sweep["metric"] == "delta_fluctuation_r2").any():
        output(perturbation(sweep, "delta_fluctuation_r2"), "d", "delta-fluctuation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--scatter-fractions", type=float, nargs=2, default=None)
    args = parser.parse_args()
    main(args.data, args.out_dir, args.scatter_fractions)
