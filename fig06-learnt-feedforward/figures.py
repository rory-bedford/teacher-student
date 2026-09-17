"""Figure 6 — build fig06.svg from the CSVs written by analysis.py.

uv run python fig06-learnt-feedforward/figures.py
uv run python fig06-learnt-feedforward/figures.py --scatter-fractions 0.5 0.1
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import OBSERVED_COLOR, UNOBSERVED_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    FIGURE_WIDTH,
    panel_label,
    r2_title,
    rate_scatter,
    spike_raster,
    use_talk_style,
)

HERE = Path(__file__).resolve().parent
COLORS = {"observed": OBSERVED_COLOR, "heldout": UNOBSERVED_COLOR}
LABELS = {"observed": "Observed (in loss)", "heldout": "Held-out"}
OPERATING_POINT = 0.1
RASTER_SECONDS = 3.0


def format_count(n):
    return (
        f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n:,}"
    )


def main(data_dir, out_path, scatter_fractions):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig06_summary.csv")
    rates = pd.read_csv(data_dir / "fig06_rates.csv")
    spikes = pd.read_csv(data_dir / "fig06_spikes.csv")

    sweep = summary[summary["recorded_pool_fraction"] < 1.0]
    fully_observed = summary[summary["recorded_pool_fraction"] >= 1.0]
    n_seeds = sweep.groupby("reconstructed_fraction")["seed"].nunique().min()

    fig = plt.figure(figsize=(FIGURE_WIDTH, 21 / 2.54), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 1.0, 0.55])

    # (b) the sweep.
    ax = fig.add_subplot(grid[0, :])
    flu = sweep[sweep["metric"] == "fluctuation_r2"]
    for group in ("observed", "heldout"):
        rows = flu[flu["group"] == group]
        stats = rows.groupby("reconstructed_fraction")[
            ["value", "ceiling_value"]
        ].mean()
        spread = rows.groupby("reconstructed_fraction")["value"].std().fillna(0.0)
        ax.scatter(
            rows["reconstructed_fraction"],
            rows["value"],
            s=4,
            color=COLORS[group],
            alpha=0.35,
            linewidths=0,
        )
        ax.errorbar(
            stats.index,
            stats["value"],
            yerr=spread,
            color=COLORS[group],
            marker="o",
            markersize=3,
            capsize=1.5,
            label=LABELS[group],
        )
        ax.plot(
            stats.index, stats["ceiling_value"], ":", color=COLORS[group], linewidth=1.0
        )
    if not fully_observed.empty:
        point = fully_observed[fully_observed["metric"] == "fluctuation_r2"]
        point = point[point["group"] == "observed"]
        ax.scatter(
            point["reconstructed_fraction"],
            point["value"],
            marker="x",
            s=25,
            color="k",
            zorder=4,
            label="Fully observed (all modelled neurons in loss)",
        )
    ax.axvline(OPERATING_POINT, color="#bbbbbb", linewidth=4, alpha=0.4, zorder=0)
    ax.text(
        OPERATING_POINT,
        0.03,
        "  our dataset\n  (~10%)",
        transform=ax.get_xaxis_transform(),
        fontsize=6,
        color="#777777",
    )
    ax.set_xlim(1.05, 0.0)
    ax.set_xlabel("Fraction of units reconstructed")
    ax.set_ylabel("Fluctuation R²")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    kappa = sweep.groupby("reconstructed_fraction")["kappa"].mean()
    params = sweep.groupby("reconstructed_fraction")["n_free_params"].mean()
    top = ax.secondary_xaxis("top")
    top.set_xticks(kappa.index)
    top.set_xticklabels(
        [f"{k:.2f}\n{format_count(params[f])}" for f, k in kappa.items()], fontsize=6
    )
    top.set_xlabel("κ (known fraction of input volume) / free parameters", fontsize=7)
    panel_label(ax, "b")

    # (a) scatters at two levels: observed vs held-out.
    levels = sorted(sweep["reconstructed_fraction"].unique())
    if scatter_fractions is None:
        scatter_fractions = [levels[len(levels) // 2], min(levels)]
    seed = int(rates["seed"].min())
    for column, (fraction, group) in enumerate(
        (f, g) for f in scatter_fractions for g in ("observed", "heldout")
    ):
        ax = fig.add_subplot(grid[1 + column // 2, column % 2])
        subset = rates[
            np.isclose(rates["reconstructed_fraction"], fraction)
            & (rates["recorded_pool_fraction"] < 1.0)
            & (rates["seed"] == seed)
            & (rates["group"] == group)
        ]
        rows = sweep[np.isclose(sweep["reconstructed_fraction"], fraction)]
        rate_scatter(
            ax, subset, r2_title(f"{LABELS[group]}, {fraction:.0%} recon.", rows, group)
        )
        ax.title.set_fontsize(6)
        ax.xaxis.label.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)
        if column == 0:
            panel_label(ax, "a")

    # (c) raster at the operating point.
    ax = fig.add_subplot(grid[3, :])
    level_spikes = spikes[np.isclose(spikes["reconstructed_fraction"], min(levels))]
    neurons = (
        level_spikes[["neuron_id", "group"]]
        .drop_duplicates()
        .sort_values("group", ascending=False)
    )
    spike_raster(
        ax,
        level_spikes[level_spikes["time_s"] <= RASTER_SECONDS].assign(
            observed=lambda d: (d["group"] == "observed").astype(int)
        ),
        [
            (n, int(g == "observed"))
            for n, g in neurons.itertuples(index=False, name=None)
        ],
        RASTER_SECONDS,
    )
    ax.set_title(
        f"{min(levels):.0%} reconstructed: teacher (grey) vs student", fontsize=7
    )
    panel_label(ax, "c")

    fig.suptitle(
        "Unreconstructed inputs break prediction of unobserved neurons\n"
        f"(fixed 50% recorded pool; held-out stimuli; mean ± SD over {n_seeds} seeds; ··· ceiling)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig06.svg")
    parser.add_argument("--scatter-fractions", type=float, nargs=2, default=None)
    args = parser.parse_args()
    main(args.data, args.out, args.scatter_fractions)
