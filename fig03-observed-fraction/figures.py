"""Figure 3 — build fig03.svg from the CSVs written by analysis.py.

    uv run python fig03-observed-fraction/figures.py
    uv run python fig03-observed-fraction/figures.py --scatter-fractions 0.25 0.02 0.005

Panel (b) shows rate scatters at three observed fractions: by default the lowest
fraction still within 10% of the ceiling (above threshold), the fraction closest to
half the ceiling (near), and the lowest fraction run (below). Pass
--scatter-fractions to choose them by hand once the curve is known.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connectome_snns.visualization import FLOOR_COLOR

from common.plotting import (
    FIGURE_WIDTH,
    GROUP_COLORS,
    GROUP_LABELS,
    ceiling_line,
    panel_label,
    r2_title,
    rate_scatter,
    seed_errorbar,
    use_talk_style,
)

HERE = Path(__file__).resolve().parent
N_NEURONS = 5000


def default_scatter_fractions(summary):
    rows = summary[
        (summary["group"] == "unobserved") & (summary["metric"] == "fluctuation_r2")
    ]
    stats = rows.groupby("obs_fraction")[["value", "ceiling_value"]].mean()
    normalised = stats["value"] / stats["ceiling_value"]
    fractions = sorted(stats.index)
    comfortable = [f for f in fractions if normalised[f] >= 0.9]
    above = comfortable[0] if comfortable else fractions[-1]
    near = float((normalised - 0.5).abs().idxmin())
    below = fractions[0]
    return [above, near, below]


def main(data_dir, out_path, scatter_fractions):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig03_summary.csv")
    rates = pd.read_csv(data_dir / "fig03_rates.csv")
    dimensionality = pd.read_csv(data_dir / "fig03_dimensionality.csv").iloc[0]
    n_seeds = summary.groupby("obs_fraction")["seed"].nunique().min()

    fig = plt.figure(figsize=(FIGURE_WIDTH, 13 / 2.54), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0])

    # (a) Fluctuation R² against observed fraction.
    ax = fig.add_subplot(grid[0, :])
    for group in ("observed", "unobserved"):
        seed_errorbar(
            ax,
            summary,
            "obs_fraction",
            group,
            "fluctuation_r2",
            GROUP_COLORS[group],
            GROUP_LABELS[group],
        )
        ceiling_line(
            ax, summary, "obs_fraction", "fluctuation_r2", group, GROUP_COLORS[group]
        )
    pr_fraction = dimensionality["participation_ratio"] / N_NEURONS
    ax.axvline(pr_fraction, color=FLOOR_COLOR, linewidth=0.8)
    ax.text(
        pr_fraction,
        0.02,
        f" participation ratio\n {dimensionality['participation_ratio']:.0f} neurons",
        transform=ax.get_xaxis_transform(),
        fontsize=6,
        color="#555555",
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Fluctuation R²")
    ax.legend(frameon=False, loc="lower right")
    top = ax.secondary_xaxis(
        "top", functions=(lambda f: f * N_NEURONS, lambda n: n / N_NEURONS)
    )
    top.set_xlabel("Observed neurons")
    ax.set_title(
        "··· ceiling (perfectly specified student)",
        fontsize=6.5,
    )
    panel_label(ax, "a")

    # (b) scatters at three fractions.
    fractions = scatter_fractions or default_scatter_fractions(summary)
    seed = int(rates["seed"].min())
    for column, fraction in enumerate(fractions):
        ax = fig.add_subplot(grid[1, column])
        subset = rates[
            np.isclose(rates["obs_fraction"], fraction)
            & (rates["seed"] == seed)
            & (rates["observed"] == 0)
        ]
        rows = summary[np.isclose(summary["obs_fraction"], fraction)]
        n_observed = int(rows["n_observed"].iloc[0])
        rate_scatter(
            ax,
            subset,
            r2_title(f"Unobserved, {n_observed} observed", rows, "unobserved"),
        )
        ax.title.set_fontsize(5)
        if column == 0:
            panel_label(ax, "b")

    fig.suptitle(
        "How few neurons do you need to observe?\n"
        f"(full reconstruction; held-out stimuli; mean ± SD over {n_seeds} seeds; scatters seed {seed})"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig03.svg")
    parser.add_argument("--scatter-fractions", type=float, nargs=3, default=None)
    args = parser.parse_args()
    main(args.data, args.out, args.scatter_fractions)
