"""Figure 5 — build fig05.svg from the CSVs written by analysis.py.

    uv run python fig05-weight-noise/figures.py

Panel (b) reads Figure 4's neuron-removal curve from
../fig04-reconstruction-errors/fig04_summary.csv, so run Figure 4's analysis first.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from connectome_snns.visualization import NEURON_REMOVAL_COLOR, WEIGHT_NOISE_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    FIGURE_WIDTH,
    GROUP_COLORS,
    GROUP_LABELS,
    ceiling_line,
    panel_label,
    seed_errorbar,
    use_talk_style,
)

HERE = Path(__file__).resolve().parent
FIG04_SUMMARY = HERE.parent / "fig04-reconstruction-errors" / "fig04_summary.csv"


def main(data_dir, fig04_summary, out_path):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig05_summary.csv")
    n_seeds = summary.groupby("weight_noise")["seed"].nunique().min()
    clipped = summary.groupby("weight_noise")["noise_clipped_fraction"].mean()

    fig = plt.figure(figsize=(FIGURE_WIDTH, 11.5 / 2.54), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # (a) Fluctuation R² against weight noise.
    ax = fig.add_subplot(grid[0, :])
    for group in ("observed", "unobserved"):
        seed_errorbar(
            ax,
            summary,
            "weight_noise",
            group,
            "fluctuation_r2",
            GROUP_COLORS[group],
            GROUP_LABELS[group],
        )
        ceiling_line(
            ax, summary, "weight_noise", "fluctuation_r2", group, GROUP_COLORS[group]
        )
    ax.set_xlabel("Weight noise")
    ax.set_ylabel("Fluctuation R²")
    ax.legend(frameon=False)
    ax.set_title(
        "··· ceiling\nweights clipped at zero: "
        + ", ".join(f"{100 * v:.1f}% @ {k:g}" for k, v in clipped.items() if k > 0),
        fontsize=6,
    )
    panel_label(ax, "a")

    # (b) the contrast: weight noise vs neuron removal, shared y.
    left = fig.add_subplot(grid[1, 0])
    rows = summary[
        (summary["group"] == "unobserved") & (summary["metric"] == "fluctuation_r2")
    ]
    stats = rows.groupby("weight_noise")["value"].agg(["mean", "std"])
    left.errorbar(
        stats.index,
        stats["mean"],
        yerr=stats["std"].fillna(0),
        color=WEIGHT_NOISE_COLOR,
        marker="o",
        markersize=3,
        capsize=1.5,
    )
    left.set_xlabel("Weight noise")
    left.set_ylabel("Fluctuation R² (unobserved)")
    left.set_title("Imprecise weights")
    panel_label(left, "b")

    right = fig.add_subplot(grid[1, 1], sharey=left)
    if Path(fig04_summary).exists():
        fig04 = pd.read_csv(fig04_summary)
        removal = fig04[
            (fig04["error_model"] == "neuron_removal")
            & (fig04["group"] == "unobserved")
            & (fig04["metric"] == "fluctuation_r2")
        ]
        stats = removal.groupby("level")[["mean_kappa_lost", "value"]].mean()
        spread = removal.groupby("level")["value"].std().fillna(0)
        right.errorbar(
            stats["mean_kappa_lost"],
            stats["value"],
            yerr=spread,
            color=NEURON_REMOVAL_COLOR,
            marker="o",
            markersize=3,
            capsize=1.5,
        )
    else:
        right.text(
            0.5, 0.5, "run fig04 analysis.py", transform=right.transAxes, ha="center"
        )
    right.set_xlabel("Input volume lost (κ)")
    right.set_title("Missing connections")
    right.tick_params(labelleft=False)

    fig.suptitle(
        "Weight precision is not the binding constraint\n"
        f"(10% observed; held-out stimuli; mean ± SD over {n_seeds} seeds)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--fig04-summary", type=Path, default=FIG04_SUMMARY)
    parser.add_argument("--out", type=Path, default=HERE / "fig05.svg")
    args = parser.parse_args()
    main(args.data, args.fig04_summary, args.out)
