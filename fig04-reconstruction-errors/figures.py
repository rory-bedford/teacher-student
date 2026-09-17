"""Figure 4 — build fig04.svg from the CSVs written by analysis.py.

uv run python fig04-reconstruction-errors/figures.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import NEURON_REMOVAL_COLOR, SYNAPSE_DROPOUT_COLOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    FIGURE_WIDTH,
    FLOOR_COLOR,
    panel_label,
    use_talk_style,
)

HERE = Path(__file__).resolve().parent
MODEL_COLORS = {
    "neuron_removal": NEURON_REMOVAL_COLOR,
    "synapse_dropout": SYNAPSE_DROPOUT_COLOR,
}
MODEL_LABELS = {
    "neuron_removal": "Neuron removal",
    "synapse_dropout": "Synapse dropout",
}
MAX_POINTS = 20000


def main(data_dir, out_path):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig04_summary.csv")
    per_neuron = pd.read_csv(data_dir / "fig04_per_neuron.csv")
    rows = summary[
        (summary["group"] == "unobserved") & (summary["metric"] == "fluctuation_r2")
    ]
    n_seeds = rows.groupby(["error_model", "level"])["seed"].nunique().min()

    fig = plt.figure(figsize=(FIGURE_WIDTH, 7.5 / 2.54), layout="constrained")
    grid = fig.add_gridspec(1, 2)

    # (a) unobserved Fluctuation R² against mean input volume lost.
    ax = fig.add_subplot(grid[0, 0])
    for model, color in MODEL_COLORS.items():
        sub = rows[rows["error_model"] == model]
        stats = sub.groupby("level")[
            [
                "mean_kappa_lost",
                "value",
                "floor_value",
                "ceiling_value",
                "noise_ceiling_value",
            ]
        ].mean()
        spread = sub.groupby("level")["value"].std().fillna(0.0)
        ax.scatter(
            sub["mean_kappa_lost"],
            sub["value"],
            s=4,
            color=color,
            alpha=0.35,
            linewidths=0,
        )
        ax.errorbar(
            stats["mean_kappa_lost"],
            stats["value"],
            yerr=spread,
            color=color,
            marker="o",
            markersize=3,
            capsize=1.5,
            label=MODEL_LABELS[model],
        )
        ax.plot(
            stats["mean_kappa_lost"],
            stats["ceiling_value"],
            ":",
            color=color,
            linewidth=1.0,
        )
        ax.plot(
            stats["mean_kappa_lost"],
            stats["noise_ceiling_value"],
            "-.",
            color=color,
            linewidth=0.8,
        )
        ax.plot(
            stats["mean_kappa_lost"],
            stats["floor_value"],
            "--",
            color=FLOOR_COLOR,
            linewidth=0.8,
        )
    ax.set_xlabel("Mean input volume lost (κ)")
    ax.set_ylabel("Fluctuation R² (unobserved)")
    ax.legend(frameon=False)
    ax.set_title("··· ceiling   -·- noise ceiling   -- floor", fontsize=6.5)
    panel_label(ax, "a")

    # (b) per-neuron Fluctuation R² against per-neuron kappa, both models pooled.
    ax = fig.add_subplot(grid[0, 1])
    neurons = per_neuron[
        (per_neuron["observed"] == 0) & (per_neuron["level"] > 0)
    ].dropna(subset=["fluctuation_r2"])
    if len(neurons) > MAX_POINTS:
        neurons = neurons.sample(MAX_POINTS, random_state=0)
    bins = np.linspace(0, 1, 21)
    for model, color in MODEL_COLORS.items():
        sub = neurons[neurons["error_model"] == model]
        ax.scatter(
            sub["kappa_lost"],
            sub["fluctuation_r2"].clip(-1, 1),
            s=1,
            color=color,
            alpha=0.15,
            linewidths=0,
            rasterized=True,
        )
        binned = sub.groupby(pd.cut(sub["kappa_lost"], bins), observed=True)[
            "fluctuation_r2"
        ].median()
        centres = [interval.mid for interval in binned.index]
        ax.plot(
            centres,
            binned.values,
            color=color,
            linewidth=1.2,
            label=f"{MODEL_LABELS[model]} (median)",
        )
    ax.set_xlabel("Per-neuron input volume lost (κ)")
    ax.set_ylabel("Per-neuron Fluctuation R²")
    ax.set_ylim(-1, 1)
    ax.legend(frameon=False, fontsize=6)
    panel_label(ax, "b")

    fig.suptitle(
        "Reconstruction errors: does it matter how you lose input?\n"
        f"(10% of retained neurons observed; held-out stimuli; mean ± SD over {n_seeds} seeds)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig04.svg")
    args = parser.parse_args()
    main(args.data, args.out)
