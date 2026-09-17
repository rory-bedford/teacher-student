"""Figure 2 — build fig02.svg from the CSVs written by analysis.py.

uv run python fig02-controls/figures.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import (
    CONFIGURATION_MODEL_COLOR,
    FLOOR_COLOR,
    FULL_CONNECTOME_COLOR,
    LEARNT_RECURRENCE_COLOR,
    SHUFFLE_WEIGHTS_COLOR,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    FIGURE_WIDTH,
    METRIC_LABELS,
    panel_label,
    use_talk_style,
)
from common.structure import (
    configuration_model_rewire,
    shuffle_weights_within_connectome,
)

HERE = Path(__file__).resolve().parent
VARIANTS = [
    "full_connectome",
    "learnt_recurrence",
    "shuffle_weights",
    "configuration_model",
]
VARIANT_LABELS = {
    "full_connectome": "Full\nconnectome",
    "learnt_recurrence": "Learnt\nrecurrence",
    "shuffle_weights": "Shuffle\nweights",
    "configuration_model": "Config.-model\nrewire",
}
VARIANT_COLORS = {
    "full_connectome": FULL_CONNECTOME_COLOR,
    "learnt_recurrence": LEARNT_RECURRENCE_COLOR,
    "shuffle_weights": SHUFFLE_WEIGHTS_COLOR,
    "configuration_model": CONFIGURATION_MODEL_COLOR,
}


def format_count(n):
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n:,}"


def grouped_bars(ax, summary, metric):
    rows = summary[summary["metric"] == metric]
    variants = [v for v in VARIANTS if v in set(rows["variant"])]
    width = 0.38
    for g, (group, alpha, hatch) in enumerate(
        (("observed", 0.45, None), ("unobserved", 1.0, None))
    ):
        for i, variant in enumerate(variants):
            sub = rows[(rows["variant"] == variant) & (rows["group"] == group)]
            x = i + (g - 0.5) * width
            ax.bar(
                x,
                sub["value"].mean(),
                width,
                color=VARIANT_COLORS[variant],
                alpha=alpha,
                hatch=hatch,
            )
            ax.errorbar(
                x,
                sub["value"].mean(),
                yerr=sub["value"].std() if len(sub) > 1 else 0,
                color="k",
                capsize=1.5,
                linewidth=0.6,
            )
            ax.scatter(
                np.full(len(sub), x),
                sub["value"],
                s=3,
                color="k",
                zorder=3,
                linewidths=0,
            )
            ax.hlines(
                sub["floor_value"].mean(),
                x - width / 2,
                x + width / 2,
                colors=FLOOR_COLOR,
                linestyles="--",
                linewidth=0.8,
            )
            ax.hlines(
                sub["ceiling_value"].mean(),
                x - width / 2,
                x + width / 2,
                colors="k",
                linestyles=":",
                linewidth=0.8,
            )
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel(METRIC_LABELS[metric])
    return variants


def schematic(fig, cell, params_by_variant):
    """Toy connectivity matrix under each variant, generated with the real functions."""
    rng = np.random.default_rng(3)
    n_e, n_i = 16, 4
    types = np.array([0] * n_e + [1] * n_i)
    assemblies = np.repeat(np.arange(4), 4)
    probability = np.where(np.equal.outer(assemblies, assemblies), 0.6, 0.08)
    toy = np.zeros((n_e + n_i, n_e + n_i), dtype=np.float32)
    toy[:n_e, :n_e] = (rng.random((n_e, n_e)) < probability) * rng.gamma(
        2.0, 0.5, (n_e, n_e)
    )
    toy[n_e:, :] = (rng.random((n_i, n_e + n_i)) < 0.5) * rng.gamma(
        2.0, 1.0, (n_i, n_e + n_i)
    )
    toy[:n_e, n_e:] = (rng.random((n_e, n_i)) < 0.4) * rng.gamma(2.0, 0.5, (n_e, n_i))
    np.fill_diagonal(toy, 0)

    matrices = {
        "full_connectome": toy,
        "learnt_recurrence": np.ones_like(toy),
        "shuffle_weights": shuffle_weights_within_connectome(
            toy, types, np.random.default_rng(1)
        ),
        "configuration_model": configuration_model_rewire(
            toy, types, np.random.default_rng(1)
        ),
    }
    grid = cell.subgridspec(1, 4, wspace=0.1)
    for k, variant in enumerate(VARIANTS):
        ax = fig.add_subplot(grid[0, k])
        matrix = matrices[variant]
        cmap = "Greys" if variant != "learnt_recurrence" else "Oranges"
        ax.imshow(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=np.percentile(toy[toy > 0], 90),
            interpolation="nearest",
        )
        ax.axhline(n_e - 0.5, color=VARIANT_COLORS[variant], linewidth=0.8)
        ax.axvline(n_e - 0.5, color=VARIANT_COLORS[variant], linewidth=0.8)
        count = params_by_variant.get(variant)
        ax.set_title(
            VARIANT_LABELS[variant].replace("\n", " ")
            + (f"\n{format_count(count)} params" if count is not None else ""),
            fontsize=6.5,
            color=VARIANT_COLORS[variant],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if k == 0:
            panel_label(ax, "c")


def main(data_dir, out_path):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig02_summary.csv")
    params_by_variant = summary.groupby("variant")["n_free_params"].first().to_dict()
    n_seeds = summary.groupby("variant")["seed"].nunique().min()

    fig = plt.figure(figsize=(FIGURE_WIDTH, 17 / 2.54), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.9, 0.7])

    ax = fig.add_subplot(grid[0])
    grouped_bars(ax, summary, "fluctuation_r2")
    ax.set_title(
        "Fluctuation R², held-out stimuli (light: observed, dark: unobserved;"
        " -- floor, ··· ceiling)",
        fontsize=7,
    )
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[1])
    grouped_bars(ax, summary, "activity_r2")
    ax.set_title("Activity R²", fontsize=7)
    panel_label(ax, "b")

    schematic(fig, grid[2], params_by_variant)

    fig.suptitle(
        "The connectome is what's doing the work: 6 parameters with it beat 25M without\n"
        f"(10% observed; mean ± SD over {n_seeds} seeds)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig02.svg")
    args = parser.parse_args()
    main(args.data, args.out)
