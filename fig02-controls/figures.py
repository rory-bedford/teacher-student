"""Figure 2 — build fig02.svg from the CSVs written by analysis.py.

uv run python fig02-controls/figures.py

Plots ``PLOTTED_VARIANTS``, a subset of the variants analysis.py scores: the weight
shuffle is trained and scored but not shown (see the constant).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import (
    CONFIGURATION_MODEL_COLOR,
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
    shuffle_weights_within_neuron,
)

HERE = Path(__file__).resolve().parent
#: Plotted bars — a SUBSET of what analysis.py scores. "shuffle_inputs" (shuffle weights
#: within neuron) is trained on every seed and stays in fig02_summary.csv / fig02_rates.csv,
#: but is not plotted (2026-09-18): it overlaps with Figure 5's weight noise, which makes
#: the same point as a graded curve. Add it back to this list to restore the bar.
PLOTTED_VARIANTS = [
    "full_connectome",
    "learnt_recurrence",
    "configuration_model",
]
VARIANT_LABELS = {
    "full_connectome": "Full\nconnectome",
    "learnt_recurrence": "Learnt\nrecurrence",
    "shuffle_inputs": "Shuffled\nweights",
    "configuration_model": "Config.-model\nrewire",
}
VARIANT_COLORS = {
    "full_connectome": FULL_CONNECTOME_COLOR,
    "learnt_recurrence": LEARNT_RECURRENCE_COLOR,
    "shuffle_inputs": SHUFFLE_WEIGHTS_COLOR,
    "configuration_model": CONFIGURATION_MODEL_COLOR,
}
#: (group, cell_type, alpha, hatch) per bar within a variant.
HELD_OUT_SERIES = (
    ("observed", "all", 0.45, None),
    ("unobserved", "all", 1.0, None),
)
#: The perturbation's non-targeted unobserved populations (targets scored separately).
PERTURBATION_SERIES = (
    ("unobserved", "excitatory", 0.45, None),
    ("unobserved", "inhibitory", 1.0, None),
)


def format_count(n):
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n:,}"


def grouped_bars(ax, summary, metric, series=None):
    """Bars per variant, two populations each (light, dark) with dotted ceilings.

    ``series`` is (group, cell_type, alpha) per bar: the held-out pairing by default,
    or the perturbation's non-targeted E and I populations.
    """
    series = series or HELD_OUT_SERIES
    rows = summary[summary["metric"] == metric]
    variants = [v for v in PLOTTED_VARIANTS if v in set(rows["variant"])]
    width = 0.38
    for g, (group, cell_type, alpha, hatch) in enumerate(series):
        for i, variant in enumerate(variants):
            sub = rows[
                (rows["variant"] == variant)
                & (rows["group"] == group)
                & (rows["cell_type"] == cell_type)
            ]
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


def schematic(fig, cell, params_by_variant, label="c"):
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
        "shuffle_inputs": shuffle_weights_within_neuron(
            toy, types, np.random.default_rng(1)
        ),
        "shuffle_weights": shuffle_weights_within_connectome(
            toy, types, np.random.default_rng(1)
        ),
        "configuration_model": configuration_model_rewire(
            toy, types, np.random.default_rng(1)
        ),
    }
    variants = [v for v in PLOTTED_VARIANTS if v in matrices]
    grid = cell.subgridspec(1, len(variants), wspace=0.1)
    for k, variant in enumerate(variants):
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
            panel_label(ax, label)


def main(data_dir, out_path, observed_fraction=None):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig02_summary.csv")
    # The grid holds two observation levels (see run_grid_search.py); plot one per figure.
    if "observed_fraction" not in summary:  # CSVs written before 2026-09-18
        summary["observed_fraction"] = np.nan
    if "cell_type" not in summary:  # CSVs written before the perturbation panel
        summary["cell_type"] = "all"
        summary["evaluation"] = "held_out"
    if observed_fraction is None:
        observed_fraction = summary["observed_fraction"].min()
    if np.isnan(observed_fraction):
        summary["observed_fraction"] = observed_fraction = 0.5
    summary = summary[np.isclose(summary["observed_fraction"], observed_fraction)]
    if summary.empty:
        raise SystemExit(f"no runs at observed_fraction {observed_fraction}")
    params_by_variant = summary.groupby("variant")["n_free_params"].first().to_dict()
    plotted = summary[summary["variant"].isin(PLOTTED_VARIANTS)]
    n_seeds = plotted.groupby("variant")["seed"].nunique().min()
    observed = observed_fraction

    has_perturbation = (summary["metric"] == "delta_activity_r2").any()
    rows = 4 if has_perturbation else 3
    ratios = [1.3, 0.9] + ([0.9] if has_perturbation else []) + [0.7]
    fig = plt.figure(
        figsize=(FIGURE_WIDTH, (22 if has_perturbation else 17) / 2.54),
        layout="constrained",
    )
    grid = fig.add_gridspec(rows, 1, height_ratios=ratios)

    ax = fig.add_subplot(grid[0])
    grouped_bars(ax, summary, "fluctuation_r2")
    ax.set_title(
        "Fluctuation R², held-out stimuli (light: observed, dark: unobserved;"
        " ··· ceiling)",
        fontsize=7,
    )
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[1])
    grouped_bars(ax, summary, "activity_r2")
    ax.set_title("Activity R²", fontsize=7)
    panel_label(ax, "b")

    # (c) perturbation: a separate row, so dropping it is one line.
    if has_perturbation:
        cells = grid[2].subgridspec(1, 2, wspace=0.05)
        for column, metric in enumerate(("delta_activity_r2", "delta_fluctuation_r2")):
            ax = fig.add_subplot(cells[0, column])
            grouped_bars(ax, summary, metric, PERTURBATION_SERIES)
            ax.set_title(METRIC_LABELS[metric], fontsize=7)
            ax.tick_params(axis="x", labelsize=5)
            if column == 0:
                # The intervention goes in the y label: the half-width axes cannot
                # hold a title describing it without overflowing the figure.
                ax.set_ylabel("Inhibiting 25% of\nunobserved I cells", fontsize=6.5)
                ax.set_title(
                    f"{METRIC_LABELS[metric]} (light: non-targeted E, dark: I)",
                    fontsize=6,
                )
                panel_label(ax, "c")
            else:
                ax.set_ylabel(None)

    schematic(fig, grid[rows - 1], params_by_variant, "d" if has_perturbation else "c")

    fig.suptitle(
        "The connectome is what's doing the work: 6 parameters with it beat 25M without\n"
        f"({observed:.0%} observed; mean ± SD over {n_seeds} seeds)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig02.svg")
    parser.add_argument(
        "--observed-fraction",
        type=float,
        help="which observation level to plot (default: the lowest present)",
    )
    args = parser.parse_args()
    main(args.data, args.out, args.observed_fraction)
