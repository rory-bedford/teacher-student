"""Figure 1 — build fig01.svg from the CSVs written by analysis.py.

uv run python fig01-full-reconstruction/figures.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import (
    EXCITATORY_COLOR,
    FIGURE_WIDTH,
    GROUP_LABELS,
    INHIBITORY_COLOR,
    delta_mean_inset,
    delta_rate_scatter,
    panel_label,
    r2_title,
    rate_scatter,
    spike_raster,
    use_talk_style,
)

HERE = Path(__file__).resolve().parent
RASTER_SECONDS = 3.0
PERTURBATION_CSV = "fig01_perturbation.csv"


def perturbation_panel(fig, cell, summary, deltas):
    """(d) the intervention's effect: teacher vs student Δrate, one point per neuron."""
    ax = fig.add_subplot(cell)
    delta_rate_scatter(ax, deltas)
    delta_mean_inset(ax, deltas)
    ax.legend(loc="upper left", markerscale=3, frameon=False, fontsize=6)
    rows = summary[summary["evaluation"] == "perturbation"]
    scores = []
    for metric, short in (
        ("delta_activity_r2", "ΔAct"),
        ("delta_fluctuation_r2", "ΔFlu"),
    ):
        values = []
        for cell_type, label in (("excitatory", "E"), ("inhibitory", "I")):
            m = rows[
                (rows["metric"] == metric)
                & (rows["group"] == "unobserved")
                & (rows["cell_type"] == cell_type)
            ]
            if not m.empty:
                values.append(
                    f"{label} {m['value'].mean():.2f} [{m['ceiling_value'].mean():.2f}]"
                )
        if values:
            scores.append(f"{short} R²: " + ", ".join(values))
    ax.set_title(
        "Inhibiting 25% of unobserved I cells\n" + "\n".join(scores), fontsize=6
    )
    panel_label(ax, "d")


def main(data_dir, out_path):
    use_talk_style()
    summary = pd.read_csv(data_dir / "fig01_summary.csv")
    rates = pd.read_csv(data_dir / "fig01_rates.csv")
    spikes = pd.read_csv(data_dir / "fig01_spikes.csv")
    seed = int(spikes["seed"].iloc[0])
    rates_seed = rates[rates["seed"] == seed]
    if "evaluation" in summary:
        summary["evaluation"] = summary["evaluation"].fillna("held_out")
    else:  # CSVs written before the perturbation panel
        summary["evaluation"] = "held_out"
    held_out = summary[summary["evaluation"] == "held_out"]
    n_seeds = held_out["seed"].nunique()
    # The perturbation CSV is absent when the teacher has no calibrated current.
    perturbation = data_dir / PERTURBATION_CSV
    deltas = pd.read_csv(perturbation) if perturbation.exists() else None
    if deltas is not None:
        deltas = deltas[deltas["seed"] == seed]
        if deltas.empty:
            deltas = None

    rows = 3 if deltas is None else 4
    height = 16 if deltas is None else 23
    ratios = [1.0, 1.35, 0.9] + ([] if deltas is None else [1.35])
    fig = plt.figure(figsize=(FIGURE_WIDTH, height / 2.54), layout="constrained")
    grid = fig.add_gridspec(rows, 2, height_ratios=ratios)

    # (a) raster: observed and unobserved neurons on a held-out stimulus.
    ax = fig.add_subplot(grid[0, :])
    neurons = (
        spikes[["neuron_id", "observed"]]
        .drop_duplicates()
        .sort_values(["observed", "neuron_id"], ascending=[False, True])
    )
    spike_raster(
        ax,
        spikes[spikes["time_s"] <= RASTER_SECONDS],
        list(neurons.itertuples(index=False, name=None)),
        RASTER_SECONDS,
    )
    ax.set_title("Held-out stimulus: teacher (grey) vs student (colour)")
    panel_label(ax, "a")

    # (b) rate scatters, observed and unobserved.
    max_rate = (
        float(
            np.nanpercentile(rates_seed[["teacher_rate_hz", "student_rate_hz"]], 99.5)
        )
        * 1.05
    )
    for column, group in enumerate(("observed", "unobserved")):
        ax = fig.add_subplot(grid[1, column])
        subset = rates_seed[rates_seed["observed"] == int(group == "observed")]
        title = r2_title(GROUP_LABELS[group], held_out, group)
        rate_scatter(ax, subset, title, max_rate=max_rate)
        if column == 0:
            ax.legend(loc="upper left", markerscale=3, frameon=False)
            panel_label(ax, "b")

    # (c) per-neuron fluctuation R², E vs I, with the group ceiling.
    for column, group in enumerate(("observed", "unobserved")):
        ax = fig.add_subplot(grid[2, column])
        subset = rates_seed[rates_seed["observed"] == int(group == "observed")]
        bins = np.linspace(-1.0, 1.0, 41)
        for cell_type, color in (
            ("excitatory", EXCITATORY_COLOR),
            ("inhibitory", INHIBITORY_COLOR),
        ):
            values = subset[subset["cell_type"] == cell_type]["fluctuation_r2"].dropna()
            ax.hist(
                np.clip(values, bins[0], bins[-1]),
                bins=bins,
                color=color,
                alpha=0.6,
                label=cell_type[0].upper(),
            )
        rows = held_out[
            (held_out["group"] == group) & (held_out["metric"] == "fluctuation_r2")
        ]
        ax.axvline(
            rows["ceiling_value"].mean(),
            color="k",
            linestyle=":",
            linewidth=0.8,
            label="ceiling",
        )
        ax.set_xlabel("Per-neuron Fluctuation R²")
        ax.set_ylabel("Neurons")
        ax.set_title(GROUP_LABELS[group])
        if column == 0:
            ax.legend(frameon=False)
            panel_label(ax, "c")

    # (d) perturbation: kept a separate panel, so dropping it is one line.
    if deltas is not None:
        perturbation_panel(fig, grid[3, :], summary, deltas)

    fig.suptitle(
        f"Full reconstruction recovers the teacher, including unobserved neurons\n"
        f"(10% observed; held-out stimuli; R² mean over {n_seeds} seeds, scatter seed {seed})"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "fig01.svg")
    args = parser.parse_args()
    main(args.data, args.out)
