"""Backup panels — kept for questions, not for the talk's main line.

    uv run python backup-figures/figures.py

    backup-a-scaling-factors   the six tied scaling factors, fully observed vs Figure 1's
                               10% observed: recovered exactly when everything is seen,
                               drifting when most of the network is simulated.

Reads the CSVs the figure folders' ``analysis.py`` already write (no evaluation here), in
the same archived style as the talk figures (``common/style.py``), one SVG per panel.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.style import (
    EXCITATORY,
    INHIBITORY,
    LEGEND_GREY,
    SINGLE,
    apply_style,
    save,
)

HERE = Path(__file__).resolve().parent
FIGURE = "backup"
SCALING_FACTORS_CSV = (
    HERE.parent / "fig01-full-reconstruction" / "fig01_scaling_factors.csv"
)
#: (value of the ``observed`` column, filled marker) per condition; the label carries the
#: run's own observed fraction, read from the CSV, so it cannot drift from the runs.
CONDITIONS = (("full", True), ("partial", False))


def condition_label(factors, condition):
    """ "Fully observed" / "10% observed", from the runs' own observed fraction."""
    rows = factors[factors["observed"] == condition]
    if rows.empty:
        return condition
    fractions = sorted(rows["observed_fraction"].unique())
    if fractions == [1.0]:
        return "Fully observed"
    return ", ".join(f"{fraction:.0%} observed" for fraction in fractions)


def scaling_factors(factors):
    """Learnt / true scaling factor per projection, one condition beside the other.

    The true model is scaling factor 1.0 for every projection (the student's physiology
    is the teacher's), so truth is a line rather than a second axis.
    """
    order = sorted(factors["scaling_factor"].unique())
    fig, ax = plt.subplots(figsize=(SINGLE[0] * 1.15, SINGLE[1]))
    offsets = np.linspace(-0.12, 0.12, len(CONDITIONS))
    for (condition, filled), offset in zip(CONDITIONS, offsets):
        rows = factors[factors["observed"] == condition]
        for x, name in enumerate(order):
            values = rows[rows["scaling_factor"] == name]
            if values.empty:
                continue
            ratio = values["value"] / values["target"]
            color = {"excitatory": EXCITATORY, "inhibitory": INHIBITORY}.get(
                name.split("_to_")[0], LEGEND_GREY
            )
            ax.scatter(
                np.full(len(ratio), x + offset),
                ratio,
                s=45,
                zorder=3,
                color=color if filled else "none",
                edgecolors=color,
                linewidths=1.4,
            )
            if len(ratio) > 1:
                ax.errorbar(
                    x + offset,
                    ratio.mean(),
                    yerr=ratio.std(),
                    color="k",
                    capsize=3,
                    linewidth=1,
                )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # The log scale's minor labels (3 x 10^0, 6 x 10^-1) collide with the ticks above.
    ax.get_yaxis().set_minor_formatter(NullFormatter())
    ax.set_ylim(0.2, 5.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [name.replace("_to_", "→").replace("_", " ") for name in order],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Learnt / True Scaling Factor")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                color=LEGEND_GREY,
                markerfacecolor=LEGEND_GREY if filled else "none",
                label=condition_label(factors, condition),
            )
            for condition, filled in CONDITIONS
        ],
        loc="upper left",
        frameon=True,
    )
    ax.set_title("Recovered When Fully Observed, Compensating When Not")
    fig.tight_layout()
    return fig


def main(out_dir, decorate=None, suffix=""):
    apply_style()
    if not SCALING_FACTORS_CSV.exists():
        raise SystemExit(
            f"{SCALING_FACTORS_CSV} missing: run fig01-full-reconstruction/analysis.py"
        )
    factors = pd.read_csv(SCALING_FACTORS_CSV)
    save(
        scaling_factors(factors),
        out_dir,
        FIGURE,
        "a",
        "scaling-factors",
        suffix,
        decorate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.out_dir)
