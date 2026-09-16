"""Regenerate the figure SVGs from figures/data/*.csv.

Run from the repo root:
    uv run python figures/make_figures.py                  # all figures
    uv run python figures/make_figures.py panel-a raster   # names matching
    uv run python figures/make_figures.py --out-dir /tmp/x # write elsewhere
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.analysis import r_squared
from connectome_snns.visualization import (
    FIGURE_BLUE,
    FIGURE_COLORS,
    FIGURE_CORAL,
    RASTER_BAND_COLOR,
    use_project_style,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

FIGURES_DIR = Path(__file__).resolve().parent
DATA_DIR = FIGURES_DIR / "data"

EXCITATORY = FIGURE_CORAL
INHIBITORY = FIGURE_BLUE
TEACHER = FIGURE_CORAL
STUDENT = FIGURE_BLUE
LEGEND_GREY = "#404040"
MAX_RATE = 50

TICK_SIZE = 18
LABEL_SIZE = 19.2
TITLE_SIZE = 21.6

PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]
TITLES = {
    "title-teacher-student-training-paradigm": "Teacher-Student Training Paradigm",
    "title-connectome-outperforms-controls": "Connectome Outperforms Controls",
}


def apply_style():
    use_project_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": LABEL_SIZE,
            "axes.titlesize": LABEL_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LABEL_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "svg.fonttype": "path",
        }
    )


def fr_scatter(name, suptitle):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, observed, label in [
        (axes[0], True, "Observed"),
        (axes[1], False, "Unobserved"),
    ]:
        sub = df[df["observed"] == observed]
        r2 = r_squared(sub["teacher_rate_hz"].values, sub["student_rate_hz"].values)
        for cell_type, color, cell_name in [
            ("inhibitory", INHIBITORY, "Inhibitory"),
            ("excitatory", EXCITATORY, "Excitatory"),
        ]:
            ct = sub[sub["cell_type"] == cell_type]
            ax.scatter(
                ct["teacher_rate_hz"],
                ct["student_rate_hz"],
                s=6,
                alpha=0.5,
                color=color,
                label=cell_name,
                rasterized=True,
            )
        ax.plot([0, MAX_RATE], [0, MAX_RATE], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(0, MAX_RATE)
        ax.set_ylim(0, MAX_RATE)
        ax.set_aspect("equal")
        ax.set_xlabel("Teacher Firing Rate (Hz)")
        ax.set_ylabel("Student Firing Rate (Hz)")
        ax.set_title(f"{label} (R² = {r2:.3f})")
        ax.legend(loc="upper left", markerscale=5, scatterpoints=1)
        for handle in ax.get_legend().legend_handles:
            handle.set_alpha(1.0)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def ff_learnt_obs45_scatter(name):
    return fr_scatter(name, "Degeneracy Under Partial Reconstruction")


def ff_known_obs10_scatter(name):
    return fr_scatter(name, "Student vs Teacher Activity")


def controls_bar(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    measures = {"Activity R²": "activity_r2", "Fluctuation R²": "fluctuation_r2"}
    ymin = min(0.0, df[list(measures.values())].values.min()) - 0.05

    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(measures))
    width = 0.8 / len(df)
    for i, (row, color) in enumerate(zip(df.itertuples(), FIGURE_COLORS)):
        ax.bar(
            x + (i - (len(df) - 1) / 2) * width,
            [getattr(row, col) for col in measures.values()],
            width,
            label=row.condition,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(measures.keys())
    ax.set_ylabel("R²")
    ax.set_ylim(ymin, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True)
    fig.tight_layout()
    return fig


def ff_known_obs10_raster(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    neuron_ids = list(dict.fromkeys(df["neuron_id"]))  # preserve top-to-bottom order
    gap = 0.3

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, nid in enumerate(reversed(neuron_ids)):
        base = i * (2 + gap)
        if i % 2 == 0:
            ax.axhspan(
                base - 0.5 - gap / 2,
                base + 1.5 + gap / 2,
                color=RASTER_BAND_COLOR,
                zorder=0,
            )
        for offset, source, color in [(0, "teacher", TEACHER), (1, "student", STUDENT)]:
            times = df[(df["neuron_id"] == nid) & (df["source"] == source)][
                "spike_time_s"
            ]
            ax.eventplot(
                times.values,
                lineoffsets=base + offset,
                linelengths=0.6,
                linewidths=2.6,
                colors=color,
            )
    n = len(neuron_ids)
    ax.set_ylim(-0.5 - gap / 2, (n - 1) * (2 + gap) + 1.5 + gap / 2)
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time (s)")
    ax.set_title("Student vs Teacher Raster")
    ax.legend(
        handles=[
            Line2D([], [], color=STUDENT, linewidth=5, label="Student"),
            Line2D([], [], color=TEACHER, linewidth=5, label="Teacher"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    return fig


def sweep_curve(name, xcol, xlabel, color, title):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[xcol], df["activity_r2"], "o-", color=color, linewidth=1.5, markersize=4)
    ax.plot(
        df[xcol], df["fluctuation_r2"], "s--", color=color, linewidth=1.5, markersize=4
    )
    ax.set_xlim(df[xcol].min() - 0.025, df[xcol].max() + 0.025)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Firing Rate R²")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=LEGEND_GREY,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=10,
                label=label,
            )
            for marker, linestyle, label in [
                ("o", "-", "Activity"),
                ("s", "--", "Fluctuation"),
            ]
        ],
        loc="lower left",
        frameon=False,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    fig.tight_layout()
    return fig


def sweep_recur_curve(name):
    return sweep_curve(
        name,
        "removed_neuron_fraction",
        "Removed Neuron Fraction",
        FIGURE_BLUE,
        "Robustness to Missing Neurons",
    )


def sweep_wn_curve(name):
    return sweep_curve(
        name,
        "weight_noise_fraction",
        "Weight Noise Fraction",
        FIGURE_CORAL,
        "Robustness to Weight Noise",
    )


def text_figure(text):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, text, fontsize=TITLE_SIZE)
    return fig


FIGURES = {
    "ff-learnt__obs-45__recur-81__wn-30__scatter": ff_learnt_obs45_scatter,
    "controls__ff-learnt__obs-90__recur-81__wn-mixed__bar": controls_bar,
    "ff-known__obs-10__recur-100__wn-0__scatter": ff_known_obs10_scatter,
    "ff-known__obs-10__recur-100__wn-0__raster": ff_known_obs10_raster,
    "sweep-recur__ff-known__obs-100__wn-0__curve": sweep_recur_curve,
    "sweep-wn__ff-known__obs-100__recur-100__curve": sweep_wn_curve,
    **{f"panel-{p}": (lambda _, p=p: text_figure(f"{p})")) for p in PANEL_LABELS},
    **{name: (lambda _, t=text: text_figure(t)) for name, text in TITLES.items()},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="substrings of figure names to make")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        if args.names and not any(n in name for n in args.names):
            continue
        fig = make(name)
        fig.savefig(args.out_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {args.out_dir / name}.svg")


if __name__ == "__main__":
    main()
