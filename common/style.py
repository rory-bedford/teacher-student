"""Archived paper-figure style (archive/figures/make_figures.py), resized for slides.

Shared by each figure's ``figures.py`` and by the PLACEHOLDER builds in
``placeholder_figures/``, which call the same panel functions with a watermark.

One file per panel. Sizes are the native size to insert into PowerPoint at 100%:
the archive's poster fonts (18 / 19.2 / 21.6 pt on 12-inch figures) are scaled down
to 12 / 13 / 14 pt on ~6.5-inch panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from connectome_snns.visualization import (
    FIGURE_BLUE,
    FIGURE_CORAL,
    RASTER_BAND_COLOR,
    use_project_style,
)
from matplotlib.lines import Line2D

EXCITATORY = FIGURE_CORAL
INHIBITORY = FIGURE_BLUE
TEACHER = FIGURE_CORAL
STUDENT = FIGURE_BLUE
LEGEND_GREY = "#404040"

TICK_SIZE = 12
LABEL_SIZE = 13
TITLE_SIZE = 14

SINGLE = (6.5, 4.5)
PAIR = (11.0, 5.2)
TRIPLE = (13.0, 4.8)
WIDE = (11.0, 3.6)

METRIC_STYLES = {  # archived sweep curves: Activity o-, Fluctuation s--
    "activity_r2": ("o", "-", "Activity"),
    "fluctuation_r2": ("s", "--", "Fluctuation"),
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
            "legend.fontsize": TICK_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "svg.fonttype": "path",
        }
    )


def save(fig, out_dir, figure, letter, slug, suffix="", decorate=None):
    """``<out_dir>/<figure>-<letter>-<slug><suffix>.svg``, e.g. fig01-a-raster.svg.

    ``decorate`` is called with the figure just before saving (the placeholder builds
    pass their watermark), so one panel function serves both the real and fake data.
    """
    if decorate is not None:
        decorate(fig)
    path = Path(out_dir) / f"{figure}-{letter}-{slug}{suffix}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def nice_max(values, step=10):
    return float(step * np.ceil(np.nanpercentile(values, 99.5) / step))


def rate_scatter(ax, rates, title, max_rate):
    """Archived teacher-vs-student firing-rate scatter, coloured by cell type."""
    for cell_type, color, name in (
        ("inhibitory", INHIBITORY, "Inhibitory"),
        ("excitatory", EXCITATORY, "Excitatory"),
    ):
        subset = rates[rates["cell_type"] == cell_type]
        ax.scatter(
            subset["teacher_rate_hz"],
            subset["student_rate_hz"],
            s=4,
            alpha=0.5,
            color=color,
            label=name,
            rasterized=True,
        )
    ax.plot([0, max_rate], [0, max_rate], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    ax.set_aspect("equal")
    ax.set_xlabel("Teacher Firing Rate (Hz)")
    ax.set_ylabel("Student Firing Rate (Hz)")
    ax.set_title(title)
    legend = ax.legend(loc="upper left", markerscale=4, scatterpoints=1)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)


def spike_raster(ax, spikes, neurons, duration_s, labels):
    """Archived raster: student (blue) above teacher (coral) per neuron, grey bands.

    Args:
        spikes: DataFrame with neuron_id, source, time_s.
        neurons: neuron ids, top to bottom.
        labels: one y label per neuron (same order).
    """
    gap = 0.3
    ticks = []
    for i, (neuron, label) in enumerate(zip(reversed(neurons), reversed(labels))):
        base = i * (2 + gap)
        if i % 2 == 0:
            ax.axhspan(
                base - 0.5 - gap / 2,
                base + 1.5 + gap / 2,
                color=RASTER_BAND_COLOR,
                zorder=0,
            )
        for offset, source, color in ((0, "teacher", TEACHER), (1, "student", STUDENT)):
            times = spikes[
                (spikes["neuron_id"] == neuron) & (spikes["source"] == source)
            ]["time_s"]
            ax.eventplot(
                times.values,
                lineoffsets=base + offset,
                linelengths=0.6,
                linewidths=1.6,
                colors=color,
            )
        ticks.append((base + 0.5, label))
    n = len(neurons)
    ax.set_ylim(-0.5 - gap / 2, (n - 1) * (2 + gap) + 1.5 + gap / 2)
    ax.set_xlim(0, duration_s)
    ax.set_xticks(np.arange(0, duration_s + 1e-9, 1.0))
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([label for _, label in ticks])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time (s)")
    ax.legend(
        handles=[
            Line2D([], [], color=STUDENT, linewidth=4, label="Student"),
            Line2D([], [], color=TEACHER, linewidth=4, label="Teacher"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )


def sweep_series(ax, rows, x_column, metric, color, marker=None, linestyle=None):
    """Seed mean ± SD of one series in the archived marker/line style."""
    default_marker, default_linestyle, _ = METRIC_STYLES[metric]
    stats = rows.groupby(x_column)["value"].agg(["mean", "std"]).reset_index()
    ax.errorbar(
        stats[x_column],
        stats["mean"],
        yerr=stats["std"].fillna(0.0),
        color=color,
        marker=marker or default_marker,
        linestyle=linestyle or default_linestyle,
        linewidth=1.5,
        markersize=5,
        capsize=2.5,
    )


def ceiling(ax, rows, x_column, color):
    stats = rows.groupby(x_column)["ceiling_value"].mean().reset_index()
    ax.plot(
        stats[x_column],
        stats["ceiling_value"],
        linestyle=":",
        color=color,
        linewidth=1.2,
        alpha=0.7,
    )


def sweep_legend(ax, series, metrics=True, ceiling_line=True, extra=()):
    """Legend outside the axes: series colours plus the archived grey metric handles."""
    handles = [
        Line2D([], [], color=color, linewidth=6, label=label)
        for label, color in series.items()
    ]
    if metrics:
        handles += [
            Line2D(
                [],
                [],
                color=LEGEND_GREY,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6,
                label=label,
            )
            for marker, linestyle, label in METRIC_STYLES.values()
        ]
    if ceiling_line:
        handles.append(
            Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Ceiling")
        )
    handles += list(extra)
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False
    )
