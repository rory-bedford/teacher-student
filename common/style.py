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
    FLOOR_COLOR,
    FULL_CONNECTOME_COLOR,
    LEARNT_RECURRENCE_COLOR,
    RASTER_BAND_COLOR,
    SLIDE_BLUE,
    SLIDE_INK,
    SLIDE_RED,
    use_project_style,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# One role, one colour, across every slide (2026-09-21, COLORSCHEME.txt). Cell type takes
# the deck's red/blue accents (the neuroscience convention). Teacher vs student is one hue
# in two shades -- dark navy for the ground truth, the deck's accent blue for the model --
# which reads as the same quantity from two sources and stays legible at line weight. The
# dark slate and the yellow accent were tried first and were, respectively, too flat for
# the teacher and too faint to carry a series.
# COLORSCHEME.txt: the deck's three accents carry the roles that appear on every slide,
# and a small set of fixed accents carries the conditions. A colour means one thing.
#
#   red / blue      excitatory / inhibitory, inside the rate scatters
#   steel blue      the teacher -- and therefore the full connectome, its first bar
#   raspberry       the student
#   navy / amber    observed / unobserved populations, in the sweeps
EXCITATORY = SLIDE_RED
INHIBITORY = SLIDE_BLUE
# Two colours from the controls bar chart, complementary and highly differentiable, carry
# every "given vs inferred" pairing in the talk:
#
#   steel blue   the teacher, the full connectome, the observed population -- what is given
#   orange       the student, the learnt model, the unobserved population -- what is inferred
#
# The teacher's colour is deliberately the first bar of Figure 2: that bar IS the
# teacher's connectivity.
TRUTH = FULL_CONNECTOME_COLOR
MODEL = LEARNT_RECURRENCE_COLOR
OBSERVED = TRUTH
UNOBSERVED = MODEL
INK = SLIDE_INK
#: Reference lines: noise ceilings, dimensionality markers. Furniture, not data.
REFERENCE_GREY = FLOOR_COLOR
#: Cell type is a colour again (red / blue), so markers stay uniform.
MARKERS = {"excitatory": "o", "inhibitory": "o"}

TEACHER = TRUTH
STUDENT = MODEL
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
    # The perturbation's deltas take the same markers as the metric they are a delta of.
    "delta_activity_r2": ("o", "-", "ΔActivity"),
    "delta_fluctuation_r2": ("s", "--", "ΔFluctuation"),
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


def clear_panels(out_dir, figure, suffix=""):
    """Delete this figure's existing panel SVGs before a rebuild.

    Panels are named per letter, so re-lettering or dropping a panel otherwise leaves an
    orphan SVG behind that looks current in the folder and can be pasted into a slide by
    mistake. Every figure's ``main`` calls this first.
    """
    for path in sorted(Path(out_dir).glob(f"{figure}-*{suffix}.svg")):
        path.unlink()


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


#: Rate scatters are linear over 0 to this many Hz (2026-09-18). The teacher's rates run
#: to ~270 Hz with a median of 0.4, so the axis is cut rather than scaled: the tail is a
#: handful of cells firing every 2 ms (nothing enforces a refractory period in the model),
#: and they are not what the figure is about. Cells beyond the limit are counted in the
#: axis label, not plotted.
RATE_MAX_HZ = 40.0
#: Tick spacing on both axes of every scatter, rates and deltas alike.
RATE_TICK_HZ = 10.0
#: Marker area for the rate scatters; the delta scatter matches it (see common.plotting).
RATE_MARKER_SIZE = 10
#: Every legend is framed and scales its markers by the same factor, so a legend looks the
#: same whichever panel it sits in (2026-09-21).
LEGEND_MARKERSCALE = 3


def nice_max(values, step=10):
    """A round upper limit covering every value (rates are plotted on a symlog axis)."""
    return float(step * np.ceil(np.nanmax(values) / step))


def rate_scatter(ax, rates, title, max_rate=RATE_MAX_HZ, clip=RATE_MAX_HZ):
    """Archived teacher-vs-student firing-rate scatter, coloured by cell type.

    Linear axes over 0 to ``clip`` Hz. Cells beyond it are simply left off -- not counted
    on the axis, not piled on its edge (2026-09-21: the count looked messy, and the clip
    belongs in the caption) -- rather than stretching the axis over a tail nothing in the
    talk depends on. Pass ``clip=None`` to plot every cell.
    """
    limit = max_rate if clip is None else min(clip, max_rate)
    for cell_type, color, name in (
        ("inhibitory", INHIBITORY, "Inhibitory"),
        ("excitatory", EXCITATORY, "Excitatory"),
    ):
        subset = rates[rates["cell_type"] == cell_type]
        teacher = subset["teacher_rate_hz"]
        student = subset["student_rate_hz"]
        if clip is not None:
            outside = (teacher > limit) | (student > limit)
            teacher, student = teacher[~outside], student[~outside]
        ax.scatter(
            teacher,
            student,
            s=RATE_MARKER_SIZE,
            alpha=0.45,
            color=color,
            marker=MARKERS[cell_type],
            label=name,
            rasterized=True,
        )
    max_rate = limit
    ax.plot([0, max_rate], [0, max_rate], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    # Ticks every 10 Hz on both axes, in every scatter (2026-09-21).
    ax.xaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.yaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.set_aspect("equal")
    ax.set_xlabel("Teacher Firing Rate (Hz)")
    ax.set_ylabel("Student Firing Rate (Hz)")
    ax.set_title(title)
    legend = ax.legend(
        loc="upper left", markerscale=LEGEND_MARKERSCALE, scatterpoints=1, frameon=True
    )
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


def sweep_series(
    ax,
    rows,
    x_column,
    metric,
    color,
    marker=None,
    linestyle=None,
    seeds=False,
    errorbars=True,
    x_group=None,
):
    """Seed mean ± SD of one series in the archived marker/line style.

    ``seeds=True`` also scatters the individual seeds behind the line: where a sweep is
    bimodal (Figure 3's low end, where a run either trains or collapses) the mean sits
    between two clusters and describes neither. With the seeds shown, ``errorbars=False``
    keeps the panel readable -- the points already carry the spread.

    ``x_group`` is the column that identifies a condition when ``x_column`` is itself
    measured per run (Figure 4 plots against the input volume each run actually lost, so
    grouping by x alone would put every seed in its own group and thread the line through
    individual runs instead of the means).
    """
    default_marker, default_linestyle, _ = METRIC_STYLES[metric]
    if x_group is None:
        stats = rows.groupby(x_column)["value"].agg(["mean", "std"]).reset_index()
    else:
        stats = (
            rows.groupby(x_group)
            .agg(
                **{
                    x_column: (x_column, "mean"),
                    "mean": ("value", "mean"),
                    "std": ("value", "std"),
                }
            )
            .reset_index()
            .sort_values(x_column)
        )
    if seeds:
        # With x measured per run (``x_group``), the seeds' own x differ by a fraction of
        # a percent -- invisible information that reads as jitter, so they are drawn at
        # the condition's mean x.
        seed_x = rows[x_column]
        if x_group is not None:
            seed_x = rows[x_group].map(stats.set_index(x_group)[x_column])
        ax.scatter(
            seed_x,
            rows["value"],
            s=RATE_MARKER_SIZE * 2,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
    ax.errorbar(
        stats[x_column],
        stats["mean"],
        yerr=stats["std"].fillna(0.0) if errorbars else None,
        color=color,
        marker=marker or default_marker,
        linestyle=linestyle or default_linestyle,
        linewidth=1.5,
        markersize=5,
        capsize=2.5,
    )


def ceiling(ax, rows, x_column, color, x_group=None):
    """Dotted ceiling of one series; ``x_group`` as in :func:`sweep_series`."""
    if x_group is None:
        stats = rows.groupby(x_column)["ceiling_value"].mean().reset_index()
    else:
        stats = (
            rows.groupby(x_group)
            .agg(
                **{
                    x_column: (x_column, "mean"),
                    "ceiling_value": ("ceiling_value", "mean"),
                }
            )
            .reset_index()
            .sort_values(x_column)
        )
    ax.plot(
        stats[x_column],
        stats["ceiling_value"],
        linestyle=":",
        color=color,
        linewidth=1.2,
        alpha=0.7,
    )


def sweep_legend(
    ax,
    series,
    metrics=True,
    ceiling_line=True,
    extra=(),
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    fontsize=None,
):
    """Series colours plus the archived grey metric handles.

    Outside the axes on the right by default; pass ``bbox_to_anchor=None`` with a corner
    ``loc`` to put it inside, which keeps a wide panel from being half legend.
    """
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
            Line2D([], [], color=LEGEND_GREY, linestyle=":", label="Noise Ceiling")
        )
    handles += list(extra)
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.9,
        fontsize=fontsize,
        handlelength=1.6,
        labelspacing=0.35,
        borderpad=0.5,
    )
