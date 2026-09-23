"""Plot style and helpers shared by every figure's ``figures.py``.

Figures are written at final size: 12 cm wide, fonts legible on a projected slide.
All colours come from ``connectome_snns.visualization``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectome_snns.visualization import use_project_style
from matplotlib.ticker import MultipleLocator

from common.style import (
    EXCITATORY,
    INHIBITORY,
    OBSERVED,
    RATE_MARKER_SIZE,
    RATE_TICK_HZ,
    REFERENCE_GREY,
    UNOBSERVED,
)

CM = 1 / 2.54
FIGURE_WIDTH = 12 * CM

EXCITATORY_COLOR = EXCITATORY
INHIBITORY_COLOR = INHIBITORY
#: Targeted cells are a triangle marker, so their colour stays neutral.
TARGETED_COLOR = REFERENCE_GREY
GROUP_COLORS = {"observed": OBSERVED, "unobserved": UNOBSERVED}
GROUP_LABELS = {"observed": "Observed", "unobserved": "Unobserved"}
METRIC_LABELS = {
    "fluctuation_r2": "Fluctuation R²",
    "activity_r2": "Activity R²",
    "delta_activity_r2": "ΔActivity R²",
    "delta_fluctuation_r2": "ΔFluctuation R²",
}
#: The perturbation's scored populations: (group, cell type, colour, label).
PERTURBATION_SERIES = (
    ("unobserved", "excitatory", EXCITATORY_COLOR, "Unobserved E"),
    ("unobserved", "inhibitory", INHIBITORY_COLOR, "Unobserved I"),
)
#: Shared wording for the perturbation panels, so the four figures agree (2026-09-21).
#: The series are the unobserved cells that were NOT targeted, E and I pooled; the value
#: is the change the intervention caused, perturbed minus unperturbed.
#: The two titles every performance panel carries (2026-09-23). What the perturbation is
#: -- an E_L shift on 25% of unobserved inhibitory cells, scored as on minus off -- belongs
#: on the slide and in each figure's README, not in the panel title.
HELD_OUT_TITLE = "Held-Out Test Set Accuracy"
PERTURBATION_TITLE = "Perturbation Accuracy"
PERTURBATION_LABEL = "Unobserved, Not Targeted"


def use_talk_style():
    use_project_style()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "svg.fonttype": "path",
        }
    )


def panel_label(ax, label):
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def rate_scatter(ax, rates, title, max_rate=None):
    """Teacher vs student firing rate, one point per neuron, coloured by E/I."""
    if max_rate is None:
        max_rate = (
            float(np.nanpercentile(rates[["teacher_rate_hz", "student_rate_hz"]], 99.5))
            * 1.05
        )
    for cell_type, color, label in (
        ("inhibitory", INHIBITORY_COLOR, "I"),
        ("excitatory", EXCITATORY_COLOR, "E"),
    ):
        subset = rates[rates["cell_type"] == cell_type]
        ax.scatter(
            subset["teacher_rate_hz"],
            subset["student_rate_hz"],
            s=2,
            alpha=0.5,
            color=color,
            label=label,
            linewidths=0,
            rasterized=True,
        )
    ax.plot([0, max_rate], [0, max_rate], "k--", linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    ax.set_aspect("equal")
    ax.set_xlabel("Teacher rate (Hz)")
    ax.set_ylabel("Student rate (Hz)")
    ax.set_title(title, fontsize=6)


def r2_title(label, summary, group, seed=None):
    """``label`` with Fluctuation/Activity R² and ceiling (one seed, or the mean)."""
    rows = summary[summary["group"] == group]
    if seed is not None:
        rows = rows[rows["seed"] == seed]
    lines = [label]
    for metric, short in (("fluctuation_r2", "Flu"), ("activity_r2", "Act")):
        m = rows[rows["metric"] == metric]
        lines.append(
            f"{short} R² {m['value'].mean():.2f} [noise ceiling {m['ceiling_value'].mean():.2f}]"
        )
    return "\n".join(lines)


def seed_errorbar(
    ax,
    summary,
    x_column,
    group,
    metric,
    color,
    label,
    marker="o",
    linestyle="-",
    cell_type=None,
):
    """Mean ± SD over seeds against ``x_column``, individual seeds as faint points.

    ``cell_type`` selects one population of the perturbation rows (whose ``group`` is
    shared by an E and an I row); held-out rows carry ``cell_type = "all"``.
    """
    rows = summary[(summary["group"] == group) & (summary["metric"] == metric)]
    if cell_type is not None:
        rows = rows[rows["cell_type"] == cell_type]
    stats = rows.groupby(x_column)["value"].agg(["mean", "std"]).reset_index()
    ax.scatter(
        rows[x_column], rows["value"], s=4, color=color, alpha=0.35, linewidths=0
    )
    ax.errorbar(
        stats[x_column],
        stats["mean"],
        yerr=stats["std"].fillna(0.0),
        color=color,
        marker=marker,
        markersize=3,
        linestyle=linestyle,
        capsize=1.5,
        label=label,
    )
    return rows


def ceiling_line(
    ax, summary, x_column, metric, group, color, label=None, cell_type=None
):
    """Dotted ceiling (perfectly specified student) for one group, averaged over seeds."""
    rows = summary[(summary["metric"] == metric) & (summary["group"] == group)]
    if cell_type is not None:
        rows = rows[rows["cell_type"] == cell_type]
    stats = rows.groupby(x_column)["ceiling_value"].mean().reset_index()
    ax.plot(
        stats[x_column],
        stats["ceiling_value"],
        linestyle=":",
        color=color,
        linewidth=1.0,
        label=label,
    )


def plotted_performance_values(summary, group, pool_by, cell_type="all"):
    """Every value the performance panels of a figure actually draw.

    Held-out rows are plotted as they come; perturbation rows are plotted POOLED over cell
    types, so the per-cell-type rows must never set the axis (2026-09-23: they were, and
    the floor dropped to -0.6 for points no panel contains).
    """
    held = summary[
        (summary.get("evaluation", "held_out") == "held_out")
        & (summary["metric"] == "fluctuation_r2")
        & (summary["group"] == group)
    ]
    if "cell_type" in held:
        held = held[held["cell_type"] == cell_type]
    values = list(held["value"])
    if "evaluation" in summary:
        delta = summary[
            (summary["evaluation"] == "perturbation")
            & (summary["metric"] == "delta_fluctuation_r2")
            & (summary["group"] == group)
        ]
        if not delta.empty:
            values += list(pool_populations(delta, pool_by)["value"])
    return values


def pool_populations(rows, by):
    """Pool the perturbation's scored populations into one number per run.

    ``by`` identifies a run (the figure's condition column plus "seed"). Values are
    weighted by cell count, so the result is what the populations' R² would be as one
    group of that size. It is an approximation -- a true pooled R² needs the smoothed
    traces, which the caches do not keep -- and it is close because the populations
    degrade together: E and I differ by less than 0.02 R² wherever the fit works at all,
    and diverge only where both are already near zero (see each figure's README).
    """
    return (
        rows.groupby(list(by))
        .apply(
            lambda group: pd.Series(
                {
                    "value": np.average(group["value"], weights=group["n_cells"]),
                    "ceiling_value": np.average(
                        group["ceiling_value"], weights=group["n_cells"]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def perturbation_sweep(
    ax, summary, x_column, metric, series=PERTURBATION_SERIES, group=None
):
    """Perturbation panel for a sweep figure: Δ R² of each population against ``x_column``.

    Reads the ``evaluation = "perturbation"`` rows written by
    ``common.perturbation.perturbation_summary_rows``. Same mean ± SD over seeds and
    dotted ceilings as the figure's held-out panel; the targeted cells are excluded from
    these populations and scored separately in the CSV. ``group`` overrides the series'
    group name, for figures that rename it (Figure 6 calls it "heldout").
    """
    rows = summary
    if "evaluation" in rows:
        rows = rows[rows["evaluation"] == "perturbation"]
    for series_group, cell_type, color, label in series:
        plotted = group or series_group
        seed_errorbar(
            ax, rows, x_column, plotted, metric, color, label, cell_type=cell_type
        )
        ceiling_line(ax, rows, x_column, metric, plotted, color, cell_type=cell_type)
    ax.set_ylabel(METRIC_LABELS[metric])
    return rows


#: Delta scatters are linear over +-this many Hz: most neurons change by far less than 20,
#: so the axis is cut rather than scaled, as for the rate scatters. 25 Hz keeps the
#: targeted cells' negative arm on the plot while filling the frame (2026-09-21).
DELTA_MAX_HZ = 25.0
#: The delta scatter holds far fewer points than a rate scatter, so markers of the same
#: nominal area read smaller. Sized up to match them by eye (2026-09-21).
DELTA_MARKER_SIZE = RATE_MARKER_SIZE * 2.2


def delta_rate_scatter(ax, deltas, symlog=False, threshold=10.0, limit=DELTA_MAX_HZ):
    """Teacher vs student Δrate per neuron; targeted cells marked.

    Linear over +-``limit`` Hz by default: a few neurons change by more than 100 Hz while
    most change by less than 20, and the tail is not what the panel is about. Pass
    ``symlog=True`` to scale the axes instead of cutting them, or ``limit=None`` to fit
    every neuron.
    """
    populations = (
        (deltas["targeted"] == 0) & (deltas["cell_type"] == "excitatory"),
        (deltas["targeted"] == 0) & (deltas["cell_type"] == "inhibitory"),
        deltas["targeted"] == 1,
    )
    styles = (
        (EXCITATORY_COLOR, "o", "Unobserved E"),
        (INHIBITORY_COLOR, "o", "Unobserved I"),
        (TARGETED_COLOR, "^", "Targeted I"),
    )
    if limit is None:
        limit = (
            float(
                np.nanpercentile(
                    deltas[["teacher_delta_rate_hz", "student_delta_rate_hz"]].abs(),
                    99.9,
                )
            )
            * 1.1
        )
    for mask, (color, marker, label) in zip(populations, styles):
        subset = deltas[mask]
        ax.scatter(
            subset["teacher_delta_rate_hz"],
            subset["student_delta_rate_hz"],
            s=DELTA_MARKER_SIZE,
            alpha=0.5,
            color=color,
            marker=marker,
            label=label,
            linewidths=0,
            rasterized=True,
        )
    ax.plot([-limit, limit], [-limit, limit], "k--", linewidth=1, alpha=0.5)
    if symlog:
        ax.set_xscale("symlog", linthresh=threshold)
        ax.set_yscale("symlog", linthresh=threshold)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    # Same tick spacing as the rate scatters, on both axes.
    ax.xaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.yaxis.set_major_locator(MultipleLocator(RATE_TICK_HZ))
    ax.set_aspect("equal")
    ax.set_xlabel("Teacher Δrate (Hz)")
    ax.set_ylabel("Student Δrate (Hz)")


def spike_raster(ax, spikes, neurons, duration_s):
    """Teacher (dark) above student (colour) for each neuron, with alternating bands.

    Args:
        spikes: DataFrame with neuron_id, observed, source, time_s.
        neurons: list of (neuron_id, observed) in top-to-bottom order.
    """
    for row, (neuron, observed) in enumerate(reversed(neurons)):
        base = row * 2.6
        if row % 2 == 0:
            ax.axhspan(base - 0.6, base + 1.9, color="#f2f2f2", zorder=0, linewidth=0)
        color = GROUP_COLORS["observed" if observed else "unobserved"]
        for offset, source, source_color in (
            (1.3, "teacher", "#333333"),
            (0.0, "student", color),
        ):
            times = spikes[
                (spikes["neuron_id"] == neuron) & (spikes["source"] == source)
            ]["time_s"]
            ax.eventplot(
                times.values,
                lineoffsets=base + offset,
                linelengths=1.0,
                linewidths=0.7,
                colors=source_color,
            )
        label = f"{'obs' if observed else 'unobs'} #{neuron}"
        ax.text(
            -0.01,
            base + 0.65,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6,
            color=color,
        )
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.8, len(neurons) * 2.6 - 0.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Time (s)")
