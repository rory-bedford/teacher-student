"""Plot style and helpers shared by every figure's ``figures.py``.

Figures are written at final size: 12 cm wide, fonts legible on a projected slide.
All colours come from ``connectome_snns.visualization``.
"""

import matplotlib.pyplot as plt
import numpy as np
from connectome_snns.visualization import (
    FIGURE_BLUE,
    FIGURE_CORAL,
    FLOOR_COLOR,
    OBSERVED_COLOR,
    UNOBSERVED_COLOR,
    use_project_style,
)

CM = 1 / 2.54
FIGURE_WIDTH = 12 * CM

EXCITATORY_COLOR = FIGURE_CORAL
INHIBITORY_COLOR = FIGURE_BLUE
GROUP_COLORS = {"observed": OBSERVED_COLOR, "unobserved": UNOBSERVED_COLOR}
GROUP_LABELS = {"observed": "Observed", "unobserved": "Unobserved"}
METRIC_LABELS = {"fluctuation_r2": "Fluctuation R²", "activity_r2": "Activity R²"}


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
    """``label`` with Fluctuation/Activity R², floor and ceiling (one seed, or the mean)."""
    rows = summary[summary["group"] == group]
    if seed is not None:
        rows = rows[rows["seed"] == seed]
    lines = [label]
    for metric, short in (("fluctuation_r2", "Flu"), ("activity_r2", "Act")):
        m = rows[rows["metric"] == metric]
        lines.append(
            f"{short} R² {m['value'].mean():.2f} "
            f"[floor {m['floor_value'].mean():.2f}, ceiling {m['ceiling_value'].mean():.2f}]"
        )
    return "\n".join(lines)


def seed_errorbar(
    ax, summary, x_column, group, metric, color, label, marker="o", linestyle="-"
):
    """Mean ± SD over seeds against ``x_column``, individual seeds as faint points."""
    rows = summary[(summary["group"] == group) & (summary["metric"] == metric)]
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


def floor_line(ax, summary, x_column, metric, group, color=FLOOR_COLOR, label="Floor"):
    """Dashed shuffled-identity floor for one group, averaged over seeds at each x."""
    rows = summary[(summary["metric"] == metric) & (summary["group"] == group)]
    stats = rows.groupby(x_column)["floor_value"].mean().reset_index()
    ax.plot(
        stats[x_column],
        stats["floor_value"],
        linestyle="--",
        color=color,
        linewidth=0.8,
        label=label,
    )


def ceiling_line(ax, summary, x_column, metric, group, color, label=None):
    """Dotted ceiling (perfectly specified student) for one group, averaged over seeds."""
    rows = summary[(summary["metric"] == metric) & (summary["group"] == group)]
    stats = rows.groupby(x_column)["ceiling_value"].mean().reset_index()
    ax.plot(
        stats[x_column],
        stats["ceiling_value"],
        linestyle=":",
        color=color,
        linewidth=1.0,
        label=label,
    )


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
