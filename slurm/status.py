"""Write STATUS.md: every figure's grid runs, finished / running / crashed / to do, with a
training-health check on each run that has metrics.

    uv run python slurm/status.py

Reads only the run directories under ../bernstein (and ../bernstein/_tests); it never
touches SLURM, so "running" means "metrics written in the last STALE_MINUTES".

Health flags (heuristics meant to catch wasted compute early, not to judge results):
- loss NaN; after WARMUP of training, van Rossum loss now > RISING_FACTOR x its minimum
- after WARMUP, an unobserved population's rate < RATE_LOW or > RATE_HIGH x the observed
  teacher's
- after half of training, a scaling factor further than SF_OFF from its target (ratio);
  mitral scaling factors are skipped for Figure 6, where learnt weights absorb them
- a finished or running run whose [training]/[optimiser]/[loss] tables differ from the
  figure's current parameters.toml ("old recipe")
"""

import importlib.util
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FIGURES = [
    "fig01-full-reconstruction",
    "fig02-controls",
    "fig03-observed-fraction",
    "fig04-reconstruction-errors",
    "fig05-weight-noise",
    "fig06-learnt-feedforward",
]
RECIPE_TABLES = ["training", "optimiser", "loss"]
STALE_MINUTES = 30
RISING_FACTOR = 1.3
RATE_LOW, RATE_HIGH = 0.3, 3.0
SF_OFF = 1.5
WARMUP = 0.1
TRIAL_MS = 15000  # teacher trial length; chunks per epoch = TRIAL_MS // chunk_size


def grid_runs(figure):
    folder = REPO / figure
    experiment = toml.load(folder / "experiment.toml")
    base = toml.load(folder / "parameters.toml")
    spec = importlib.util.spec_from_file_location(figure, folder / "run_grid_search.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    names = [name for _, name in grid.custom_config_generator(base)]
    return Path(experiment["output_dir"]), base, names


def recipe_matches(run_params, base):
    return all(run_params.get(t, {}) == base.get(t, {}) for t in RECIPE_TABLES)


def health(run_dir, params):
    """(progress text, latest loss text, list of flags) from training_metrics.csv."""
    metrics_file = run_dir / "training_metrics.csv"
    if not metrics_file.exists():
        return "", "", []
    try:
        m = pd.read_csv(metrics_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return "", "", []
    if m.empty:
        return "", "", []
    # The metrics' "epoch" column counts training chunks.
    total_epochs = params["training"]["total_epochs"]
    chunks_per_epoch = TRIAL_MS // params["simulation"]["chunk_size"]
    epoch = float(m["epoch"].iloc[-1]) / chunks_per_epoch
    progress = epoch / total_epochs
    flags = []

    vr = m["van_rossum_loss"]
    first, low, last = vr.iloc[0], vr.min(), vr.iloc[-1]
    if m["total_loss"].isna().any() or not math.isfinite(last):
        flags.append("loss NaN")
    elif progress >= WARMUP and last > RISING_FACTOR * low:
        flags.append(f"VR rising ({low:.0f} -> {last:.0f})")
    loss_text = f"{first:.0f} -> {last:.0f}"

    last_row = m.iloc[-1]
    for cell_type in ["excitatory", "inhibitory"]:
        student = last_row.get(f"firing_rate/student_unobserved_{cell_type}")
        teacher = last_row.get(f"firing_rate/teacher_observed_{cell_type}")
        if progress < WARMUP or student is None or teacher is None or not teacher > 0:
            continue
        ratio = student / teacher
        if not RATE_LOW <= ratio <= RATE_HIGH:
            flags.append(
                f"unobs {cell_type[:3]} rate {student:.1f} vs {teacher:.1f} Hz"
            )

    if progress >= 0.5:
        for column in m.columns:
            learnt_ff = "reconstructed_fraction" in params.get("student", {})
            if learnt_ff and column.startswith("scaling_factors/mitral"):
                continue
            if column.startswith("scaling_factors/") and column.endswith("_value"):
                target = last_row.get(column.replace("_value", "_target"))
                if target and target > 0:
                    ratio = last_row[column] / target
                    if not 1 / SF_OFF <= ratio <= SF_OFF:
                        name = column.split("/")[1].removesuffix("_value")
                        flags.append(f"SF {name} at {ratio:.2f}x target")

    return f"{epoch:.1f}/{total_epochs}", loss_text, flags


def run_row(name, run_dir, base):
    if not run_dir.exists():
        return {"run": name, "state": "to do"}
    params_file = run_dir / "parameters.toml"
    params = toml.load(params_file) if params_file.exists() else base
    progress, loss, flags = health(run_dir, params)
    if (run_dir / "final_model_state.pt").exists():
        state = "finished"
    elif (run_dir / "log.err").exists():
        state = "CRASHED"
    else:
        metrics = run_dir / "training_metrics.csv"
        newest = (
            metrics.stat().st_mtime if metrics.exists() else run_dir.stat().st_mtime
        )
        stale = time.time() - newest > STALE_MINUTES * 60
        state = (
            f"stalled? (no metrics for >{STALE_MINUTES} min)" if stale else "running"
        )
    if params_file.exists() and not recipe_matches(params, base):
        flags.insert(0, "old recipe")
    return {
        "run": name,
        "state": state,
        "epoch": progress,
        "VR loss": loss,
        "flags": "; ".join(flags),
    }


def table(rows):
    columns = ["run", "state", "epoch", "VR loss", "flags"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main():
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    out = [
        "# Run status",
        "",
        f"Generated {now} by `uv run python slurm/status.py` (do not edit by hand).",
        "",
        (
            "States come from the run folders: *finished* has `final_model_state.pt`, "
            "*CRASHED* has `log.err`, *running* wrote metrics in the last "
            f"{STALE_MINUTES} min. VR loss is first logged -> latest. Flags: see "
            "`slurm/status.py`."
        ),
        "",
    ]
    summary = [
        "| figure | finished | running | crashed | to do | flagged |",
        "|---|---|---|---|---|---|",
    ]
    sections = []
    for figure in FIGURES:
        grid_dir, base, names = grid_runs(figure)
        rows = [run_row(n, grid_dir / n, base) for n in names]
        extra = sorted(
            p.name
            for p in grid_dir.glob("*")
            if p.is_dir() and p.name != "slurm" and p.name not in names
        )
        count = {
            s: sum(r["state"].startswith(s) for r in rows)
            for s in ["finished", "running", "CRASHED", "to do", "stalled"]
        }
        flagged = sum(bool(r.get("flags")) for r in rows)
        summary.append(
            f"| {figure} | {count['finished']}/{len(rows)} | {count['running']} | "
            f"{count['CRASHED'] + count['stalled']} | {count['to do']} | {flagged} |"
        )
        sections += [f"## {figure}", "", table(rows), ""]
        if extra:
            sections += [
                f"Folders that are not grid runs: {', '.join(extra)}",
                "",
            ]

    tests_dir = (
        Path(toml.load(REPO / FIGURES[0] / "experiment.toml")["output_dir"]).parent
        / "_tests"
    )
    if tests_dir.exists():
        rows = []
        for run_dir in sorted(p for p in tests_dir.iterdir() if p.is_dir()):
            figure = next((f for f in FIGURES if run_dir.name.startswith(f[:5])), None)
            base = toml.load(REPO / figure / "parameters.toml") if figure else {}
            rows.append(run_row(run_dir.name, run_dir, base))
        sections += [
            "## Tests (`bernstein/_tests`, not grid runs)",
            "",
            table(rows),
            "",
        ]

    (REPO / "STATUS.md").write_text("\n".join(out + summary + [""] + sections))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
