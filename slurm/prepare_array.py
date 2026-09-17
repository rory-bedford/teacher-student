"""Expand figure grids into SLURM array tasks, one run per task.

For each figure, runs that figure's own ``custom_config_generator`` (from
``run_grid_search.py``), skips runs that already finished, and writes for every
remaining run a ``parameters.toml`` and ``experiment.toml`` (output_dir =
``<grid dir>/<run name>``, exactly as ``./run --grid`` would).

All requested figures go into ONE submission at ``<data>/slurm/<stamp>/`` so a single
throttle covers them: one array per GPU type, each a ``tasks.txt`` of experiment configs,
ordered by seed so the first seed of every condition runs first. ``submit.sh`` there
submits them all.

The code is snapshotted too: a detached git worktree of the current commit at
``<stamp>/code``, which every task runs from (with this repo's ``.venv``). You can keep
committing and editing here while arrays are queued; a task only ever sees the code its
configs were generated with. The connectome-snns library is *not* snapshotted (it is an
editable install), so library changes do reach queued tasks; the run framework records
the library commit and dirty flag in each run's metadata.

    uv run python slurm/prepare_array.py [--dry-run] [--seeds 44[,45]] \\
        [--runs obs-0.02__seed-44,wn-0.3__seed-44] [--split v100:4[,a40:2]] \\
        fig01-full-reconstruction fig02-controls ...

``--seeds`` and ``--runs`` (run folder names) restrict which grid runs are prepared.

``--split <gres>:<n>,...`` deals the runs out over one array per GPU type
(``--gres=gpu:<gres>:1``), each throttled to ``n`` GPUs; the throttles may add up to at
most ``MAX_CONCURRENT``. Without it: one array on the sbatch script's default GPU,
throttled to ``MAX_CONCURRENT``. Nothing is submitted; run the printed ``submit.sh``.
"""

import importlib.util
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.grid import skip_completed

#: Wall-clock request per figure (measured ~3.6 h / ~5 h on a Quadro RTX 5000, with
#: margin for slower nodes). A submission uses the longest of its figures.
TIME_LIMITS = {
    "fig01-full-reconstruction": "08:00:00",
    "fig02-controls": "08:00:00",
    "fig03-observed-fraction": "08:00:00",
    "fig04-reconstruction-errors": "08:00:00",
    "fig05-weight-noise": "08:00:00",
    "fig06-learnt-feedforward": "10:00:00",
}

#: Most cluster GPUs one submission may hold at once. No hard cluster limit, only fair use;
#: the user OKed up to ~16 while the cluster is quiet (2026-09-17).
MAX_CONCURRENT = 16


def clean_commit():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise SystemExit(f"Commit your changes first; the repo is dirty:\n{status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def snapshot_code(code_dir, commit):
    """Detached worktree of ``commit``; the tasks run the figure scripts from here."""
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(code_dir), commit],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return code_dir


def pending_runs(figure, seeds=None, names=None):
    """(experiment config, grid dir, [(params, run name)]) still to run for a figure."""
    folder = REPO / figure
    experiment_path = folder / "experiment.toml"
    experiment = toml.load(experiment_path)
    spec = importlib.util.spec_from_file_location(figure, folder / "run_grid_search.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    generator = skip_completed(
        experiment_path, grid.custom_config_generator, unfinished="skip"
    )
    runs = list(generator(toml.load(experiment["parameters_file"])))
    if seeds is not None:
        runs = [r for r in runs if r[0]["simulation"]["seed"] in seeds]
    if names is not None:
        runs = [r for r in runs if r[1] in names]
    return experiment, Path(experiment["output_dir"]), runs


def write_configs(figure, experiment, grid_dir, runs, submission_dir, code_dir):
    """Write one experiment/parameters pair per run; return [(seed, task line)]."""
    config_dir = submission_dir / "configs" / figure
    config_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / figure / "run_grid_search.py", grid_dir / "run_grid_search.py")
    script = code_dir / Path(experiment["script"]).resolve().relative_to(REPO)
    tasks = []
    for index, (params, description) in enumerate(runs):
        params_file = config_dir / f"params_{index:03d}.toml"
        with open(params_file, "w") as f:
            toml.dump(params, f)
        run_experiment = deepcopy(experiment)
        run_experiment["script"] = str(script)
        run_experiment["parameters_file"] = str(params_file)
        run_experiment["output_dir"] = str(grid_dir / description)
        if run_experiment.get("wandb", {}).get("enabled", False):
            run_experiment["wandb"]["notes"] = description
        experiment_file = config_dir / f"experiment_{index:03d}.toml"
        with open(experiment_file, "w") as f:
            toml.dump(run_experiment, f)
        tasks.append(
            (params["simulation"]["seed"], f"{experiment_file}\t{figure}/{description}")
        )
    return tasks


def parse_list(argv, flag):
    if flag not in argv:
        return None
    value = argv[argv.index(flag) + 1]
    del argv[argv.index(flag) : argv.index(flag) + 2]
    return value.split(",")


def main():
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    seeds = parse_list(argv, "--seeds")
    seeds = None if seeds is None else {int(s) for s in seeds}
    names = parse_list(argv, "--runs")
    names = None if names is None else set(names)
    split = parse_list(argv, "--split")
    split = (
        [(gres, int(n)) for gres, n in (p.split(":") for p in split)]
        if split
        else [(None, MAX_CONCURRENT)]
    )
    if sum(n for _, n in split) > MAX_CONCURRENT:
        raise SystemExit(f"--split throttles add up to more than {MAX_CONCURRENT} GPUs")
    figures = argv
    if not figures:
        raise SystemExit(__doc__)

    pending = {figure: pending_runs(figure, seeds, names) for figure in figures}
    for figure, (_, _, runs) in pending.items():
        print(f"{figure}: {len(runs)} runs")
        if dry_run:
            for _, description in runs:
                print(f"  {description}")
    pending = {f: p for f, p in pending.items() if p[2]}
    if dry_run or not pending:
        return

    commit = clean_commit()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    data_dir = next(iter(pending.values()))[1].parent
    submission_dir = data_dir / "slurm" / stamp
    code_dir = snapshot_code(submission_dir / "code", commit)

    tasks = []
    for figure, (experiment, grid_dir, runs) in pending.items():
        tasks += write_configs(
            figure, experiment, grid_dir, runs, submission_dir, code_dir
        )
    # Stable sort: every condition's first seed before any condition's second.
    tasks = [line for _, line in sorted(tasks, key=lambda t: t[0])]
    time_limit = max(TIME_LIMITS[f] for f in pending)

    lines = [
        "#!/bin/bash",
        f"# {len(tasks)} runs, commit {commit[:8]}",
        "set -euo pipefail",
    ]
    for part, (gres, throttle) in enumerate(split):
        part_tasks = tasks[part :: len(split)]
        if not part_tasks:
            continue
        part_dir = submission_dir / (gres or "default")
        (part_dir / "logs").mkdir(parents=True, exist_ok=True)
        (part_dir / "tasks.txt").write_text("\n".join(part_tasks) + "\n")
        (part_dir / "commit.txt").write_text(commit + "\n")
        (part_dir / "code.txt").write_text(f"{code_dir}\n")
        gres_flag = f"--gres=gpu:{gres}:1 " if gres else ""
        lines.append(
            f"sbatch --array=0-{len(part_tasks) - 1}%{throttle} --time={time_limit} "
            f"{gres_flag}--job-name=bern-{gres or 'default'} "
            f"--output={part_dir}/logs/%a.log {REPO}/slurm/run_array.sbatch {part_dir}"
        )
    submit = submission_dir / "submit.sh"
    submit.write_text("\n".join(lines) + "\n")
    submit.chmod(0o755)
    print(f"\n{len(tasks)} runs, commit {commit[:8]} -> {submission_dir}")
    print(f"Submit on the login node:\n  {submit}")


if __name__ == "__main__":
    main()
