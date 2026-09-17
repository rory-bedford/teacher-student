"""Expand figure grids into one run config per SLURM array task.

For each figure, runs that figure's own ``custom_config_generator`` (from
``run_grid_search.py``), skips runs that already finished, and writes for every
remaining run a ``parameters.toml`` and ``experiment.toml`` (output_dir =
``<grid dir>/<run name>``, exactly as ``./run --grid`` would). The list of experiment
configs goes to ``tasks.txt`` and the current commit to ``commit.txt``.

The code is snapshotted too: a detached git worktree of that commit at
``<array dir>/code``, which every task runs from (with this repo's ``.venv``). You can
keep committing and editing here while arrays are queued; a task only ever sees the code
its configs were generated with. The connectome-snns library is *not* snapshotted (it is
an editable install), so library changes do reach queued tasks; the run framework records
the library commit and dirty flag in each run's metadata.

    uv run python slurm/prepare_array.py fig03-observed-fraction fig04-reconstruction-errors

prints the ``sbatch`` command for each figure. Nothing is submitted. Add ``--dry-run`` to
list the runs without writing anything. ``--split a10080g:2,a40:2`` deals the runs out
over one array per GPU type (``--gres=gpu:<type>:1``), each throttled to that many GPUs;
the throttles may add up to at most ``MAX_CONCURRENT``.
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

#: Wall-clock request per figure (measured on a Quadro RTX 5000, with margin).
TIME_LIMITS = {
    "fig01-full-reconstruction": "06:00:00",
    "fig02-controls": "06:00:00",
    "fig03-observed-fraction": "06:00:00",
    "fig04-reconstruction-errors": "06:00:00",
    "fig05-weight-noise": "06:00:00",
    "fig06-learnt-feedforward": "10:00:00",
}


#: Cluster GPUs one figure's array may hold at once (the user's cap is 8 in total).
MAX_CONCURRENT = 8


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


def prepare(figure, commit, stamp, dry_run=False, split=None):
    folder = REPO / figure
    experiment_path = folder / "experiment.toml"
    experiment = toml.load(experiment_path)
    grid_dir = Path(experiment["output_dir"])

    spec = importlib.util.spec_from_file_location("grid", folder / "run_grid_search.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    generator = skip_completed(
        experiment_path, grid.custom_config_generator, unfinished="skip"
    )
    runs = list(generator(toml.load(experiment["parameters_file"])))
    if not runs:
        print(f"{figure}: every run already finished")
        return
    if dry_run:
        print(f"\n{figure}: {len(runs)} runs")
        for _, description in runs:
            print(f"  {description}")
        return

    array_dir = grid_dir / "slurm" / stamp
    config_dir = array_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    code_dir = snapshot_code(array_dir / "code", commit)
    shutil.copy2(folder / "run_grid_search.py", grid_dir / "run_grid_search.py")

    tasks = []
    for index, (params, description) in enumerate(runs):
        params_file = config_dir / f"params_{index:03d}.toml"
        with open(params_file, "w") as f:
            toml.dump(params, f)
        run_experiment = deepcopy(experiment)
        run_experiment["script"] = str(
            code_dir / Path(experiment["script"]).resolve().relative_to(REPO)
        )
        run_experiment["parameters_file"] = str(params_file)
        run_experiment["output_dir"] = str(grid_dir / description)
        if run_experiment.get("wandb", {}).get("enabled", False):
            run_experiment["wandb"]["notes"] = description
        experiment_file = config_dir / f"experiment_{index:03d}.toml"
        with open(experiment_file, "w") as f:
            toml.dump(run_experiment, f)
        tasks.append(f"{experiment_file}\t{description}")

    (array_dir / "commit.txt").write_text(commit + "\n")
    job = figure.split("-")[0]
    if not split:
        (array_dir / "code.txt").write_text(f"{code_dir}\n")
        (array_dir / "tasks.txt").write_text("\n".join(tasks) + "\n")
        (array_dir / "logs").mkdir(exist_ok=True)
        print(f"\n{figure}: {len(tasks)} runs -> {array_dir}")
        print(sbatch_line(array_dir, len(tasks), MAX_CONCURRENT, figure, job))
        return

    # One array per GPU type, dealing runs out in turn so each gets a mix of the grid.
    print(f"\n{figure}: {len(tasks)} runs -> {array_dir}")
    for part, (gres, throttle) in enumerate(split):
        part_tasks = tasks[part :: len(split)]
        part_dir = array_dir / gres
        (part_dir / "logs").mkdir(parents=True, exist_ok=True)
        (part_dir / "tasks.txt").write_text("\n".join(part_tasks) + "\n")
        (part_dir / "commit.txt").write_text(commit + "\n")
        (part_dir / "code.txt").write_text(f"{code_dir}\n")
        print(
            sbatch_line(
                part_dir,
                len(part_tasks),
                throttle,
                figure,
                f"{job}-{gres}",
                f"--gres=gpu:{gres}:1 ",
            )
        )


def sbatch_line(array_dir, n_tasks, throttle, figure, job, gres=""):
    return (
        f"  sbatch --array=0-{n_tasks - 1}%{throttle} --time={TIME_LIMITS[figure]} "
        f"{gres}--job-name={job} --output={array_dir}/logs/%a.log "
        f"{REPO}/slurm/run_array.sbatch {array_dir}"
    )


def parse_split(argv):
    """``--split a10080g:2,a40:2`` -> [("a10080g", 2), ("a40", 2)]."""
    if "--split" not in argv:
        return None
    parts = argv[argv.index("--split") + 1].split(",")
    split = [(gres, int(throttle)) for gres, throttle in (p.split(":") for p in parts)]
    if sum(throttle for _, throttle in split) > MAX_CONCURRENT:
        raise SystemExit(f"--split throttles add up to more than {MAX_CONCURRENT} GPUs")
    return split


def main():
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    split = parse_split(argv)
    if split:
        del argv[argv.index("--split") : argv.index("--split") + 2]
    figures = [a for a in argv if a != "--dry-run"]
    if not figures:
        raise SystemExit(__doc__)
    commit = "dry-run" if dry_run else clean_commit()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    print(f"Commit {commit[:8]}")
    for figure in figures:
        prepare(figure, commit, stamp, dry_run, split)


if __name__ == "__main__":
    main()
