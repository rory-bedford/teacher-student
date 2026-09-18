"""Short diagnostic runs (a few epochs) that vary one setting at a time.

    uv run python slurm/prepare_probes.py [--epochs 12] [--gres v100] [--seed 44]

Writes one config per (condition, variant) under <data>/probes/<stamp>/, with a code
snapshot and a submit.sh, exactly like slurm/prepare_array.py but for throwaway runs:
outputs go to <data>/probes/<stamp>/runs/<name>, never into a figure's grid.
"""

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import toml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from slurm.prepare_array import clean_commit, snapshot_code

#: Conditions to probe: (label, figure folder, [student] overrides).
CONDITIONS = [
    ("obs0.02", "fig03-observed-fraction", {"observed_fraction": 0.02}),
    ("obs0.10", "fig03-observed-fraction", {"observed_fraction": 0.10}),
]
#: Variants: (label, {table: {key: value}}).
VARIANTS = [
    ("baseline", {}),
    ("nopenalty", {"loss": {"hidden_rate_mean": 0.0, "hidden_rate_std": 0.0}}),
    ("nostd", {"loss": {"hidden_rate_std": 0.0}}),
    ("lr2e-3", {"optimiser": {"lr_scaling": 2e-3, "lr_min_scaling": 2e-4}}),
    ("clip1", {"optimiser": {"grad_clip_scaling": 1.0}}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--gres", default="v100")
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()

    commit = clean_commit()
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    data = Path(
        toml.load(REPO / CONDITIONS[0][1] / "experiment.toml")["output_dir"]
    ).parent
    root = data / "probes" / stamp
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    code = snapshot_code(root / "code", commit)

    tasks = []
    for condition, figure, student in CONDITIONS:
        experiment = toml.load(REPO / figure / "experiment.toml")
        base = toml.load(REPO / figure / "parameters.toml")
        for variant, overrides in VARIANTS:
            params = deepcopy(base)
            params["simulation"]["seed"] = args.seed
            params["student"].update(student)
            params["training"]["total_epochs"] = args.epochs
            for table, values in overrides.items():
                params[table].update(values)
            name = f"{condition}__{variant}"
            params_file = root / "configs" / f"{name}.parameters.toml"
            toml.dump(params, open(params_file, "w"))
            run_experiment = deepcopy(experiment)
            run_experiment["script"] = str(
                code / Path(experiment["script"]).resolve().relative_to(REPO)
            )
            run_experiment["parameters_file"] = str(params_file)
            run_experiment["output_dir"] = str(root / "runs" / name)
            run_experiment["description"] = f"PROBE {name} ({args.epochs} epochs)"
            if run_experiment.get("wandb", {}).get("enabled", False):
                run_experiment["wandb"] = dict(run_experiment["wandb"])
                run_experiment["wandb"].update({"group": "probes", "notes": name})
            experiment_file = root / "configs" / f"{name}.experiment.toml"
            toml.dump(run_experiment, open(experiment_file, "w"))
            tasks.append(f"{experiment_file}\t{name}")

    (root / "tasks.txt").write_text("\n".join(tasks) + "\n")
    (root / "commit.txt").write_text(commit + "\n")
    (root / "code.txt").write_text(f"{code}\n")
    submit = root / "submit.sh"
    submit.write_text(
        "#!/bin/bash\n"
        f"# {len(tasks)} probe runs ({args.epochs} epochs), commit {commit[:8]}\n"
        "set -euo pipefail\n"
        f"sbatch --array=0-{len(tasks) - 1}%{args.concurrent} --time=02:00:00 "
        f"--gres=gpu:{args.gres}:1 --job-name=probe "
        f"--output={root}/logs/%a.log {REPO}/slurm/run_array.sbatch {root}\n"
    )
    submit.chmod(0o755)
    print(f"{len(tasks)} probes -> {root}")
    for line in tasks:
        print("  " + line.split("\t")[1])
    print(f"\nSubmit on the login node:\n  {submit}")


if __name__ == "__main__":
    main()
