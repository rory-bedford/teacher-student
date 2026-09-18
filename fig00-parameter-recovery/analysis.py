"""Figure 0 — collect the six tied scaling factors each run recovered.

Run after training (no held-out evaluation is needed; the factors and the targets they
should recover are already logged by training):

    uv run python fig00-parameter-recovery/analysis.py

Writes, next to this script:
    fig00_scaling_factors.csv   figure, run, observed_fraction, seed, scaling_factor,
                                value, target

Sources, all read through each figure's ``experiment.toml``:
    Figure 1                the headline 50%-observed runs
    Figure 2                its fully observed full-connectome runs (every neuron
                            teacher-forced: the identifiability anchor at 1.0)
    Figure 3                the observed-fraction sweep below 50%
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import completed_runs, run_parameters, scaling_factor_rows

HERE = Path(__file__).resolve().parent
#: (label, figure folder, glob over that figure's grid). The fully observed runs are
#: Figure 2's, where observed_fraction = 1.0 leaves no unobserved population at all.
SOURCES = (
    ("fig01", "fig01-full-reconstruction", "*"),
    ("fig02", "fig02-controls", "connectome-fully-observed__*"),
    ("fig03", "fig03-observed-fraction", "*"),
)


def grid_dir(figure):
    return Path(
        load_experiment_config(HERE.parent / figure / "experiment.toml")["output_dir"]
    )


def main(out_dir):
    rows = []
    for label, figure, pattern in SOURCES:
        grid = grid_dir(figure)
        for run in sorted(set(completed_runs(grid)) & set(grid.glob(pattern))):
            params = run_parameters(run)
            factors = scaling_factor_rows(
                run,
                figure=label,
                run=run.name,
                seed=int(params["simulation"]["seed"]),
                observed_fraction=float(params["student"]["observed_fraction"]),
            )
            if not factors:  # learnt recurrence has no tied recurrent factors
                print(f"  {run.name}: no tied scaling factors")
                continue
            rows += factors
            print(f"  {label} {run.name}: {len(factors)} factors")
    if not rows:
        raise SystemExit("No completed runs with scaling factors")

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "fig00_scaling_factors.csv", index=False)
    ratio = table["value"] / table["target"]
    print(
        pd.DataFrame({"observed_fraction": table["observed_fraction"], "ratio": ratio})
        .groupby("observed_fraction")["ratio"]
        .agg(["count", "min", "max"])
        .round(3)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.out)
