"""Figure 3 — evaluate the observed-fraction sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply the 50% point):
    uv run python fig03-observed-fraction/analysis.py

Writes, next to this script:
    fig03_summary.csv          obs_fraction, n_observed, seed, evaluation{held_out,perturbation},
                               group, cell_type, n_cells, metric, value, ceiling_value
    fig03_rates.csv            obs_fraction, neuron_id, cell_type, observed, seed, rates, fluctuation_r2

The perturbation rows need the teacher's calibrated current
(``slurm/submit_perturbation.sh``); without it only the held-out rows are written.

The teacher's dimensionality (the band marked on panel a) is not computed here: figures.py
reads generate-teacher-activity/teacher_dimensionality.csv (generate-teacher-activity/dimensionality.py).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    collect,
    completed_runs,
    run_parameters,
)
from common.perturbation import collect_perturbation

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"
#: Fractions reported by the figure, matching run_grid_search.OBSERVED_FRACTIONS. Runs at
#: 0.5% and 1% observed finished but are not reported: below ~25% the fit is unreliable
#: (see run_grid_search.py), 2% is kept to show the cliff. Their run folders are still on
#: disk, so add a fraction back here to include it again.
REPORTED_FRACTIONS = (0.02, 0.05, 0.25, 0.5)  # 0.5 comes from Figure 1


def reported(run_dir):
    fraction = float(run_parameters(run_dir)["student"]["observed_fraction"])
    return any(abs(fraction - f) < 1e-9 for f in REPORTED_FRACTIONS)


def label(params, evaluation):
    return {
        "obs_fraction": float(params["student"]["observed_fraction"]),
        "n_observed": int(evaluation["n_observed"]),
    }


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_runs = completed_runs(baseline_dir)
    runs = baseline_runs + completed_runs(runs_dir)
    skipped = [r for r in runs if not reported(r)]
    for run in skipped:
        print(f"  not reported (observed fraction outside the figure): {run.name}")
    runs = [r for r in runs if reported(r)]
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)
    delta_summary, _ = collect_perturbation(runs, label, device)
    summary = pd.concat([summary, pd.DataFrame(delta_summary)], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig03_summary.csv", index=False)
    rates.to_csv(out_dir / "fig03_rates.csv", index=False)
    print(
        summary.groupby(["obs_fraction", "evaluation", "group", "cell_type", "metric"])[
            ["value", "ceiling_value"]
        ].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=load_experiment_config(HERE / "experiment.toml")["output_dir"],
    )
    parser.add_argument(
        "--baseline", type=Path, default=load_experiment_config(BASELINE)["output_dir"]
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.runs, args.baseline, args.out)
