"""Figure 1 — evaluate every trained seed on held-out stimuli and write the CSVs.

Run after training:
    uv run python fig01-full-reconstruction/analysis.py

Writes, next to this script:
    fig01_summary.csv   seed, evaluation{held_out,perturbation}, group, cell_type, n_cells,
                        metric, value, ceiling_value
    fig01_rates.csv     neuron_id, cell_type, observed, seed, teacher/student rate, fluctuation_r2
    fig01_scaling_factors.csv   seed, scaling_factor, value, target
    fig01_spikes.csv    neuron_id, observed, seed, source, time_s   (raster, first seed)
    fig01_perturbation.csv  seed, neuron_id, cell_type, targeted, teacher/student/ceiling
                        delta_rate_hz   (one row per unobserved neuron)

The perturbation rows and CSV need the teacher's calibrated current
(``slurm/submit_perturbation.sh``); without it the held-out CSVs are still written.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    completed_runs,
    evaluate_run,
    pick_raster_neurons,
    rate_rows,
    scaling_factor_rows,
    spike_rows,
    summary_rows,
)
from common.perturbation import collect_perturbation

HERE = Path(__file__).resolve().parent
N_RASTER_OBSERVED = 3
N_RASTER_UNOBSERVED = 3


def main(runs_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No completed runs under {runs_dir}")

    summary, rates, factors = [], [], []
    for run in runs:
        print(f"Evaluating {run}")
        evaluation = evaluate_run(run, device)
        summary += summary_rows(evaluation)
        rates += rate_rows(evaluation)
        factors += scaling_factor_rows(run, seed=int(evaluation["seed"]))

    delta_summary, delta_rates = collect_perturbation(
        runs, lambda params, _: {}, device
    )
    summary += delta_summary

    first = evaluate_run(runs[0], device)
    neurons = pick_raster_neurons(first, N_RASTER_OBSERVED, N_RASTER_UNOBSERVED)
    spikes = spike_rows(first, neurons)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "fig01_summary.csv", index=False)
    pd.DataFrame(rates).to_csv(out_dir / "fig01_rates.csv", index=False)
    pd.DataFrame(spikes).to_csv(out_dir / "fig01_spikes.csv", index=False)
    pd.DataFrame(factors).to_csv(out_dir / "fig01_scaling_factors.csv", index=False)
    if delta_rates:
        pd.DataFrame(delta_rates).to_csv(
            out_dir / "fig01_perturbation.csv", index=False
        )
    print(
        summary.groupby(["evaluation", "group", "cell_type", "metric"])[
            ["value", "ceiling_value"]
        ].agg(["mean", "std"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=load_experiment_config(HERE / "experiment.toml")["output_dir"],
        help="grid-search output directory (default: output_dir of experiment.toml)",
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.runs, args.out)
