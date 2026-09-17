"""Figure 1 — evaluate every trained seed on held-out stimuli and write the CSVs.

Run after training:
    uv run python fig01-full-reconstruction/analysis.py

Writes, next to this script:
    fig01_summary.csv   seed, group, metric, value, floor_value, ceiling_value,
                        noise_ceiling_value
    fig01_rates.csv     neuron_id, cell_type, observed, seed, teacher/student rate, fluctuation_r2
    fig01_spikes.csv    neuron_id, observed, seed, source, time_s   (raster, first seed)
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
    spike_rows,
    summary_rows,
)

HERE = Path(__file__).resolve().parent
N_RASTER_OBSERVED = 2
N_RASTER_UNOBSERVED = 3


def main(runs_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No completed runs under {runs_dir}")

    summary, rates = [], []
    for run in runs:
        print(f"Evaluating {run}")
        evaluation = evaluate_run(run, device)
        summary += summary_rows(evaluation)
        rates += rate_rows(evaluation)

    first = evaluate_run(runs[0], device)
    neurons = pick_raster_neurons(first, N_RASTER_OBSERVED, N_RASTER_UNOBSERVED)
    spikes = spike_rows(first, neurons)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / "fig01_summary.csv", index=False)
    pd.DataFrame(rates).to_csv(out_dir / "fig01_rates.csv", index=False)
    pd.DataFrame(spikes).to_csv(out_dir / "fig01_spikes.csv", index=False)
    print(
        pd.DataFrame(summary)
        .groupby(["group", "metric"])[["value", "floor_value"]]
        .agg(["mean", "std"])
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
