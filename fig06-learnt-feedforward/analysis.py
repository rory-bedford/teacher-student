"""Figure 6 — evaluate the reconstructed-fraction sweep on held-out stimuli.

Run after training:
    uv run python fig06-learnt-feedforward/analysis.py

Groups follow the README: "observed" are modelled neurons from the recorded pool (in the
loss); "heldout" are modelled neurons from the held-out pool (simulated, never in the
loss). kappa is the fraction of each modelled neuron's input volume (feedforward +
recurrent teacher weights) that comes from inside the reconstructed segment.

Writes, next to this script:
    fig06_summary.csv   reconstructed_fraction, recorded_pool_fraction, kappa, n_free_params, n_in_loss,
                        seed, group, metric, value, floor_value, ceiling_value
    fig06_rates.csv     reconstructed_fraction, recorded_pool_fraction, neuron_id, cell_type, group, seed,
                        teacher_rate_hz, student_rate_hz, fluctuation_r2
    fig06_spikes.csv    reconstructed_fraction, neuron_id, group, seed, source, time_s
                        (one observed and one held-out neuron per level, first seed)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    completed_runs,
    evaluate_run,
    pick_raster_neurons,
    rate_rows,
    run_parameters,
    spike_rows,
    summary_rows,
)

HERE = Path(__file__).resolve().parent
GROUP_NAMES = {"observed": "observed", "unobserved": "heldout"}


def main(runs_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No completed runs under {runs_dir}")

    summary, rates, spikes = [], [], []
    rastered_levels = set()
    for run in sorted(runs, key=lambda r: run_parameters(r)["simulation"]["seed"]):
        print(f"Evaluating {run}")
        evaluation = evaluate_run(run, device)
        student = run_parameters(run)["student"]
        modelled = np.concatenate(
            [evaluation["observed_ids"], evaluation["unobserved_ids"]]
        )
        labels = {
            "reconstructed_fraction": float(student["reconstructed_fraction"]),
            "recorded_pool_fraction": float(student["recorded_pool_fraction"]),
        }
        for row in summary_rows(
            evaluation,
            **labels,
            kappa=float(np.mean(evaluation["kappa"][modelled])),
            n_free_params=int(evaluation["n_free_params"]),
            n_in_loss=int(evaluation["n_observed"]),
        ):
            row["group"] = GROUP_NAMES[row["group"]]
            summary.append(row)
        for row in rate_rows(evaluation, **labels):
            row["group"] = "observed" if row.pop("observed") else "heldout"
            rates.append(row)

        level = (labels["reconstructed_fraction"], labels["recorded_pool_fraction"])
        if level not in rastered_levels and evaluation["n_unobserved"] > 0:
            rastered_levels.add(level)
            for row in spike_rows(
                evaluation, pick_raster_neurons(evaluation, 1, 1), **labels
            ):
                row["group"] = "observed" if row.pop("observed") else "heldout"
                spikes.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "fig06_summary.csv", index=False)
    pd.DataFrame(rates).to_csv(out_dir / "fig06_rates.csv", index=False)
    pd.DataFrame(spikes).to_csv(out_dir / "fig06_spikes.csv", index=False)
    print(
        summary.groupby(
            ["recorded_pool_fraction", "reconstructed_fraction", "group", "metric"]
        )[["kappa", "n_free_params", "n_in_loss", "value", "ceiling_value"]].mean()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=load_experiment_config(HERE / "experiment.toml")["output_dir"],
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.runs, args.out)
