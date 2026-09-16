"""Figure 3 — evaluate the observed-fraction sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply the 10% point):
    uv run python fig03-observed-fraction/analysis.py

Writes, next to this script:
    fig03_summary.csv          obs_fraction, n_observed, seed, group, metric, value, floor_value, ceiling_value
    fig03_rates.csv            obs_fraction, neuron_id, cell_type, observed, seed, rates, fluctuation_r2
    fig03_dimensionality.csv   participation_ratio, n_pcs_90pct_var, smoothing_sigma_ms, window_s, n_neurons
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    collect,
    completed_runs,
    held_out_trial,
    run_parameters,
    smooth,
)

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"


def label(params, evaluation):
    return {
        "obs_fraction": float(params["student"]["observed_fraction"]),
        "n_observed": int(evaluation["n_observed"]),
    }


def teacher_dimensionality(run_dir, device):
    """Participation ratio of the teacher's held-out activity, smoothed as for Fluctuation R².

    Smoothing, burn-in and window are the evaluation's own, so the number is commensurable
    with the metric reported everywhere else.
    """
    evaluation_cfg = run_parameters(run_dir)["evaluation"]
    trial = zarr.open_group(held_out_trial(run_dir, device), mode="r")
    dt = float(trial.attrs["dt"])
    burn_in = int(evaluation_cfg["burn_in_ms"] / dt)
    spikes = np.array(trial["output_spikes"][0, burn_in:, :])
    activity = torch.from_numpy(
        smooth(spikes, evaluation_cfg["fluctuation_tau_ms"], dt)
    ).double()
    activity = activity.to(device) - activity.to(device).mean(dim=0)
    covariance = activity.T @ activity / (activity.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0).flip(0).cpu().numpy()
    explained = np.cumsum(eigenvalues) / eigenvalues.sum()
    return {
        "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum()),
        "n_pcs_90pct_var": int(np.searchsorted(explained, 0.9) + 1),
        "smoothing_sigma_ms": float(evaluation_cfg["fluctuation_tau_ms"]),
        "window_s": spikes.shape[0] * dt / 1000.0,
        "n_neurons": int(spikes.shape[1]),
    }


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_runs = completed_runs(baseline_dir)
    runs = baseline_runs + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)
    dimensionality = pd.DataFrame([teacher_dimensionality(runs[0], device)])

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig03_summary.csv", index=False)
    rates.to_csv(out_dir / "fig03_rates.csv", index=False)
    dimensionality.to_csv(out_dir / "fig03_dimensionality.csv", index=False)
    print(
        summary.groupby(["obs_fraction", "group", "metric"])[
            ["value", "floor_value", "ceiling_value"]
        ].mean()
    )
    print(dimensionality)


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
