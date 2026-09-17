"""Participation ratio and PCA spectrum of the teacher's activity.

Spikes of all recurrent neurons are smoothed with the Gaussian used for Fluctuation R²
(sigma and burn-in from ``fig01-full-reconstruction/parameters.toml`` [evaluation]); the
covariance is pooled over every training trial after its burn-in, with one mean per
neuron over all of them. PCA is the eigendecomposition of that covariance.

    uv run python generate-teacher-activity/dimensionality.py

Writes, next to this script:
    teacher_pca_spectrum.csv       component, eigenvalue, variance_fraction, cumulative_fraction
    teacher_dimensionality.csv     participation_ratio, n_pcs_50/80/90/95pct_var, smoothing_sigma_ms,
                                   burn_in_ms, n_trials, seconds_per_trial, n_neurons
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import torch
import zarr
from connectome_snns.utils.reproducibility import load_experiment_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.evaluation import smooth

EVALUATION_PARAMETERS = HERE.parent / "fig01-full-reconstruction" / "parameters.toml"
VARIANCE_THRESHOLDS = (0.5, 0.8, 0.9, 0.95)


def main(teacher_dir, out_dir):
    evaluation = toml.load(EVALUATION_PARAMETERS)["evaluation"]
    sigma_ms = float(evaluation["fluctuation_tau_ms"])
    burn_in_ms = float(evaluation["burn_in_ms"])
    data = zarr.open_group(str(teacher_dir / "results" / "spike_data.zarr"), mode="r")
    spikes = data["output_spikes"]
    dt = float(data.attrs["dt"])
    burn_in = int(burn_in_ms / dt)
    n_trials, n_steps, n_neurons = spikes.shape

    # Pooled second moments, one trial at a time.
    total = torch.zeros(n_neurons, dtype=torch.float64)
    outer = torch.zeros(n_neurons, n_neurons, dtype=torch.float64)
    n_samples = 0
    started = time.time()
    for trial in range(n_trials):
        activity = torch.from_numpy(
            smooth(np.asarray(spikes[trial]), sigma_ms, dt)[burn_in:]
        ).double()
        total += activity.sum(dim=0)
        outer += activity.T @ activity
        n_samples += activity.shape[0]
        print(
            f"  trial {trial + 1}/{n_trials} ({time.time() - started:.0f} s)",
            flush=True,
        )
    mean = total / n_samples
    covariance = (outer - n_samples * torch.outer(mean, mean)) / (n_samples - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0).flip(0).numpy()

    fraction = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(fraction)
    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, eigenvalues.size + 1),
            "eigenvalue": eigenvalues,
            "variance_fraction": fraction,
            "cumulative_fraction": cumulative,
        }
    )
    summary = {
        "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum()),
        **{
            f"n_pcs_{round(100 * t)}pct_var": int(np.searchsorted(cumulative, t) + 1)
            for t in VARIANCE_THRESHOLDS
        },
        "smoothing_sigma_ms": sigma_ms,
        "burn_in_ms": burn_in_ms,
        "n_trials": int(n_trials),
        "seconds_per_trial": (n_steps - burn_in) * dt / 1000.0,
        "n_neurons": int(n_neurons),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    spectrum.to_csv(out_dir / "teacher_pca_spectrum.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "teacher_dimensionality.csv", index=False)
    print(pd.Series(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(load_experiment_config(HERE / "experiment.toml")["output_dir"]),
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.teacher, args.out)
