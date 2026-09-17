"""Figure 5 — evaluate the weight-noise sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply weight noise 0):
    uv run python fig05-weight-noise/analysis.py

Writes, next to this script:
    fig05_summary.csv   weight_noise, noise_clipped_fraction, seed, group, metric, value, ceiling_value
    fig05_rates.csv     weight_noise, neuron_id, cell_type, observed, seed, rates, fluctuation_r2

noise_clipped_fraction is the fraction of non-zero weights the archived noise pushed
below zero, and which were clipped to zero — the answer to the README's sign-flip question.
"""

import argparse
import sys
from pathlib import Path

import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import collect, completed_runs

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"


def label(params, evaluation):
    return {
        "weight_noise": float(params["student"].get("weight_noise", 0.0)),
        "noise_clipped_fraction": float(evaluation["noise_clipped_fraction"]),
    }


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(baseline_dir) + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig05_summary.csv", index=False)
    rates.to_csv(out_dir / "fig05_rates.csv", index=False)
    print(
        summary.groupby(["weight_noise", "group", "metric"])[
            ["value", "ceiling_value", "noise_clipped_fraction"]
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
