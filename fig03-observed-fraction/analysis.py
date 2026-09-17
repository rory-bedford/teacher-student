"""Figure 3 — evaluate the observed-fraction sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply the 10% point):
    uv run python fig03-observed-fraction/analysis.py

Writes, next to this script:
    fig03_summary.csv          obs_fraction, n_observed, seed, group, metric, value, ceiling_value
    fig03_rates.csv            obs_fraction, neuron_id, cell_type, observed, seed, rates, fluctuation_r2

The teacher's participation ratio (marked on panel a) is not computed here: figures.py
reads generate-teacher-activity/teacher_dimensionality.csv (generate-teacher-activity/dimensionality.py).
"""

import argparse
import sys
from pathlib import Path

import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import (
    collect,
    completed_runs,
)

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"


def label(params, evaluation):
    return {
        "obs_fraction": float(params["student"]["observed_fraction"]),
        "n_observed": int(evaluation["n_observed"]),
    }


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_runs = completed_runs(baseline_dir)
    runs = baseline_runs + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig03_summary.csv", index=False)
    rates.to_csv(out_dir / "fig03_rates.csv", index=False)
    print(
        summary.groupby(["obs_fraction", "group", "metric"])[
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
