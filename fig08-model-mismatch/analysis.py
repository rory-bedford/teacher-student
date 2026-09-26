"""Figure 8 — evaluate the threshold-heterogeneity sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply heterogeneity 0):
    uv run python fig08-model-mismatch/analysis.py

Writes, next to this script:
    fig08_summary.csv          threshold_heterogeneity, seed,
                               evaluation{held_out,perturbation}, group, cell_type,
                               n_cells, metric, value, ceiling_value
    fig08_rates.csv            threshold_heterogeneity, neuron_id, cell_type, observed,
                               seed, rates, fluctuation_r2
    fig08_scaling_factors.csv  threshold_heterogeneity, seed, scaling_factor, value,
                               target -- what the six trained parameters do under the
                               mismatch

The ceiling is the perfectly specified student -- the teacher's thresholds -- so it does
not depend on the mismatch and equals Figure 1's; figures.py draws it once, from Figure
1's rows.
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
    scaling_factor_rows,
)
from common.perturbation import collect_perturbation

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"


def mismatch(params):
    return {
        "threshold_heterogeneity": float(
            params["student"].get("threshold_heterogeneity", 0.0)
        )
    }


def label(params, evaluation):
    return mismatch(params)


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(baseline_dir) + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)
    delta_summary, _ = collect_perturbation(runs, label, device)
    summary = pd.concat([summary, pd.DataFrame(delta_summary)], ignore_index=True)
    factors = []
    for run in runs:
        params = run_parameters(run)
        factors += scaling_factor_rows(
            run, seed=int(params["simulation"]["seed"]), **mismatch(params)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig08_summary.csv", index=False)
    rates.to_csv(out_dir / "fig08_rates.csv", index=False)
    pd.DataFrame(factors).to_csv(out_dir / "fig08_scaling_factors.csv", index=False)
    print(
        summary.groupby(
            ["threshold_heterogeneity", "evaluation", "group", "cell_type", "metric"]
        )[["value", "ceiling_value"]].mean()
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
