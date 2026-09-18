"""Figure 4 — evaluate neuron removal and synapse dropout (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply level 0 of both error models):
    uv run python fig04-reconstruction-errors/analysis.py

kappa_lost(i) = sum of teacher recurrent weights onto neuron i that the student lost,
divided by the sum of all teacher recurrent weights onto i. Computed from the
connectivity and the removal mask alone; the feedforward input, which is always
reconstructed, is not in the denominator.

Writes, next to this script:
    fig04_summary.csv      error_model, level, mean_kappa_lost, seed,
                           evaluation{held_out,perturbation}, group, cell_type, n_cells, metric,
                           value, ceiling_value
    fig04_per_neuron.csv   error_model, level, seed, neuron_id, cell_type, observed, kappa_lost,
                           fluctuation_r2, teacher_rate_hz, student_rate_hz
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
    rate_rows,
    run_parameters,
    summary_rows,
)
from common.perturbation import (
    evaluate_perturbation,
    perturbation_summary_rows,
    unavailable_reason,
)

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"
ERROR_MODELS = {
    "neuron_removal": "neuron_removal_fraction",
    "synapse_dropout": "synapse_dropout_fraction",
}


def conditions(params):
    """(error_model, level) pairs a run belongs to; the baseline belongs to both models."""
    student = params["student"]
    active = [
        (model, float(student[key]))
        for model, key in ERROR_MODELS.items()
        if student.get(key, 0.0) > 0
    ]
    if len(active) > 1:
        raise ValueError("a run may apply only one error model")
    return active or [(model, 0.0) for model in ERROR_MODELS]


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(baseline_dir) + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")

    summary, per_neuron = [], []
    for run in runs:
        print(f"Evaluating {run}")
        evaluation = evaluate_run(run, device)
        kappa = np.nan_to_num(evaluation["kappa"], nan=0.0)
        modelled = np.concatenate(
            [evaluation["observed_ids"], evaluation["unobserved_ids"]]
        )
        reason = unavailable_reason(run)
        if reason is not None:
            print(f"  no perturbation: {reason}")
        perturbation = None if reason else evaluate_perturbation(run, device)
        for model, level in conditions(run_parameters(run)):
            labels = {
                "error_model": model,
                "level": level,
                "mean_kappa_lost": float(kappa[modelled].mean()),
            }
            summary += summary_rows(evaluation, **labels)
            if perturbation is not None:
                summary += perturbation_summary_rows(perturbation, **labels)
            for row in rate_rows(evaluation, error_model=model, level=level):
                row["kappa_lost"] = float(kappa[row["neuron_id"]])
                per_neuron.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "fig04_summary.csv", index=False)
    columns = [
        "error_model",
        "level",
        "seed",
        "neuron_id",
        "cell_type",
        "observed",
        "kappa_lost",
        "fluctuation_r2",
        "teacher_rate_hz",
        "student_rate_hz",
    ]
    pd.DataFrame(per_neuron)[columns].to_csv(
        out_dir / "fig04_per_neuron.csv", index=False
    )
    print(
        summary.groupby(
            ["error_model", "level", "evaluation", "group", "cell_type", "metric"]
        )[["mean_kappa_lost", "value", "ceiling_value"]].mean()
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
