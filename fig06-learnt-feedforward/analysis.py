"""Figure 6 — evaluate the learnt-feedforward student against a known-input control.

Run after training:
    uv run python fig06-learnt-feedforward/analysis.py

Both conditions run at 50% observed, so each carries an observed and an unobserved group:

    known_input         Figure 1's runs, not retrained here: the same student with the
                        feedforward weights given.
    learnt_feedforward  this figure's runs: the recurrent connectome given, every mitral
                        weight learnt.

The figure's claim is the difference between the two groups within the learnt condition,
against Figure 1 where both groups are predicted equally well.

Writes, next to this script:
    fig06_summary.csv   variant, n_free_params, total_epochs, seed,
                        evaluation{held_out,perturbation}, group, cell_type, n_cells,
                        metric, value, ceiling_value
    fig06_rates.csv     variant, neuron_id, cell_type, observed, seed, rates, fluctuation_r2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import collect, completed_runs
from common.perturbation import collect_perturbation

HERE = Path(__file__).resolve().parent
#: The known-input control is Figure 1's seed-matched runs at the same 50% observed, which
#: are not retrained here. Both conditions must sit at the same observation level or the
#: observed/unobserved comparison is not like for like.
CONTROL = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"


def label(params, evaluation):
    return {
        "variant": (
            "learnt_feedforward"
            if params["student"].get("learnt_feedforward", False)
            else "known_input"
        ),
        "total_epochs": int(params["training"]["total_epochs"]),
        "n_free_params": int(evaluation["n_free_params"]),
    }


def main(runs_dir, control_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(runs_dir) + completed_runs(control_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)
    delta_summary, _ = collect_perturbation(runs, label, device)
    summary = pd.concat([summary, pd.DataFrame(delta_summary)], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig06_summary.csv", index=False)
    rates.to_csv(out_dir / "fig06_rates.csv", index=False)
    print(
        summary.groupby(["variant", "evaluation", "group", "cell_type", "metric"])[
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
        "--control", type=Path, default=load_experiment_config(CONTROL)["output_dir"]
    )
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.runs, args.control, args.out)
