"""Figure 2 — evaluate the connectivity controls (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply the full-connectome variant):
    uv run python fig02-controls/analysis.py

Writes, next to this script:
    fig02_summary.csv   variant, n_free_params, seed, group, metric, value, floor_value, ceiling_value,
                        noise_ceiling_value
    fig02_rates.csv     variant, n_free_params, neuron_id, cell_type, observed, seed, rates, fluctuation_r2
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
VARIANT_NAMES = {
    "connectome": "full_connectome",
    "learnt": "learnt_recurrence",
    "shuffle_weights": "shuffle_weights",
    "configuration_model": "configuration_model",
}


def label(params, evaluation):
    return {
        "variant": VARIANT_NAMES[
            params["student"].get("recurrent_model", "connectome")
        ],
        "n_free_params": int(evaluation["n_free_params"]),
    }


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(baseline_dir) + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "fig02_summary.csv", index=False)
    rates.to_csv(out_dir / "fig02_rates.csv", index=False)
    print(
        summary.groupby(["variant", "group", "metric"])[
            ["value", "floor_value", "ceiling_value"]
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
