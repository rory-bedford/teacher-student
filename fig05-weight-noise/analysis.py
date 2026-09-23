"""Figure 5 — evaluate the weight-noise sweep (and Figure 1) on held-out stimuli.

Run after training (Figure 1's runs supply weight noise 0):
    uv run python fig05-weight-noise/analysis.py

Writes, next to this script:
    fig05_summary.csv   weight_noise, noise_clipped_fraction, seed, evaluation{held_out,perturbation},
                        group, cell_type, n_cells, metric, value, ceiling_value
    fig05_rates.csv     weight_noise, neuron_id, cell_type, observed, seed, rates, fluctuation_r2
    fig05_weight_perturbation.csv
                        weight_noise, original, noisy -- a sample of single synapses
                        before and after the perturbation, for the panel that shows what
                        the manipulation does to a weight. Needs no trained run: it
                        applies the same function training applies.

noise_clipped_fraction is the fraction of non-zero weights the archived noise pushed
below zero, and which were clipped to zero — the answer to the README's sign-flip question.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from connectome_snns.utils.reproducibility import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import collect, completed_runs
from common.perturbation import collect_perturbation
from common.structure import apply_weight_noise

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "fig01-full-reconstruction" / "experiment.toml"
TEACHER = HERE.parent / "fig00-teacher-activity" / "experiment.toml"


def label(params, evaluation):
    return {
        "weight_noise": float(params["student"].get("weight_noise", 0.0)),
        "noise_clipped_fraction": float(evaluation["noise_clipped_fraction"]),
    }


#: Noise levels shown side by side, and how many synapses are sampled for each.
#: 0.7 rather than 0.5 (2026-09-23): at 0.7 the teacher's weights and the student's
#: correlate at r = 0.81, which is the weight-against-synapse-volume correlation measured
#: in Holler et al., "Structure and function of a neocortical synapse" -- so the right-hand
#: panel shows the perturbation at the precision a real reconstruction achieves.
PERTURBATION_LEVELS = (0.1, 0.7)
PERTURBATION_SAMPLE = 3000
PERTURBATION_SEED = 44


def weight_perturbation(teacher_dir, out_dir):
    """A sample of recurrent synapses before and after the noise, per level.

    Applies :func:`common.structure.apply_weight_noise` exactly as the training runs do,
    per (presynaptic, postsynaptic) cell-type block, so the panel shows the manipulation
    the sweep actually performs rather than an illustration of it.
    """
    structure = np.load(Path(teacher_dir) / "results" / "network_structure.npz")
    weights = structure["recurrent_weights"]
    cell_types = structure["cell_type_indices"]
    n_types = int(cell_types.max()) + 1
    rows = []
    for noise in PERTURBATION_LEVELS:
        rng = np.random.default_rng(PERTURBATION_SEED)
        noisy = weights.copy()
        clipped = total = 0
        for source in range(n_types):
            for target in range(n_types):
                block = np.outer(cell_types == source, cell_types == target)
                block &= weights != 0
                if not block.any():
                    continue
                values = weights[block]
                perturbed, n_clipped = apply_weight_noise(values, noise, rng)
                noisy[block] = perturbed
                clipped += n_clipped
                total += values.size
        nonzero = np.flatnonzero(weights.ravel() != 0)
        sample = np.random.default_rng(PERTURBATION_SEED).choice(
            nonzero, size=min(PERTURBATION_SAMPLE, nonzero.size), replace=False
        )
        # The statistics are the whole non-zero population's, not the plotted sample's:
        # the rescale preserves each block's mean and SD exactly, and a few hundred
        # sampled synapses do not show that (their own SD moves with whichever outliers
        # the draw happens to contain).
        before, after = weights.ravel()[nonzero], noisy.ravel()[nonzero]
        rows.append(
            pd.DataFrame(
                {
                    "weight_noise": noise,
                    "original": weights.ravel()[sample],
                    "noisy": noisy.ravel()[sample],
                    "correlation": np.corrcoef(before, after)[0, 1],
                    "clipped_fraction": clipped / total,
                    "population_mean": before.mean(),
                    "population_mean_noisy": after.mean(),
                    "population_sd": before.std(),
                    "population_sd_noisy": after.std(),
                    "n_synapses": before.size,
                }
            )
        )
        print(
            f"  weight noise {noise:g}: r = {rows[-1]['correlation'].iloc[0]:.4f}, "
            f"{100 * clipped / total:.1f}% clipped to zero"
        )
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out_dir / "fig05_weight_perturbation.csv", index=False)


def main(runs_dir, baseline_dir, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = completed_runs(baseline_dir) + completed_runs(runs_dir)
    if not runs:
        raise SystemExit("No completed runs")
    summary, rates = collect(runs, label, device)
    delta_summary, _ = collect_perturbation(runs, label, device)
    summary = pd.concat([summary, pd.DataFrame(delta_summary)], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    weight_perturbation(
        load_experiment_config(TEACHER)["output_dir"],
        out_dir,
    )
    summary.to_csv(out_dir / "fig05_summary.csv", index=False)
    rates.to_csv(out_dir / "fig05_rates.csv", index=False)
    print(
        summary.groupby(["weight_noise", "evaluation", "group", "cell_type", "metric"])[
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
