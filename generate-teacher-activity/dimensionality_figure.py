"""Teacher dimensionality figure from the CSVs written by dimensionality.py.

    uv run python generate-teacher-activity/dimensionality_figure.py

(a) Cumulative variance explained against PCA component (log x), with the participation
ratio and the 90%-variance component count marked. (b) Variance fraction per component
(log-log).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from connectome_snns.visualization import FIGURE_BLUE, FIGURE_CORAL, FLOOR_COLOR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.plotting import FIGURE_WIDTH, panel_label, use_talk_style


def main(data_dir, out_path):
    use_talk_style()
    spectrum = pd.read_csv(data_dir / "teacher_pca_spectrum.csv")
    summary = pd.read_csv(data_dir / "teacher_dimensionality.csv").iloc[0]
    pr = summary["participation_ratio"]
    n90 = int(summary["n_pcs_90pct_var"])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, 5.5 / 2.54), layout="constrained"
    )
    left.plot(spectrum["component"], spectrum["cumulative_fraction"], color=FIGURE_BLUE)
    left.axvline(pr, color=FIGURE_CORAL, linewidth=0.8, label=f"PR = {pr:.1f}")
    left.axhline(0.9, color=FLOOR_COLOR, linestyle="--", linewidth=0.8)
    left.axvline(
        n90, color=FLOOR_COLOR, linestyle="--", linewidth=0.8, label=f"90% at {n90} PCs"
    )
    left.set_xscale("log")
    left.set_xlabel("PCA component")
    left.set_ylabel("Cumulative variance explained")
    left.set_ylim(0, 1.02)
    left.legend(frameon=False, loc="upper left")
    panel_label(left, "a")

    right.loglog(
        spectrum["component"], spectrum["variance_fraction"], color=FIGURE_BLUE
    )
    right.axvline(pr, color=FIGURE_CORAL, linewidth=0.8)
    right.set_xlabel("PCA component")
    right.set_ylabel("Variance fraction")
    # Below ~1e-6 the spectrum is numerical noise of the eigensolver.
    right.set_ylim(1e-6, 1)
    panel_label(right, "b")

    fig.suptitle(
        f"Teacher dimensionality ({int(summary['n_neurons'])} neurons, "
        f"{int(summary['n_trials'])} trials x {summary['seconds_per_trial']:.0f} s, "
        f"Gaussian σ = {summary['smoothing_sigma_ms']:.0f} ms)"
    )
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--out", type=Path, default=HERE / "teacher_dimensionality.svg")
    args = parser.parse_args()
    main(args.data, args.out)
