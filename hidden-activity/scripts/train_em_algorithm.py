"""
EM training with CLAMPED E-step and VISIBLE loss.

E-step: Visible neurons clamped to teacher spikes; only hidden neurons inferred.
M-step: Loss computed on visible neurons only.

This is for TRAINING: scaling factors start perturbed from target.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _em_core import run_em_training


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    return run_em_training(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        params_file=Path(params_file),
        e_step_type="clamped",
        loss_type="visible",
        wandb_config=wandb_config,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
