"""Figure 5 — train a student with noisy connectome weights.

The student and training loop are shared by every figure (``common/``); what this
figure trains is set entirely by ``parameters.toml``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.training import train_student


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    return train_student(input_dir, output_dir, params_file, wandb_config, resume_from)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
