"""
Grid search over weight-noise level for the scaling-factor bias check.

Sweeps ``student_mismatch.noise_frac`` with ``missing_unit_fraction`` held at 0,
so per-synapse weight noise is the only source of mismatch the scaling factors
cannot undo. ``noise_frac = 0`` is the reference point: there the target scaling
factors are exactly the true model, so any shrinkage of the learnt values below
them is a training bias rather than compensation.

Analysed by ``full-inference/bias-check`` — point its ``training_runs`` input at
this sweep's output directory and it evaluates every run.

Run:
    ./run --grid full-inference/no-hidden-units/experiment-noise-sweep.toml
"""

import sys
from copy import deepcopy

from connectome_snns.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0]  # Edit with available GPU IDs

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]


def custom_config_generator(base_params):
    """Generate configs for varying weight-noise levels.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for noise in NOISE_LEVELS:
        params = deepcopy(base_params)
        params["student_mismatch"]["noise_frac"] = float(noise)
        params["student_mismatch"]["missing_unit_fraction"] = 0.0
        yield params, f"noise-{noise:.2f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment-noise-sweep.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
