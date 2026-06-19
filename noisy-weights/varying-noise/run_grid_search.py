"""
Grid search over varying noise levels.

Tests network robustness by training with noise from 5% to 50% in 5% increments.

Run:
    python run_grid_search.py experiment.toml
"""

import sys
from copy import deepcopy
import numpy as np


from utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate configs for varying noise levels.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for noise in np.arange(0.05, 0.55, 0.05):
        params = deepcopy(base_params)
        params["weight_noise"]["noise_frac"] = float(noise)
        yield params, f"noise-{noise:.2f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
