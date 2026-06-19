"""
Convergence check with multiple random seeds at fixed noise.

Tests whether the network reliably learns from different random weight perturbations.
Fixed noise level (0.4), varying seeds from 42 to 51.

Run:
    python run_grid_search.py experiment.toml
"""

import sys
from copy import deepcopy


from utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate configs for different random seeds.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    # Fixed noise level, varying seeds
    noise_frac = 0.4

    for seed in range(42, 47):
        params = deepcopy(base_params)
        params["simulation"]["seed"] = seed
        params["weight_noise"]["noise_frac"] = noise_frac
        yield params, f"seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
