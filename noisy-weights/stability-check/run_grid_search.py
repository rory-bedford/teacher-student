"""
Stability check with weight noise (0.3) but no SF perturbation, varying seeds.

Tests whether weight noise alone causes instability when scaling factors start
at the correct values. Multiple seeds to check consistency.

Run:
    python run_grid_search.py experiment.toml
"""

import sys
from copy import deepcopy


from connectome_snns.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate configs for different random seeds (noise=0.3, no SF perturbation).

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for seed in range(42, 52):
        params = deepcopy(base_params)
        params["simulation"]["seed"] = seed
        params["weight_noise"]["noise_frac"] = 0.3
        yield params, f"seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
