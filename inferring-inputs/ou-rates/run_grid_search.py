"""
Grid search over recurrent smoothing tau.

Tests the effect of different smoothing time constants on training
with OU-reconstructed feedforward inputs.

Run:
    python run_grid_search.py experiment.toml
"""

import sys
from copy import deepcopy


from utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate configs for varying recurrent smoothing tau.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for tau in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        params = deepcopy(base_params)
        params["training"]["recurrent_smoothing_tau"] = tau
        yield params, f"tau-{tau:.0f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
