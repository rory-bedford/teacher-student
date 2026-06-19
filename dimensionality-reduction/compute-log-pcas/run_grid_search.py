"""
Grid search over Gaussian smoothing kernel width for log-PCA.

Tests: 50, 100, 200, 500, 1000, 2000 ms smoothing.

Run:
    python run_grid_search.py experiment.toml
"""

import sys
from copy import deepcopy


from connectome_snns.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0]


def custom_config_generator(base_params):
    """Generate configs for varying Gaussian smoothing sigma.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for sigma_ms in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        params = deepcopy(base_params)
        params["pca"]["gaussian_sigma_ms"] = sigma_ms
        yield params, f"sigma-{sigma_ms:.0f}ms"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
