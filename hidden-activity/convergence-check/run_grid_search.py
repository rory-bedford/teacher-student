"""
Grid search over hidden cell fraction.

Sweeps hidden_cell_fraction from 0.1 to 0.9 in increments of 0.1.

Run:
    ./run --grid experiments/teacher-student/hidden-activity/increasing-hidden-fraction/clamped-visible-grid-search.toml
"""

import sys
from copy import deepcopy


from connectome_snns.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [1]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate configs for varying hidden cell fractions.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for seed in range(44, 47):
        params = deepcopy(base_params)
        params["simulation"]["seed"] = int(seed)
        yield params, f"seed-{seed:.1f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
