"""Figure 1 — three seeds of the fully reconstructed student.

The seed draws the scaling-factor perturbation and which neurons are observed.

Run:
    ./run --grid fig01-full-reconstruction/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]


def custom_config_generator(base_params):
    for seed in SEEDS:
        params = deepcopy(base_params)
        params["simulation"]["seed"] = seed
        yield params, f"seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
