"""Figure 2 — the three connectivity controls x three seeds (9 runs).

The full-connectome variant is Figure 1 and is read from its runs by analysis.py.
Runs are ordered seed by seed, so every control has one seed before any has two.

Run:
    ./run --grid fig02-controls/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
VARIANTS = ["learnt", "shuffle_weights", "configuration_model"]


def custom_config_generator(base_params):
    for seed in SEEDS:
        for variant in VARIANTS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["recurrent_model"] = variant
            yield params, f"{variant}__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
