"""Figure 4 — two error models x four levels x three seeds (24 runs).

Level 0 is Figure 1 and is read from its runs by analysis.py. Synapse dropout (the new
arm) comes before neuron removal, and runs are ordered seed by seed.

Run:
    ./run --grid fig04-reconstruction-errors/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
ERROR_MODELS = {
    "synapse_dropout": "synapse_dropout_fraction",
    "neuron_removal": "neuron_removal_fraction",
}


def custom_config_generator(base_params):
    for seed in SEEDS:
        for model, key in ERROR_MODELS.items():
            for level in LEVELS:
                params = deepcopy(base_params)
                params["simulation"]["seed"] = seed
                params["student"][key] = level
                yield params, f"{model}-{level:g}__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
