"""Figure 8 — threshold heterogeneity: three levels x three seeds (9 runs).

Heterogeneity 0 is Figure 1 and is read from its runs by analysis.py. Runs are ordered
seed by seed, so every level has one seed before any has two.

Run:
    ./run --grid fig08-model-mismatch/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
#: SD (mV) of the per-neuron threshold offset. The teacher's thresholds sit 22 mV (E) and
#: 15 mV (I) above rest.
THRESHOLD_HETEROGENEITY = [1.0, 2.0, 4.0]


def custom_config_generator(base_params):
    for seed in SEEDS:
        for sd in THRESHOLD_HETEROGENEITY:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["threshold_heterogeneity"] = sd
            yield params, f"heterogeneity-{sd:g}mV__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
