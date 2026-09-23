"""Figure 5 — five weight-noise levels x three seeds (15 runs).

Weight noise 0 is Figure 1 and is read from its runs by analysis.py. Runs are ordered
seed by seed. To extend past 0.5,
append to NOISE_LEVELS.

Run:
    ./run --grid fig05-weight-noise/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
#: 0.6 and 0.7 added on 2026-09-23 so the sweep reaches the precision of a real
#: reconstruction: r = 0.81 between teacher and student weights at 0.7, against the
#: measured weight-volume correlation of Holler et al.
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def custom_config_generator(base_params):
    for seed in SEEDS:
        for noise in NOISE_LEVELS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["weight_noise"] = noise
            yield params, f"wn-{noise:g}__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
