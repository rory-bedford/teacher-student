"""Figure 6 — three seeds of one condition: learnt feedforward, full recurrent connectome.

There is no sweep. The seed draws the scaling-factor perturbation and the initialisation
of the learnt blocks; every run sees the whole recurrent connectome and every neuron.

Run:
    ./run --grid fig06-learnt-feedforward/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0]  # Edit with available GPU IDs
SEEDS = [46]  # 44 and 45 are on the cluster (slurm/20260923-133124); local run does 46


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
