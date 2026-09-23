"""Figure 6 — ten reconstructed fractions x three seeds, plus one fully observed run (31 runs).

Runs are ordered seed by seed, from the 10% endpoint up. The fully observed control
(every modelled neuron in the loss, 10% reconstructed) follows the first seed's sweep.

Run:
    ./run --grid fig07-partial-reconstruction/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
#: Even steps of 0.1 (2026-09-21, was [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]): the collapse is at
#: the TOP of this sweep -- held-out R² falls 0.97 -> 0.42 between full reconstruction and
#: 70%, then is flat and negative below 30% -- so the range from 0.7 to 1.0 needed
#: resolving, not the bottom end.
RECONSTRUCTED_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FULLY_OBSERVED = {"reconstructed_fraction": 0.1, "seed": 44}


def custom_config_generator(base_params):
    for seed in SEEDS:
        for fraction in RECONSTRUCTED_FRACTIONS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["reconstructed_fraction"] = fraction
            yield params, f"recon-{fraction:g}__seed-{seed}"
        if seed == FULLY_OBSERVED["seed"]:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["reconstructed_fraction"] = FULLY_OBSERVED[
                "reconstructed_fraction"
            ]
            params["student"]["recorded_pool_fraction"] = 1.0
            yield (
                params,
                f"recon-{FULLY_OBSERVED['reconstructed_fraction']:g}-fully-observed__seed-{seed}",
            )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
