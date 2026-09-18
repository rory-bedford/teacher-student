"""Figure 2 — the connectivity controls at two observation levels (21 runs).

At the figure's observed fraction (50%) the three controls run against Figure 3's obs-0.5
runs as the full-connectome baseline. A second set repeats all four variants, baseline
included, with *every* modelled neuron observed: the regime of the archived controls
figure, where each neuron's recurrent input is the teacher's own activity, so a wrong
connectome degrades gracefully instead of compounding in a closed loop.
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
VARIANTS = ["learnt", "shuffle_inputs", "configuration_model"]
#: The fully observed comparison needs its own baseline, so it includes "connectome".
FULLY_OBSERVED_VARIANTS = ["connectome"] + VARIANTS


def custom_config_generator(base_params):
    for seed in SEEDS:
        for variant in VARIANTS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["recurrent_model"] = variant
            yield params, f"{variant}__seed-{seed}"
        for variant in FULLY_OBSERVED_VARIANTS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["recurrent_model"] = variant
            params["student"]["observed_fraction"] = 1.0
            yield params, f"{variant}-fully-observed__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
