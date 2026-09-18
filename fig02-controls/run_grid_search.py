"""Figure 2 — the connectivity controls at the figure's observed fraction (9 runs).

The three controls run at 50% observed against Figure 1's seed-matched runs as the
full-connectome baseline, which is not retrained here. Runs are ordered seed by seed, so
every control has one seed before any has two.

A fully observed set (all four variants, baseline included, observed_fraction = 1.0) was
dropped on 2026-09-18: it appears in no panel. The runs already finished for it are left
in place and show up as non-grid folders in STATUS.md.

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
