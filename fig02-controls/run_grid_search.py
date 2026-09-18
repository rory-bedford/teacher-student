"""Figure 2 — the connectivity controls, plus the fully observed check (12 runs).

The three controls run at 50% observed against Figure 1's runs as the full-connectome
baseline, which is not retrained here. Runs are ordered seed by seed, so
every control has one seed before any has two.

The fully observed set of all four variants was dropped on 2026-09-18 (it appears in no
panel); only the full-connectome one is kept, for backup-figures/. Its already-finished
siblings are left in place and show up as non-grid folders in STATUS.md.

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
#: The fully observed full-connectome run is kept (one per seed) for the backup panel in
#: backup-figures/: with every neuron teacher-forced there are no unobserved neurons, so
#: the rate penalties do not exist as loss terms and the six scaling factors are
#: recovered exactly. It is the identifiability check, not a bar in Figure 2.
FULLY_OBSERVED_VARIANTS = ["connectome"]


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
