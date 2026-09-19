"""Figure 3 — three observed fractions x three seeds (9 runs).

The 50% point of the sweep is Figure 1 itself (analysis.py reads its runs), so it is not
run again here.

The 10% point is Figure 1 and is read from its runs by analysis.py. Runs are ordered
seed by seed (coarse curve first), and within a seed from the lowest fraction up, since
the break point is expected below 10%.

Run:
    ./run --grid fig03-observed-fraction/experiment.toml
"""

import sys
from copy import deepcopy
from pathlib import Path

from connectome_snns.utils.experiment_runners import run_custom_search

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid import skip_completed

CUDA_VISIBLE_DEVICES = [0, 1]  # Edit with available GPU IDs
SEEDS = [44, 45, 46]
#: 2% is the first unreliable point: below ~25% observed a run either trains or collapses (the
#: unobserved AND observed excitatory populations fall silent, whatever the rate-penalty
#: targets or learning rate -- see probes on 2026-09-18), so 0.5% and 1% are not reported.
#: The 2% point is kept deliberately, to show where the fit stops being reliable.
OBSERVED_FRACTIONS = [0.02, 0.05, 0.25]


def custom_config_generator(base_params):
    for seed in SEEDS:
        for fraction in OBSERVED_FRACTIONS:
            params = deepcopy(base_params)
            params["simulation"]["seed"] = seed
            params["student"]["observed_fraction"] = fraction
            yield params, f"obs-{fraction:g}__seed-{seed}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment.toml"
    run_custom_search(
        experiment_config_path=config_path,
        config_generator=skip_completed(config_path, custom_config_generator),
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
