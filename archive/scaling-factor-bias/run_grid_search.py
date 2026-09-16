"""
Grid search over the fraction of hidden activity.

Repeats the scaling-factor-bias experiment at each hidden fraction from 0.1 to
0.9. The student's weights are the teacher's throughout, so scaling factors of
1.0 are the true model at *every* point of the sweep — what varies is only how
much of the network's activity has to be inferred rather than observed.

The question is whether the learnt scaling factors drift below 1.0, and whether
that drift grows with the fraction hidden. Because 1.0 is known to be the
optimum here, any such drift is a property of training rather than of the loss.

Run:
    ./run --grid scaling-factor-bias/experiment-sweep.toml
"""

import sys
from copy import deepcopy

import numpy as np
from connectome_snns.utils.experiment_runners import run_custom_search

CUDA_VISIBLE_DEVICES = [0]  # Edit with available GPU IDs


def custom_config_generator(base_params):
    """Generate one config per hidden-activity fraction.

    Args:
        base_params: Loaded parameters TOML as dict

    Yields:
        (params_dict, description_string) tuples
    """
    for fraction in np.arange(0.1, 1.0, 0.1):
        params = deepcopy(base_params)
        params["simulation"]["hidden_cell_fraction"] = float(round(fraction, 1))
        yield params, f"hidden-frac-{fraction:.1f}"


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment-sweep.toml"
    results = run_custom_search(
        experiment_config_path=config_path,
        config_generator=custom_config_generator,
        cuda_devices=CUDA_VISIBLE_DEVICES,
        grid_script_path=__file__,
    )
