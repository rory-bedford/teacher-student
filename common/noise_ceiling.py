"""Poisson redraws of the held-out trial's mitral input, for the noise ceiling.

The held-out trial's mitral input is an inhomogeneous Poisson process driven by an OU
rate trajectory, and ``test_inputs.zarr`` stores those rates. The noise ceiling
(``common.evaluation``) runs the perfectly specified student with exactly the run's
teacher forcing (observed and unreconstructed teacher spikes) but with the mitral
spikes redrawn from the same rates, and scores it against the held-out teacher. It is
the best any student can do when the only thing it lacks is the trial's Poisson input
noise.
"""

import numpy as np
import zarr

N_RESAMPLES = 5
RESAMPLE_SEED = 1234


def mitral_redraws(test_inputs_path):
    """``N_RESAMPLES`` Poisson redraws of the mitral spikes: bool (K, time, n_mitral)."""
    test_inputs = zarr.open_group(str(test_inputs_path), mode="r")
    rates = np.asarray(test_inputs["rates"][0])  # (time, n_mitral), Hz
    probabilities = rates * float(test_inputs.attrs["dt"]) / 1000.0
    rng = np.random.default_rng(RESAMPLE_SEED)
    return rng.random((N_RESAMPLES, *rates.shape)) < probabilities[None]
