"""``parameters.toml``, validated: one class per table, and an unknown key is an error.

A misspelt key fails when the file is loaded, naming the key, instead of being dropped
while its default stays in force. Optional entries default to off (0, or the
unmanipulated student), so a new loss term or manipulation is one new field with a
default and every existing file still loads unchanged.

The physiology tables (``[recurrent]``, ``[feedforward]``) and the teacher's
``[simulation]`` and odours use the library's configs.
"""

from pathlib import Path
from typing import Literal

import toml
from connectome_snns.configs import SimulationConfig
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.configs.odours import OdourInputConfig
from pydantic import BaseModel, ConfigDict, model_validator

from common.structure import RECURRENT_MODELS


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")


# =====================================================================
# Student runs (fig01-fig08)
# =====================================================================


class Simulation(Table):
    seed: int  # overridden per run by run_grid_search.py
    chunk_size: int


class Student(Table):
    """The manipulation: what the student knows of the teacher (see common/structure.py)."""

    observed_fraction: float | None = None  # of modelled neurons
    recurrent_model: Literal[RECURRENT_MODELS] = "connectome"
    neuron_removal_fraction: float = 0.0
    synapse_dropout_fraction: float = 0.0
    weight_noise: float = 0.0
    threshold_heterogeneity: float = 0.0  # SD of the threshold offsets (mV)
    learnt_feedforward: bool = False
    # Figure 7: a random segment of the pooled units is reconstructed.
    reconstructed_fraction: float | None = None
    recorded_pool_fraction: float | None = None


class Training(Table):
    total_epochs: int
    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    plot_size: int
    burn_in_chunks: int
    weight_perturbation_variance: (
        float  # log-normal perturbation of the scaling factors
    )
    mixed_precision: bool


class Optimiser(Table):
    surrgrad_scale: float
    lr_scaling: float
    lr_min_scaling: float
    grad_clip_scaling: float
    beta1: float
    beta2: float
    eps: float
    # Learnt weights (only runs that learn some): their own learning rate and clip.
    lr_weights: float | None = None
    lr_min_weights: float | None = None
    grad_clip_weights: float | None = None
    low_rank: int = 1


class Loss(Table):
    """Loss weights (0 = off) and the van Rossum kernel."""

    van_rossum: float
    van_rossum_tau_rise: float
    van_rossum_tau_decay: float
    hidden_rate_mean: float = 0.0  # unobserved population mean rate vs target, per type
    hidden_rate_std: float = 0.0  # unobserved population rate SD vs target, per type
    rate_target: Literal["observed", "population"] = "observed"
    learnt_weight_l1: float = 0.0


class Evaluation(Table):
    burn_in_ms: float  # discarded from the held-out trial before scoring
    fluctuation_tau_ms: float  # Gaussian sigma for Fluctuation R²

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_keys(cls, data):
        # The shuffled-identity floor, dropped from evaluation (859adb5); early runs
        # still carry its permutation count.
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "n_floor_permutations"}
        return data


class StudentParameters(Table):
    simulation: Simulation
    student: Student
    training: Training
    optimiser: Optimiser
    loss: Loss
    evaluation: Evaluation
    feedforward: FeedforwardLayerConfig
    recurrent: RecurrentLayerConfig


def load_student_parameters(path) -> StudentParameters:
    return StudentParameters.model_validate(toml.load(Path(path)))


# =====================================================================
# Teacher (fig00)
# =====================================================================


class Odours(BaseModel):
    """The shared OU modulation, and one table per odour."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, OdourInputConfig]

    tau: float
    temperature: float
    sigma: float

    @property
    def inputs(self) -> dict[str, OdourInputConfig]:
        return dict(self.__pydantic_extra__)


class TeacherParameters(Table):
    simulation: SimulationConfig
    odours: Odours
    feedforward: FeedforwardLayerConfig
    recurrent: RecurrentLayerConfig


def load_teacher_parameters(path) -> TeacherParameters:
    return TeacherParameters.model_validate(toml.load(Path(path)))
