"""
Full-inference: hidden-units variant.

A fraction of recurrent neurons are unobserved — only the visible
neurons' spikes are available as a training target. The student learns:

* the full feedforward weight matrix (low-rank ``U @ V``, shared across
  the hidden and visible layers for the mitral block), and
* per-cell-type scaling factors on the recurrent connections.

The architecture is a two-layer SNN: a recurrent hidden layer feeding
into a visible layer, so gradients from the visible loss flow back
through the visible→hidden feedforward path to the FF weights.

Optional perturbations applied to the student before training:

* ``hidden_fraction``       — fraction of neurons made unobservable.
* ``noise_frac``            — multiplicative log-normal noise on the
                              recurrent weights (per cell-type pair).
* ``missing_unit_fraction`` — a fraction of recurrent neurons are
                              structurally removed from the student.
* ``input_type``            — feedforward signal: actual mitral spikes,
                              homogeneous Poisson, or latent OU rates.
"""

import math
from pathlib import Path

import numpy as np
import toml
import torch
import torch.nn as nn
import wandb
import zarr
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from connectome_snns.configs import (
    DATALOADER_KWARGS,
    StudentHyperparameters,
    StudentSimulationConfig,
)
from connectome_snns.configs.conductance_based import FeedforwardLayerConfig, RecurrentLayerConfig
from connectome_snns.dataloaders.supervised import (
    CyclicSampler,
    ExactFFDataset,
    HomogeneousPoissonFFDataset,
    VisibleDrivenCollate,
)
from connectome_snns.dataloaders.supervised.latent_ou import LatentOUDataset
from connectome_snns.network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    Projection,
    ScalingFactorProjection,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN
from connectome_snns.snn_runners import SNNTrainer
from connectome_snns.training_utils import AsyncLogger
from connectome_snns.training_utils.losses import VanRossumLoss
from connectome_snns.visualization import plot_spike_trains


# =====================================================================
# Helper functions
# =====================================================================


def mask_hidden_units(
    weights,
    feedforward_weights,
    recurrent_mask,
    feedforward_mask,
    cell_type_indices,
    hidden_unit_fraction,
    rng,
):
    """Remove a fraction of recurrent neurons from the student's model."""
    n_neurons = weights.shape[0]
    n_hidden = max(1, int(n_neurons * hidden_unit_fraction))

    all_indices = np.arange(n_neurons)
    rng.shuffle(all_indices)
    hidden_indices = np.sort(all_indices[:n_hidden])
    visible_indices = np.sort(all_indices[n_hidden:])

    vis_weights = weights[visible_indices][:, visible_indices]
    vis_ff_weights = feedforward_weights[:, visible_indices]
    vis_rec_mask = recurrent_mask[visible_indices][:, visible_indices]
    vis_ff_mask = feedforward_mask[:, visible_indices]
    vis_cell_type_indices = cell_type_indices[visible_indices]

    return (
        vis_weights,
        vis_ff_weights,
        vis_rec_mask,
        vis_ff_mask,
        vis_cell_type_indices,
        visible_indices,
        hidden_indices,
    )


def apply_weight_noise(weights, noise_frac, rng=None, preserve_statistics=True):
    """Apply multiplicative log-normal noise to weights with statistics preservation."""
    if rng is None:
        rng = np.random.default_rng()

    nonzero_mask = weights != 0
    orig_mean = weights[nonzero_mask].mean()
    orig_std = weights[nonzero_mask].std()

    multiplier = np.exp(
        noise_frac * rng.standard_normal(weights.shape) - noise_frac**2 / 2
    )
    noisy_weights = weights * multiplier

    if preserve_statistics and orig_std > 0:
        noisy_nz = noisy_weights[nonzero_mask]
        noisy_mean = noisy_nz.mean()
        noisy_std = noisy_nz.std()

        if noisy_std > 0:
            noisy_weights[nonzero_mask] = orig_mean + (noisy_nz - noisy_mean) * (
                orig_std / noisy_std
            )
            noisy_weights[nonzero_mask] = np.maximum(noisy_weights[nonzero_mask], 0)

    return noisy_weights


# =====================================================================
# Shared-sliced low-rank FF projection (experiment-local)
# =====================================================================
#
# Per-output-cell-type rank-``ff_rank`` factorisation of the mitral FF
# block. Each output cell type gets its own ``(U_ct, V_ct)`` Parameter
# pair shared across layers; each layer instantiates a
# ``SharedSlicedLowRankProjection`` that holds references to those same
# Parameters and a layer-specific column-index buffer selecting the
# layer's subset of target neurons (unobserved vs visible). Gradients
# from both layers accumulate into the same ``U_ct`` / ``V_ct``.


class SharedSlicedLowRankProjection(Projection):
    """LowRank projection with column-sliced shared parameters.

    Holds references to externally constructed ``U`` (shape
    ``(n_source, rank)``) and ``V_full`` (shape ``(rank, n_total)``)
    Parameters. ``forward()`` returns ``exp(U @ V_full[:, cols]) * mask``
    where ``cols`` selects this layer's target columns.

    The same ``U`` / ``V_full`` Parameter objects are passed to multiple
    instances (one per layer) so the optimiser sees a single tensor and
    gradients from all layers accumulate into it naturally. The
    Parameters are stored as plain attributes via ``object.__setattr__``
    so ``nn.Module``'s registration hook does not re-register them on
    each instance (which would make ``model.parameters()`` double-count).
    """

    def __init__(
        self,
        U: nn.Parameter,
        V_full: nn.Parameter,
        cols,
        mask=None,
    ):
        n_source = int(U.shape[0])
        n_target = int(cols.shape[0])
        super().__init__(n_source, n_target)
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "V_full", V_full)
        self.register_buffer("cols", cols.long())
        if mask is None:
            m = torch.ones(n_source, n_target)
        else:
            m = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
            m = m.float()
        if tuple(m.shape) != (n_source, n_target):
            raise ValueError(f"mask shape {tuple(m.shape)} != ({n_source}, {n_target})")
        self.register_buffer("mask", m)

    @property
    def caching_mode(self) -> str:
        return "weights"

    def forward(self) -> torch.Tensor:
        V_slice = self.V_full[:, self.cols]
        return torch.exp(self.U @ V_slice) * self.mask


# =====================================================================
# Auxiliary losses (experiment-local)
# =====================================================================


class HiddenRateMeanPenalty(nn.Module):
    """MSE between student hidden population MEAN rate and per-cell-type target.

    Target rates come from the *visible* teacher population (visible and
    hidden are drawn from the same cell-type distribution, so visible
    statistics are unbiased estimators of hidden ones).
    """

    required_inputs: list = ["hidden_spikes"]

    def __init__(
        self,
        model,
        target_rates_hz: torch.Tensor,
        dt_ms: float,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("target_rates_hz", target_rates_hz)
        self.dt_ms = dt_ms

    def forward(self, hidden_spikes: torch.Tensor) -> torch.Tensor:
        batch, n_t, _ = hidden_spikes.shape
        duration_s = n_t * self.dt_ms / 1000.0
        ct_h = self.model.layer1.cell_type_indices
        loss = hidden_spikes.new_zeros(())
        for ct in torch.unique(ct_h):
            mask = ct_h == ct
            n_ct = mask.sum().clamp(min=1).float()
            total_spikes = hidden_spikes[:, :, mask].sum()
            mean_rate = total_spikes / (n_ct * duration_s * batch)
            target = self.target_rates_hz[int(ct)]
            loss = loss + (mean_rate - target).pow(2)
        return loss


class HiddenRateStdPenalty(nn.Module):
    """MSE between student hidden population rate STD and per-cell-type target.

    Keeps the rate distribution's spread pinned to the teacher's without
    touching individual neurons (which would collapse the distribution).
    Separate from the mean penalty because mean and std live on different
    scales and deserve independent loss weights.
    """

    required_inputs: list = ["hidden_spikes"]

    def __init__(
        self,
        model,
        target_rate_stds_hz: torch.Tensor,
        dt_ms: float,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("target_rate_stds_hz", target_rate_stds_hz)
        self.dt_ms = dt_ms

    def forward(self, hidden_spikes: torch.Tensor) -> torch.Tensor:
        batch, n_t, _ = hidden_spikes.shape
        duration_s = n_t * self.dt_ms / 1000.0
        # Per-neuron rate (averaged over time and batch)
        per_neuron_rate = hidden_spikes.sum(dim=(0, 1)) / (duration_s * batch)
        ct_h = self.model.layer1.cell_type_indices
        loss = hidden_spikes.new_zeros(())
        for ct in torch.unique(ct_h):
            mask = ct_h == ct
            if mask.sum() < 2:
                continue
            std_rate = per_neuron_rate[mask].std(unbiased=False)
            target_std = self.target_rate_stds_hz[int(ct)]
            loss = loss + (std_rate - target_std).pow(2)
        return loss


class FFMatrixL1Penalty(nn.Module):
    """L1 penalty on one cell type's reconstructed linear-space FF weights.

    Computes ``mean(exp(U_ct @ V_ct))`` for a single cell type. Register
    one instance per type to get independent loss weights.
    """

    required_inputs: list = []

    def __init__(self, U_ct: torch.Tensor, V_ct: torch.Tensor):
        super().__init__()
        # Store as plain attrs (not nn.Parameters) so they stay tied to the
        # originals and don't get registered as duplicate params on the loss.
        object.__setattr__(self, "_U", U_ct)
        object.__setattr__(self, "_V", V_ct)

    def forward(self) -> torch.Tensor:
        return torch.exp(self._U @ self._V).mean()


def build_losses(
    model,
    *,
    van_rossum_kwargs: dict,
    loss_weights_cfg,
    hidden_rate_targets: torch.Tensor,
    hidden_rate_std_targets: torch.Tensor,
    dt_ms: float,
    U_per_type: list[torch.Tensor] | None = None,
    V_per_type: list[torch.Tensor] | None = None,
    cell_type_names: list[str] | None = None,
):
    """Build the (loss_functions, loss_weights) dict pair for an SNNTrainer."""
    losses = {"van_rossum": VanRossumLoss(**van_rossum_kwargs)}
    weights = {"van_rossum": loss_weights_cfg.van_rossum}

    if loss_weights_cfg.hidden_rate_mean > 0:
        losses["hidden_rate_mean"] = HiddenRateMeanPenalty(
            model, target_rates_hz=hidden_rate_targets, dt_ms=dt_ms
        )
        weights["hidden_rate_mean"] = loss_weights_cfg.hidden_rate_mean

    if loss_weights_cfg.hidden_rate_std > 0:
        losses["hidden_rate_std"] = HiddenRateStdPenalty(
            model, target_rate_stds_hz=hidden_rate_std_targets, dt_ms=dt_ms
        )
        weights["hidden_rate_std"] = loss_weights_cfg.hidden_rate_std

    # Per-cell-type L1 penalties on FF weights.
    # Uses ff_l1_{type_name} keys if present, falls back to ff_matrix_l1 for all.
    if (
        U_per_type is not None
        and V_per_type is not None
        and cell_type_names is not None
    ):
        for ct_idx, ct_name in enumerate(cell_type_names):
            per_type_key = f"ff_l1_{ct_name}"
            w = getattr(loss_weights_cfg, per_type_key, 0.0)
            if w == 0.0:
                w = getattr(loss_weights_cfg, "ff_matrix_l1", 0.0)
            if w > 0:
                losses[per_type_key] = FFMatrixL1Penalty(
                    U_per_type[ct_idx], V_per_type[ct_idx]
                )
                weights[per_type_key] = w

    return losses, weights


# =====================================================================
# Main
# =====================================================================


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    print("\n" + "=" * 60)
    print("Full Inference Training (Two-Layer)")
    print("=" * 60 + "\n")

    # ======================================
    # Phase 0: Config & Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors_cfg = data.get("scaling_factors", {})

    training_cfg = data["training"]
    optimiser_cfg = data["optimiser"]
    mismatch_cfg = data.get("student_mismatch", {})

    # Mismatch parameters
    hidden_fraction = mismatch_cfg.get("hidden_fraction", 0.0)
    missing_unit_fraction = mismatch_cfg.get("missing_unit_fraction", 0.0)
    noise_frac = mismatch_cfg.get("noise_frac", 0.0)
    input_type = mismatch_cfg.get("input_type", "constant")
    firing_rate_hz = mismatch_cfg.get("firing_rate_hz", 6.0)
    ff_init = mismatch_cfg.get("ff_init", "constant")
    if ff_init not in ("constant", "teacher"):
        raise ValueError(f"ff_init must be 'constant' or 'teacher', got {ff_init!r}")

    # Training parameters
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    epochs = training_cfg["epochs"]
    chunks_per_update = training_cfg["chunks_per_update"]
    log_interval = training_cfg["log_interval"]
    checkpoint_interval = training_cfg["checkpoint_interval"]
    plot_size = training_cfg["plot_size"]
    mixed_precision = training_cfg["mixed_precision"]
    burn_in_chunks = training_cfg.get("burn_in_chunks", 0)
    weight_perturbation_variance = training_cfg["weight_perturbation_variance"]
    ff_rank = training_cfg["ff_rank"]
    ff_smoothing_tau = training_cfg.get("ff_smoothing_tau", None)

    # Optimiser parameters (single grad-clip across all param groups)
    lr_weights = optimiser_cfg["lr_weights"]
    lr_scaling = optimiser_cfg["lr_scaling"]
    lr_min_weights = optimiser_cfg["lr_min_weights"]
    lr_min_scaling = optimiser_cfg["lr_min_scaling"]
    grad_clip = optimiser_cfg["grad_clip"]
    surrgrad_scale = optimiser_cfg["surrgrad_scale"]

    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
    adam_eps = getattr(hyperparameters, "eps", 1e-8)
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    print("\nStudent mismatch configuration:")
    print(f"  hidden_fraction:      {hidden_fraction}")
    print(f"  missing_unit_fraction: {missing_unit_fraction}")
    print(f"  noise_frac:           {noise_frac}")
    print(f"  input_type:           {input_type}")
    print(f"  ff_init:              {ff_init}")

    # ======================================
    # Load Network Structure
    # ======================================

    network_structure = np.load(input_dir / "network_structure.npz")

    weights = network_structure["recurrent_weights"].copy()
    teacher_ff_weights = network_structure["feedforward_weights"].copy()
    cell_type_indices = network_structure["cell_type_indices"].copy()
    feedforward_cell_type_indices = network_structure[
        "feedforward_cell_type_indices"
    ].copy()
    recurrent_mask = network_structure["recurrent_connectivity"].copy()
    feedforward_mask_np = network_structure["feedforward_connectivity"].copy()

    n_neurons_original = weights.shape[0]

    # ======================================
    # Phase A: Three-Tier Neuron Classification
    # ======================================

    structural_hidden_indices = np.array([], dtype=int)

    if missing_unit_fraction > 0:
        masking_rng = np.random.default_rng(seed)
        (
            weights,
            teacher_ff_weights,
            recurrent_mask,
            feedforward_mask_np,
            cell_type_indices,
            structural_visible_indices,
            structural_hidden_indices,
        ) = mask_hidden_units(
            weights,
            teacher_ff_weights,
            recurrent_mask,
            feedforward_mask_np,
            cell_type_indices,
            missing_unit_fraction,
            masking_rng,
        )
        print(f"\nStructural masking: {len(structural_hidden_indices)} neurons removed")
        print(
            f"  Remaining: {len(structural_visible_indices)} neurons "
            f"(from {n_neurons_original} original)"
        )

    n_neurons = weights.shape[0]

    if hidden_fraction > 0:
        n_hidden = int(n_neurons * hidden_fraction)
        all_indices = np.arange(n_neurons)
        np.random.shuffle(all_indices)
        unobserved_indices = np.sort(all_indices[:n_hidden])
        visible_indices = np.sort(all_indices[n_hidden:])
    else:
        unobserved_indices = np.array([], dtype=int)
        visible_indices = np.arange(n_neurons)

    n_visible = len(visible_indices)
    n_unobserved = len(unobserved_indices)

    print("\nNeuron classification (after structural masking):")
    print(f"  Visible:    {n_visible}")
    print(f"  Unobserved: {n_unobserved}")
    print(f"  Total in model: {n_neurons}")

    # ======================================
    # Phase B: Weight Noise
    # ======================================

    if noise_frac > 0:
        weight_noise_rng = np.random.default_rng(seed + 1)
        n_input_types = cell_type_indices.max() + 1

        print(
            f"\nApplying {noise_frac * 100:.1f}% multiplicative noise "
            f"to recurrent weights (per cell-type pair)..."
        )

        for in_type in range(n_input_types):
            for out_type in range(n_input_types):
                in_mask = cell_type_indices == in_type
                out_mask = cell_type_indices == out_type
                pair_mask = np.outer(in_mask, out_mask)

                pair_weights = weights[pair_mask]
                nonzero = pair_weights != 0
                if nonzero.sum() == 0:
                    continue

                noisy_pair = apply_weight_noise(
                    pair_weights.reshape(-1),
                    noise_frac,
                    rng=weight_noise_rng,
                    preserve_statistics=True,
                )
                weights[pair_mask] = noisy_pair

    # ======================================
    # Cell and Synapse Parameters
    # ======================================

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)
    n_ff_synapse_types = len(feedforward.get_synapse_params())

    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        o = cp.copy()
        o["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(o)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        o = sp.copy()
        o["cell_id"] = sp["cell_id"] + n_ff_cell_types
        o["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(o)

    # ======================================
    # Scaling Factors and Perturbation
    # ======================================

    sf_feedforward = np.array(scaling_factors_cfg["feedforward"])
    sf_recurrent = np.array(scaling_factors_cfg["recurrent"])

    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    perturbation_factors = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )
    target_scaling_factors = 1.0 / perturbation_factors

    # Perturb all weights
    all_ct = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )
    n_total_inputs = feedforward_mask_np.shape[0] + n_neurons
    concatenated_weights = np.concatenate([teacher_ff_weights, weights], axis=0)

    perturbed_weights = concatenated_weights.copy()
    for input_idx in range(n_total_inputs):
        input_type_idx = all_ct[input_idx]
        for output_idx in range(n_neurons):
            output_type_idx = cell_type_indices[output_idx]
            perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                input_type_idx, output_type_idx
            ]

    n_ff_teacher = teacher_ff_weights.shape[0]
    perturbed_rec_weights = perturbed_weights[n_ff_teacher:, :]

    output_cell_type_names = recurrent.cell_types.names
    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names

    # ======================================
    # Load Dataset
    # ======================================

    if input_type == "latents":
        spike_dataset = LatentOUDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
        )
        n_ff_inputs = spike_dataset.n_latents
        learnable_ff_cell_type_indices = np.zeros(n_ff_inputs, dtype=np.int64)
    elif input_type == "exact_spikes":
        spike_dataset = ExactFFDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
            ff_smoothing_tau=ff_smoothing_tau,
        )
        n_ff_inputs = n_ff_teacher
        learnable_ff_cell_type_indices = feedforward_cell_type_indices.copy()
    elif input_type == "constant":
        spike_dataset = HomogeneousPoissonFFDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
            firing_rate_override=firing_rate_hz,
        )
        n_ff_inputs = n_ff_teacher
        learnable_ff_cell_type_indices = feedforward_cell_type_indices.copy()
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    batch_size = spike_dataset.batch_size
    dt = spike_dataset.dt
    num_chunks = spike_dataset.num_chunks

    print(f"\nDataset: {input_type}, {num_chunks} chunks x {batch_size} batch")

    # ======================================
    # Initialise Learnable FF Weights
    # ======================================

    learnable_ff_mask = np.ones((n_ff_inputs, n_neurons), dtype=bool)

    if ff_init == "teacher":
        # Oracle init: start from the ground-truth teacher FF matrix. Only
        # valid when n_ff_inputs == n_ff_teacher (i.e. not latent mode),
        # since the teacher matrix has the true presynaptic dimension.
        if n_ff_inputs != n_ff_teacher:
            raise ValueError(
                f"ff_init='teacher' requires n_ff_inputs ({n_ff_inputs}) "
                f"to match n_ff_teacher ({n_ff_teacher}); use input_type "
                f"'exact_spikes' or 'constant', not 'latents'."
            )
        learnable_ff_weights = teacher_ff_weights.astype(np.float32).copy()
        print("  FF weight init: teacher (oracle)")
    else:
        learnable_ff_weights = np.zeros((n_ff_inputs, n_neurons), dtype=np.float32)
        for type_idx, type_name in enumerate(output_cell_type_names):
            type_mask = cell_type_indices == type_idx
            mean_ff_type = teacher_ff_weights[:, type_mask].sum(axis=0).mean()
            constant_val = mean_ff_type / n_ff_inputs
            learnable_ff_weights[:, type_mask] = constant_val
            print(f"  FF weight init -> {type_name}: {constant_val:.6f}")

    # Apply perturbation to learnable FF weights. When weight_perturbation
    # _variance = 0 this is a no-op (perturbation_factors are all 1). We
    # skip the loop entirely in oracle mode for clarity — applying it to
    # teacher weights with non-zero perturbation would defeat the point.
    if ff_init != "teacher":
        for input_idx in range(n_ff_inputs):
            ff_type_idx = learnable_ff_cell_type_indices[input_idx]
            for output_idx in range(n_neurons):
                output_type_idx = cell_type_indices[output_idx]
                learnable_ff_weights[input_idx, output_idx] *= perturbation_factors[
                    ff_type_idx, output_type_idx
                ]

    # ======================================
    # Split Into Two-Layer Weight Matrices
    # ======================================

    # Layer 1 (hidden/unobserved neurons):
    #   FF: [learnable_FF→hidden, visible→hidden]
    #   Recurrent: hidden→hidden

    layer1_ff_weights = np.concatenate(
        [
            learnable_ff_weights[:, unobserved_indices],
            perturbed_rec_weights[visible_indices][:, unobserved_indices],
        ],
        axis=0,
    )
    layer1_ff_mask = np.concatenate(
        [
            learnable_ff_mask[:, unobserved_indices],
            recurrent_mask[visible_indices][:, unobserved_indices],
        ],
        axis=0,
    )
    layer1_rec_weights = perturbed_rec_weights[unobserved_indices][
        :, unobserved_indices
    ]
    layer1_rec_mask = recurrent_mask[unobserved_indices][:, unobserved_indices]
    layer1_cell_type_indices = cell_type_indices[unobserved_indices]
    layer1_ff_cell_type_indices = np.concatenate(
        [
            learnable_ff_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    # Layer 2 (visible neurons):
    #   FF: [learnable_FF→visible, hidden→visible, visible→visible]
    #   No recurrence

    layer2_ff_weights = np.concatenate(
        [
            learnable_ff_weights[:, visible_indices],
            perturbed_rec_weights[unobserved_indices][:, visible_indices],
            perturbed_rec_weights[visible_indices][:, visible_indices],
        ],
        axis=0,
    )
    layer2_ff_mask = np.concatenate(
        [
            learnable_ff_mask[:, visible_indices],
            recurrent_mask[unobserved_indices][:, visible_indices],
            recurrent_mask[visible_indices][:, visible_indices],
        ],
        axis=0,
    )
    layer2_cell_type_indices = cell_type_indices[visible_indices]
    layer2_ff_cell_type_indices = np.concatenate(
        [
            learnable_ff_cell_type_indices,
            cell_type_indices[unobserved_indices] + n_ff_cell_types,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    print(
        f"\n  Layer 1 (hidden): FF {layer1_ff_weights.shape}, "
        f"Rec {layer1_rec_weights.shape}"
    )
    print(f"  Layer 2 (visible): FF {layer2_ff_weights.shape}")

    # ======================================
    # Save Initial State & Targets
    # ======================================

    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors,
    )
    np.savez(
        targets_dir / "neuron_classification.npz",
        visible_indices=visible_indices,
        unobserved_indices=unobserved_indices,
        structural_hidden_indices=structural_hidden_indices,
        n_neurons_original=n_neurons_original,
        n_neurons=n_neurons,
        hidden_fraction=hidden_fraction,
        missing_unit_fraction=missing_unit_fraction,
    )

    initial_dir = output_dir / "initial_state"
    initial_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        initial_dir / "layer1.npz",
        ff_weights=layer1_ff_weights,
        rec_weights=layer1_rec_weights,
        ff_mask=layer1_ff_mask,
        rec_mask=layer1_rec_mask,
    )
    np.savez(
        initial_dir / "layer2.npz",
        ff_weights=layer2_ff_weights,
        ff_mask=layer2_ff_mask,
    )

    # ==============================
    # Wandb
    # ==============================

    wandb_logger = None
    if wandb_config:
        run_name = wandb_config.pop("name", None) or output_dir.name
        wandb_logger = wandb.init(
            name=run_name,
            dir=str(output_dir),
            config={
                "hidden_fraction": hidden_fraction,
                "missing_unit_fraction": missing_unit_fraction,
                "noise_frac": noise_frac,
                "input_type": input_type,
                "n_visible": n_visible,
                "n_unobserved": n_unobserved,
                "n_neurons": n_neurons,
            },
            **wandb_config,
        )
        wandb.define_metric("epoch")

    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # ==============================
    # Collate & Dataloader
    # ==============================

    # Map visible_indices (in reduced model space) to original neuron indices
    # for indexing into the dataset's target_spikes (which has all N_original neurons).
    if missing_unit_fraction > 0:
        visible_original = structural_visible_indices[visible_indices]
    else:
        visible_original = visible_indices
    visible_tensor = torch.from_numpy(visible_original).long()
    collate_fn = VisibleDrivenCollate(visible_tensor)

    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    # ==============================
    # Teacher Data for Hidden Stats
    # ==============================

    teacher_zarr_root = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    teacher_spikes_zarr = teacher_zarr_root["output_spikes"]

    # Cache teacher hidden spikes in memory to avoid network filesystem reads
    # during stats_computer. Only batch 0, hidden columns — typically <50 MB.
    # Map to original indices if structural masking was applied.
    if n_unobserved > 0:
        if missing_unit_fraction > 0:
            unobserved_original = structural_visible_indices[unobserved_indices]
        else:
            unobserved_original = unobserved_indices
        teacher_hidden_cache = np.array(teacher_spikes_zarr[0, :, unobserved_original])
    else:
        teacher_hidden_cache = None

    # ==============================
    # Teacher Visible Rate Targets
    # ==============================
    # Population-mean firing rate per cell type, computed from the teacher
    # *visible* neurons only. Used as an unbiased target for the hidden-rate
    # penalty (using hidden teacher rates would leak unobservable state).
    teacher_visible_cache = np.array(teacher_spikes_zarr[0, :, visible_original])
    _dur_s = teacher_visible_cache.shape[0] * dt / 1000.0
    _visible_rates_hz = teacher_visible_cache.sum(axis=0) / _dur_s
    _visible_ct = cell_type_indices[visible_indices]
    _target_rates = np.zeros(len(output_cell_type_names), dtype=np.float32)
    _target_stds = np.zeros(len(output_cell_type_names), dtype=np.float32)
    for _ct in range(len(output_cell_type_names)):
        _m = _visible_ct == _ct
        if _m.any():
            _target_rates[_ct] = _visible_rates_hz[_m].mean()
            _target_stds[_ct] = _visible_rates_hz[_m].std()
    teacher_visible_rate_targets = torch.from_numpy(_target_rates).to(device)
    teacher_visible_rate_std_targets = torch.from_numpy(_target_stds).to(device)
    print(
        "\nTeacher visible population rates (Hz): "
        + ", ".join(
            f"{output_cell_type_names[i]}={_target_rates[i]:.2f}±{_target_stds[i]:.2f}"
            for i in range(len(output_cell_type_names))
        )
    )

    # Teacher FF weights in the reduced (post-masking) neuron space for
    # cosine-sim metric. Built once, broadcast to device at use time.
    if missing_unit_fraction > 0:
        teacher_ff_reduced = teacher_ff_weights  # already reduced in Phase A
    else:
        teacher_ff_reduced = teacher_ff_weights
    # Reorder columns so [unobserved | visible] matches the student's
    # (layer1-hidden then layer2-visible) column space for cosine-sim.
    teacher_ff_student_order = np.concatenate(
        [
            teacher_ff_reduced[:, unobserved_indices],
            teacher_ff_reduced[:, visible_indices],
        ],
        axis=1,
    ).astype(np.float32)
    teacher_ff_flat = teacher_ff_student_order.flatten()
    teacher_ff_norm = float(np.linalg.norm(teacher_ff_flat) + 1e-12)

    # ==============================
    # Stats Computer
    # ==============================

    # Holds a reference to the TwoLayerSNN so stats_computer can introspect
    # its FF-input matrices for the cosine-similarity-to-teacher metric.
    current_model_ref = {"model": None}

    def stats_computer(snapshot):
        """Compute summary statistics for logging."""
        # spikes shape: (1, time, n_visible) — Layer 2 outputs visible only
        student_visible_spikes = snapshot["spikes"][0, :, :]
        n_timesteps = student_visible_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        teacher_visible_spikes = snapshot["target_spikes"][0, :n_timesteps, :]
        teacher_visible_rates = teacher_visible_spikes.sum(axis=0) / duration_s
        student_visible_rates = student_visible_spikes.sum(axis=0) / duration_s

        stats = {}

        # Visible neuron firing rates
        visible_ct = cell_type_indices[visible_indices]
        for type_idx, type_name in enumerate(output_cell_type_names):
            mask = visible_ct == type_idx
            if mask.sum() > 0:
                stats[f"firing_rate/teacher_visible_{type_name}_mean"] = float(
                    teacher_visible_rates[mask].mean()
                )
                stats[f"firing_rate/student_visible_{type_name}_mean"] = float(
                    student_visible_rates[mask].mean()
                )

        # Hidden neuron firing rates (student from hidden_spikes, teacher from zarr)
        if n_unobserved > 0 and "hidden_spikes" in snapshot:
            student_hidden_spikes = snapshot["hidden_spikes"][0, :n_timesteps, :]
            student_hidden_rates = student_hidden_spikes.sum(axis=0) / duration_s

            epoch = snapshot.get("epoch", 0)
            n_chunks_accumulated = n_timesteps // chunk_size
            start_chunk = (epoch - n_chunks_accumulated + 1) % num_chunks
            total_timesteps = num_chunks * chunk_size
            start_t = start_chunk * chunk_size
            end_t = start_t + n_timesteps

            if end_t <= total_timesteps:
                teacher_hidden_spikes = teacher_hidden_cache[start_t:end_t]
            else:
                teacher_hidden_spikes = np.concatenate(
                    [
                        teacher_hidden_cache[start_t:total_timesteps],
                        teacher_hidden_cache[: end_t - total_timesteps],
                    ],
                    axis=0,
                )
            teacher_hidden_rates = teacher_hidden_spikes.sum(axis=0) / duration_s

            unobserved_ct = cell_type_indices[unobserved_indices]
            for type_idx, type_name in enumerate(output_cell_type_names):
                mask = unobserved_ct == type_idx
                if mask.sum() > 0:
                    stats[f"firing_rate/teacher_hidden_{type_name}_mean"] = float(
                        teacher_hidden_rates[mask].mean()
                    )
                    stats[f"firing_rate/student_hidden_{type_name}_mean"] = float(
                        student_hidden_rates[mask].mean()
                    )

        # Recurrent scaling factors (from Layer 1). FF SFs are absorbed into
        # the learned low-rank weights, so we don't log them.
        sf_rec = snapshot["scaling_factors"]

        # target_scaling_factors rows are FF types then recurrent types;
        # drop the FF rows since FF SFs are absorbed into learned weights.
        n_ff_types = len(feedforward.cell_types.names)
        rec_target_sf = target_scaling_factors[n_ff_types:]
        rec_source_names = input_cell_type_names[n_ff_types:]

        for source_idx in range(sf_rec.shape[0]):
            source_name = rec_source_names[source_idx]
            for target_idx in range(sf_rec.shape[1]):
                target_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_name}_to_{target_name}"
                target_val = rec_target_sf[source_idx, target_idx]
                if target_val != 0:
                    normalized = sf_rec[source_idx, target_idx] / target_val
                else:
                    normalized = sf_rec[source_idx, target_idx]
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        # FF-input matrix diagnostics: cosine similarity to teacher plus
        # max/mean weight (the latter is the direct indicator of whether
        # L2 is keeping runaway outliers under control).
        model_ref = current_model_ref["model"]
        if model_ref is not None:
            with torch.no_grad():
                # ``weights_FF`` is assembled from projections on each
                # access; slice off the mitral (true-FF) rows.
                ff_h = (
                    model_ref.layer1.weights_FF[:n_ff_inputs, :].detach().cpu().numpy()
                )
                ff_v = (
                    model_ref.layer2.weights_FF[:n_ff_inputs, :].detach().cpu().numpy()
                )
            student_ff_flat = np.concatenate([ff_h, ff_v], axis=1).flatten()
            student_ff_norm = np.linalg.norm(student_ff_flat) + 1e-12
            cos_sim = float(
                np.dot(student_ff_flat, teacher_ff_flat)
                / (student_ff_norm * teacher_ff_norm)
            )
            stats["ff_matrix/cosine_sim_teacher"] = cos_sim
            stats["ff_matrix/max_weight"] = float(max(ff_h.max(), ff_v.max()))
            stats["ff_matrix/mean_weight"] = float(
                (ff_h.sum() + ff_v.sum()) / (ff_h.size + ff_v.size)
            )

        return stats

    # ==============================
    # Plot Generator
    # ==============================

    def plot_generator(spikes, target_spikes, input_spikes, epoch=0, **kwargs):
        figures = {}

        # spikes are already visible-only from Layer 2
        n_plot_vis = min(10, spikes.shape[2])
        interleaved_vis = np.zeros((1, spikes.shape[1], 2 * n_plot_vis))
        for i in range(n_plot_vis):
            interleaved_vis[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved_vis[0, :, 2 * i + 1] = spikes[0, :, i]

        figures["spike_comparison_visible"] = plot_spike_trains(
            spikes=interleaved_vis,
            dt=dt,
            cell_type_indices=np.array([0, 1] * n_plot_vis),
            cell_type_names=["Target", "Trained"],
            n_neurons_plot=2 * n_plot_vis,
            n_compared=2,
            fraction=1.0,
            random_seed=None,
            title=f"Target vs Trained ({n_plot_vis} visible)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        # Hidden neuron comparison
        hidden_spikes = kwargs.get("hidden_spikes")
        if n_unobserved > 0 and hidden_spikes is not None:
            n_plot_hid = min(10, hidden_spikes.shape[2])
            n_time = spikes.shape[1]
            n_chunks_in_plot = n_time // chunk_size
            start_chunk = (epoch - n_chunks_in_plot + 1) % num_chunks
            start_t = start_chunk * chunk_size
            end_t = start_t + n_time

            total_timesteps = num_chunks * chunk_size
            if end_t <= total_timesteps:
                teacher_hid = teacher_hidden_cache[start_t:end_t]
            else:
                teacher_hid = np.concatenate(
                    [
                        teacher_hidden_cache[start_t:total_timesteps],
                        teacher_hidden_cache[: end_t - total_timesteps],
                    ],
                    axis=0,
                )

            interleaved_hid = np.zeros((1, n_time, 2 * n_plot_hid))
            for i in range(n_plot_hid):
                interleaved_hid[0, :, 2 * i] = teacher_hid[:, i]
                interleaved_hid[0, :, 2 * i + 1] = hidden_spikes[0, :, i]

            figures["spike_comparison_hidden"] = plot_spike_trains(
                spikes=interleaved_hid,
                dt=dt,
                cell_type_indices=np.array([0, 1] * n_plot_hid),
                cell_type_names=["Teacher", "Student"],
                n_neurons_plot=2 * n_plot_hid,
                n_compared=2,
                fraction=1.0,
                random_seed=None,
                title=f"Teacher vs Student ({n_plot_hid} hidden)",
                ylabel="Neuron",
                figsize=(14, 8),
            )

        return figures

    # ==============================================================
    # Build Projections
    # ==============================================================
    #
    # Strategy:
    #   * Mitral → output cell type (LowRank): per-output-type shared
    #     U_ct / V_ct, sliced per layer via SharedSlicedLowRankProjection.
    #     U_ct: (n_ff_inputs, ff_rank). V_ct: (ff_rank, n_of_type) — over
    #     all neurons of that cell type (hidden + visible), so a single
    #     V_ct serves both layers via column-slicing.
    #   * Recurrent rerouted FF rows (visible → hidden, hidden → visible,
    #     visible → visible) become ScalingFactorProjection(connectome,
    #     init_sf=1.0). The perturbed teacher block is the frozen
    #     connectome; the SF starts at 1.0 (FF SFs are already baked into
    #     ``perturbed_rec_weights``).
    #   * Layer 1 hidden→hidden recurrent: ScalingFactorProjection per pair.
    #
    # Per-source-cell-type sharing constraint: within a single source
    # cell type, all (src, tgt) projections must share the same
    # caching_mode. Mitral is "weights" (LowRank); each rerouted-rec
    # source cell type is "scaling_factors". Distinct cell-type ids, so
    # the constraint holds.

    total_chunks = epochs * num_chunks

    print(f"\n{'=' * 60}")
    print(f"TRAINING ({epochs} epochs, {total_chunks} chunks)")
    print(f"  FF rank:            {ff_rank}")
    print(f"  FF smoothing tau:   {ff_smoothing_tau}")
    print(f"  lr weights:         {lr_weights}")
    print(f"  lr scaling:         {lr_scaling}")
    print(f"  grad clip:          {grad_clip}")
    print(f"{'=' * 60}")

    # --- Per-output-cell-type SVD of the (n_ff_inputs, n_of_type) FF
    # block built from ``learnable_ff_weights`` (already perturbed). One
    # (U_ct, V_ct) Parameter pair per output type, shared across layers.

    ff_cell_type_names = list(feedforward.cell_types.names[:n_ff_cell_types])
    if len(ff_cell_type_names) != 1:
        raise NotImplementedError(
            "SharedSlicedLowRankProjection assumes a single FF (mitral) cell "
            "type covering all true-FF rows; got "
            f"{len(ff_cell_type_names)} FF cell types: {ff_cell_type_names!r}."
        )

    U_per_type: list[nn.Parameter] = []
    V_per_type: list[nn.Parameter] = []
    # ct_neurons_full[ct] = indices (student space) of all neurons of
    # that output cell type, ordered as the columns of V_ct.
    ct_neurons_full: list[np.ndarray] = []
    # Per-layer column-slice into ct_neurons_full[ct]: positions of
    # unobserved (layer 1) and visible (layer 2) neurons of that type.
    ct_layer1_cols: list[np.ndarray] = []
    ct_layer2_cols: list[np.ndarray] = []

    unobs_set = set(unobserved_indices.tolist())
    vis_set = set(visible_indices.tolist())

    for ct_idx, ct_name in enumerate(output_cell_type_names):
        ct_mask = cell_type_indices == ct_idx
        ct_neurons = np.flatnonzero(ct_mask)  # absolute student-space ids
        ct_neurons_full.append(ct_neurons)

        ff_block = learnable_ff_weights[:, ct_neurons].astype(np.float32)
        log_block = np.log(np.maximum(ff_block, 1e-8))
        log_block_t = torch.from_numpy(log_block)

        rank_eff = min(ff_rank, *log_block_t.shape)
        U_full, S_full, Vh_full = torch.linalg.svd(log_block_t, full_matrices=False)
        # Zero out singular values that are float32-noise relative to the
        # leading one. The FF init is effectively rank-1 (uniform per
        # output type), so without this the SVD reconstruction picks up
        # a spurious ±0.5-in-log-space perturbation from tail singular
        # values before any optimiser step has run.
        S_eff = S_full[:rank_eff].clone()
        S_eff[S_eff < S_full[0] * 1e-4] = 0.0
        U_init = U_full[:, :rank_eff] * S_eff.sqrt().unsqueeze(0)
        V_init = S_eff.sqrt().unsqueeze(1) * Vh_full[:rank_eff, :]
        var_explained = float(
            (S_full[:rank_eff] ** 2).sum() / max((S_full**2).sum(), 1e-10)
        )

        U_param = nn.Parameter(U_init.contiguous())
        V_param = nn.Parameter(V_init.contiguous())
        U_per_type.append(U_param)
        V_per_type.append(V_param)

        l1_cols = np.array(
            [i for i, n in enumerate(ct_neurons) if int(n) in unobs_set],
            dtype=np.int64,
        )
        l2_cols = np.array(
            [i for i, n in enumerate(ct_neurons) if int(n) in vis_set],
            dtype=np.int64,
        )
        ct_layer1_cols.append(l1_cols)
        ct_layer2_cols.append(l2_cols)

        print(
            f"  {ct_name}: U {tuple(U_param.shape)}, V {tuple(V_param.shape)}"
            f"  ({var_explained:.1%} variance; rank {rank_eff})"
        )

    # --- Layer 1 projections ---
    # Source FF cell types (layer1_ff_cell_type_indices namespace):
    #   * mitral (id 0..n_ff_cell_types-1) — SharedSliced (unobserved cols).
    #   * recurrent (id n_ff_cell_types + ct) — visible→hidden rerouted,
    #     ScalingFactorProjection per (src_ct, tgt_ct) pair.

    layer1_ff_projections: dict = {}

    for ff_id, ff_name in enumerate(ff_cell_type_names):
        for tgt_id, tgt_name in enumerate(output_cell_type_names):
            l1_cols_t = torch.from_numpy(ct_layer1_cols[tgt_id]).long()
            layer1_ff_projections[(ff_name, tgt_name)] = SharedSlicedLowRankProjection(
                U=U_per_type[tgt_id],
                V_full=V_per_type[tgt_id],
                cols=l1_cols_t,
                mask=None,
            )

    rec_idx_l1 = {
        name: np.flatnonzero(layer1_cell_type_indices == ct_id)
        for ct_id, name in enumerate(output_cell_type_names)
    }
    for src_ct_id, src_name in enumerate(output_cell_type_names):
        src_rows = np.flatnonzero(
            layer1_ff_cell_type_indices == src_ct_id + n_ff_cell_types
        )
        for tgt_ct_id, tgt_name in enumerate(output_cell_type_names):
            tgt_cols = rec_idx_l1[tgt_name]
            block = (
                layer1_ff_weights[np.ix_(src_rows, tgt_cols)]
                * layer1_ff_mask[np.ix_(src_rows, tgt_cols)].astype(np.float32)
            ).astype(np.float32)
            layer1_ff_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                connectome=block, init_sf=1.0
            )

    # Layer 1 recurrent (hidden→hidden) — ScalingFactor per pair.
    layer1_rec_projections: dict = {}
    for src_ct_id, src_name in enumerate(output_cell_type_names):
        src_rows = rec_idx_l1[src_name]
        for tgt_ct_id, tgt_name in enumerate(output_cell_type_names):
            tgt_cols = rec_idx_l1[tgt_name]
            block = (
                layer1_rec_weights[np.ix_(src_rows, tgt_cols)]
                * layer1_rec_mask[np.ix_(src_rows, tgt_cols)].astype(np.float32)
            ).astype(np.float32)
            layer1_rec_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                connectome=block, init_sf=1.0
            )

    # --- Layer 2 projections ---
    # Source FF cell types in layer 2's namespace:
    #   * mitral — SharedSliced (visible cols), sharing U_ct/V_ct with layer 1.
    #   * recurrent (id n_ff_cell_types + ct) — appears in layer 2's input
    #     rows from BOTH the hidden→visible and visible→visible blocks.
    #     They share the same source cell-type id, so the simulator
    #     processes them as a single combined source-row block. We build
    #     one ScalingFactorProjection per (src_ct, tgt_ct) covering all
    #     such source rows in their natural order.

    layer2_ff_projections: dict = {}

    rec_idx_l2 = {
        name: np.flatnonzero(layer2_cell_type_indices == ct_id)
        for ct_id, name in enumerate(output_cell_type_names)
    }
    for ff_id, ff_name in enumerate(ff_cell_type_names):
        for tgt_id, tgt_name in enumerate(output_cell_type_names):
            l2_cols_t = torch.from_numpy(ct_layer2_cols[tgt_id]).long()
            layer2_ff_projections[(ff_name, tgt_name)] = SharedSlicedLowRankProjection(
                U=U_per_type[tgt_id],
                V_full=V_per_type[tgt_id],
                cols=l2_cols_t,
                mask=None,
            )

    for src_ct_id, src_name in enumerate(output_cell_type_names):
        src_rows = np.flatnonzero(
            layer2_ff_cell_type_indices == src_ct_id + n_ff_cell_types
        )
        for tgt_ct_id, tgt_name in enumerate(output_cell_type_names):
            tgt_cols = rec_idx_l2[tgt_name]
            block = (
                layer2_ff_weights[np.ix_(src_rows, tgt_cols)]
                * layer2_ff_mask[np.ix_(src_rows, tgt_cols)].astype(np.float32)
            ).astype(np.float32)
            layer2_ff_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                connectome=block, init_sf=1.0
            )

    # ==============================================================
    # Build Layers and TwoLayerSNN
    # ==============================================================

    layer1 = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=layer1_rec_projections,
        ff_projections=layer1_ff_projections,
        cell_type_indices=layer1_cell_type_indices,
        cell_type_indices_FF=layer1_ff_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params=recurrent_synapse_params,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=layer2_ff_projections,
        cell_type_indices=layer2_cell_type_indices,
        cell_type_indices_FF=layer2_ff_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    model = TwoLayerSNN(layer1, layer2, n_ff=n_ff_inputs, return_hidden_spikes=True)
    # Register the shared U/V Parameters on the model so .to(device),
    # state_dict(), and clip_grad_norm_(model.parameters(), ...) all reach
    # them. The SharedSlicedLowRankProjection instances reference the same
    # Python objects via object.__setattr__, so this does not double-count.
    model._shared_ff_U = nn.ParameterList(U_per_type)
    model._shared_ff_V = nn.ParameterList(V_per_type)
    model.to(device)
    current_model_ref["model"] = model

    # ==============================================================
    # Losses, Optimiser, Scheduler
    # ==============================================================

    losses, loss_weights = build_losses(
        model,
        van_rossum_kwargs=dict(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=dt,
            window_size=chunk_size,
            device=device,
        ),
        loss_weights_cfg=hyperparameters.loss_weight,
        hidden_rate_targets=teacher_visible_rate_targets,
        hidden_rate_std_targets=teacher_visible_rate_std_targets,
        dt_ms=dt,
        U_per_type=U_per_type,
        V_per_type=V_per_type,
        cell_type_names=output_cell_type_names,
    )

    # Parameter groups:
    #   - "weights": shared U_ct, V_ct (one Parameter each per output
    #     cell type, registered once even though both layers reference
    #     them via SharedSlicedLowRankProjection).
    #   - "scaling": log_sf parameters from every ScalingFactorProjection
    #     across Layer 1 (recurrent + rerouted-FF) and Layer 2 (rerouted-
    #     FF only; the mitral SharedSliced LowRank instances expose no
    #     .log_sf and are skipped naturally).
    weight_params: list[nn.Parameter] = [*U_per_type, *V_per_type]

    sf_params: list[nn.Parameter] = []
    for proj in model.layer1.rec_projections.values():
        if isinstance(proj, ScalingFactorProjection):
            sf_params.append(proj.log_sf)
    for proj in model.layer1.ff_projections.values():
        if isinstance(proj, ScalingFactorProjection):
            sf_params.append(proj.log_sf)
    for proj in model.layer2.projections.values():
        if isinstance(proj, ScalingFactorProjection):
            sf_params.append(proj.log_sf)

    param_groups = [{"params": weight_params, "lr": lr_weights}]
    if sf_params:
        param_groups.append({"params": sf_params, "lr": lr_scaling})

    optimiser = torch.optim.Adam(param_groups, betas=(beta1, beta2), eps=adam_eps)

    def make_cosine_lambda(lr_init, lr_min, n_epochs):
        frac = lr_min / lr_init

        def cosine_lr_lambda(epoch):
            return frac + (1 - frac) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2

        return cosine_lr_lambda

    lr_pairs = [(lr_weights, lr_min_weights)]
    if sf_params:
        lr_pairs.append((lr_scaling, lr_min_scaling))
    lambdas = [make_cosine_lambda(li, lm, total_chunks) for li, lm in lr_pairs]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambdas)

    def grad_clip_fn(model):
        # Joint clip across [U, V] so the relative U↔V update direction
        # is preserved when only the combined norm blows up.
        trainable_weights = [p for p in weight_params if p.grad is not None]
        if trainable_weights:
            torch.nn.utils.clip_grad_norm_(trainable_weights, max_norm=grad_clip)
        trainable_sf = [p for p in sf_params if p.grad is not None]
        if trainable_sf:
            torch.nn.utils.clip_grad_norm_(trainable_sf, max_norm=grad_clip)

    scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")
    pbar = tqdm(range(total_chunks), desc="Training", unit="chunk")

    # Weight snapshot callback: save the reconstructed mitral→neuron matrix
    # (log space) at every checkpoint so analysis notebooks can recover the
    # full student FF weight trajectory.
    snapshot_dir = output_dir / "weight_snapshots"
    snapshot_dir.mkdir(exist_ok=True)

    def save_weight_snapshot(epoch, out_dir, model):
        l1_w = model.layer1.weights_FF.detach().cpu().numpy()
        l2_w = model.layer2.weights_FF.detach().cpu().numpy()
        full_W = np.zeros((n_ff_inputs, n_neurons), dtype=np.float32)
        full_W[:, unobserved_indices] = l1_w[:n_ff_inputs, :]
        full_W[:, visible_indices] = l2_w[:n_ff_inputs, :]
        np.savez(
            snapshot_dir / f"epoch_{epoch:06d}.npz",
            ff_weights=np.log(np.maximum(full_W, 1e-12)),
            epoch=epoch,
        )

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions=losses,
        loss_weights=loss_weights,
        device=device,
        num_epochs=total_chunks,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_clip_fn=grad_clip_fn,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
        checkpoint_callbacks=[save_weight_snapshot],
    )
    trainer.metrics_logger = metrics_logger
    if wandb_logger:
        trainer.wandb_logger = wandb_logger

    best_loss = trainer.train(output_dir=output_dir)

    # ==============================
    # Cleanup
    # ==============================

    print(f"\n{'=' * 60}")
    print("Full Inference Training Complete")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"{'=' * 60}")

    if metrics_logger:
        metrics_logger.close()

    if wandb_logger:
        wandb.finish()

    return best_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("params_file", type=Path)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
