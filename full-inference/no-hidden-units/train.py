"""
Full-inference: no-hidden variant.

The entire network is observed — every recurrent neuron's spikes are
available as a training target. The student learns:

* the full feedforward weight matrix (low-rank ``U @ V``), and
* per-cell-type scaling factors on the recurrent connections.

Training is a single low-rank pass that learns the feedforward weights
(via ``exp(U @ V)``) while jointly refining recurrent scaling factors
(or, for the no-connectome control, full-rank recurrent weights).

Optional perturbations applied to the student before training:

* ``noise_frac``            — multiplicative log-normal noise on the
                              recurrent weights (per cell-type pair).
* ``missing_unit_fraction`` — a fraction of recurrent neurons are
                              structurally removed from the student.
* ``shuffle_connectome``    — recurrent weights are permuted within each
                              cell-type-pair block (preserves per-pair
                              distribution and sparsity, destroys wiring).
* ``shuffle_weights``       — like ``shuffle_connectome`` but the binary
                              connectome is held fixed; only the nonzero
                              weight values are permuted among existing
                              connections within each cell-type-pair block.
* ``input_type``            — feedforward signal: actual mitral spikes,
                              homogeneous Poisson, or latent OU rates.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from dataloaders.supervised import (
    ExactFFDataset,
    HomogeneousPoissonFFDataset,
    CyclicSampler,
    FeedforwardCollate,
    VisibleSubsetCollate,
)
from dataloaders.supervised.latent_ou import LatentOUDataset
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.projections import (
    FullRankProjection,
    LowRankProjection,
    ScalingFactorProjection,
)
from training_utils.losses import VanRossumLoss
from training_utils import load_checkpoint, AsyncLogger
from configs import (
    DATALOADER_KWARGS,
    StudentSimulationConfig,
    StudentTrainingConfig,
    StudentHyperparameters,
)
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from snn_runners import SNNTrainer
import toml
import wandb
from visualization.neuronal_dynamics import plot_spike_trains


# ================================================================
# Student-mismatch helpers (copied from full-inference/train.py)
# ================================================================


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
    """Multiplicative log-normal noise with optional mean/std preservation."""
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


def shuffle_connectome_by_cell_type(weights, recurrent_mask, cell_type_indices, rng):
    """Shuffle recurrent weights within each cell-type-pair block.

    For every (in_type, out_type) sub-block of the recurrent matrix, we draw
    one permutation and apply it jointly to the weights and the connectivity
    mask. This preserves the block's exact nonzero count (sparsity) and the
    multiset of weight values (distribution) — only the target positions
    change. Used as a control: the student inherits the teacher's per-pair
    statistics but none of the specific wiring.
    """
    weights = weights.copy()
    recurrent_mask = recurrent_mask.copy()

    n_types = int(cell_type_indices.max()) + 1
    for in_type in range(n_types):
        for out_type in range(n_types):
            in_m = cell_type_indices == in_type
            out_m = cell_type_indices == out_type
            pair_mask = np.outer(in_m, out_m)
            n_pair = int(pair_mask.sum())
            if n_pair == 0:
                continue
            perm = rng.permutation(n_pair)
            w_block = weights[pair_mask]
            m_block = recurrent_mask[pair_mask]
            weights[pair_mask] = w_block[perm]
            recurrent_mask[pair_mask] = m_block[perm]

    return weights, recurrent_mask


def shuffle_weights_within_connectome_by_cell_type(
    weights, recurrent_mask, cell_type_indices, rng
):
    """Permute nonzero weight values within each cell-type-pair block.

    The binary connectome (``recurrent_mask``) is held fixed: every
    existing connection stays where it is, every absent one stays
    absent. Only the *values* at the nonzero positions are reshuffled,
    independently within each (in_type, out_type) sub-block. Preserves
    the exact wiring topology and the per-pair multiset of weight
    values; destroys the pairing between specific connections and their
    learned weights.
    """
    weights = weights.copy()

    n_types = int(cell_type_indices.max()) + 1
    for in_type in range(n_types):
        for out_type in range(n_types):
            in_m = cell_type_indices == in_type
            out_m = cell_type_indices == out_type
            pair_mask = np.outer(in_m, out_m) & recurrent_mask
            n_pair = int(pair_mask.sum())
            if n_pair == 0:
                continue
            perm = rng.permutation(n_pair)
            w_vals = weights[pair_mask]
            weights[pair_mask] = w_vals[perm]

    return weights


def _check_resume_checkpoint(output_dir):
    """Verify a resumable checkpoint exists at ``output_dir/checkpoints``."""
    checkpoint_dir = output_dir / "checkpoints"
    has_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))
    if not has_checkpoints:
        raise RuntimeError(f"Cannot resume from {output_dir}: no checkpoints found.")


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train network with learn-ff-connectivity: low-rank FF weights + scaling factors.

    Feedforward inputs are mitral cell spike trains from spike_data.zarr.
    Learns a low-rank FF weight matrix (U @ V) while recovering
    perturbed recurrent scaling factors.

    Args:
        input_dir (Path): Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir (Path): Directory where training outputs will be saved
        params_file (Path): Path to the file containing training parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to output directory to resume from
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    # Load configuration sections
    simulation = StudentSimulationConfig(**data["simulation"])
    training = StudentTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})
    mismatch_cfg = data.get("student_mismatch", {})

    # Student-mismatch knobs (entire network observed — no hidden_fraction)
    missing_unit_fraction = mismatch_cfg.get("missing_unit_fraction", 0.0)
    noise_frac = mismatch_cfg.get("noise_frac", 0.0)
    input_type = mismatch_cfg.get("input_type", "exact_spikes")
    firing_rate_hz = mismatch_cfg.get("firing_rate_hz", 6.0)
    shuffle_connectome = mismatch_cfg.get("shuffle_connectome", False)
    shuffle_weights = mismatch_cfg.get("shuffle_weights", False)
    no_connectome = mismatch_cfg.get("no_connectome", False)
    if sum([shuffle_connectome, shuffle_weights, no_connectome]) > 1:
        raise ValueError(
            "shuffle_connectome, shuffle_weights, and no_connectome are "
            "mutually exclusive."
        )
    if input_type not in ("constant", "exact_spikes", "latents"):
        raise ValueError(
            f"input_type must be constant|exact_spikes|latents, got {input_type!r}"
        )
    # Extract parameters into plain Python variables
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    chunks_per_update = training.chunks_per_update
    log_interval = training.log_interval
    checkpoint_interval = training.checkpoint_interval
    plot_size = training.plot_size
    mixed_precision = training.mixed_precision
    burn_in_chunks = getattr(training, "burn_in_chunks", 0)
    weight_perturbation_variance = training.weight_perturbation_variance

    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay
    van_rossum_rate_tau_rise = hyperparameters.van_rossum_rate_tau_rise
    van_rossum_rate_tau_decay = hyperparameters.van_rossum_rate_tau_decay
    loss_weight_van_rossum = hyperparameters.loss_weight.van_rossum
    loss_weight_van_rossum_rate = hyperparameters.loss_weight.van_rossum_rate

    ff_rank = data["hyperparameters"].get("ff_rank", 10)

    # Phase-specific parameters
    gradient_config = data.get("gradient", {})

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # Get cell and synapse parameters
    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    n_ff_cell_types = len(feedforward_cell_params)
    n_rec_cell_types = len(recurrent_cell_params)
    n_ff_synapse_types = len(feedforward_synapse_params)

    # Get base scaling factors from config (teacher's "true" SF, used both
    # as the student's initial SF and as the per-pair multiplier when
    # comparing student weights against the teacher in stats_computer).
    sf_feedforward = np.array(scaling_factors["feedforward"], dtype=np.float32)
    sf_recurrent = np.array(scaling_factors["recurrent"], dtype=np.float32)

    # ================================================================
    # Load teacher network structure
    # ================================================================

    network_structure = np.load(input_dir / "network_structure.npz")
    teacher_weights = network_structure["recurrent_weights"].copy()
    teacher_ff_weights = network_structure["feedforward_weights"].copy()
    cell_type_indices = network_structure["cell_type_indices"].copy()
    recurrent_mask = network_structure["recurrent_connectivity"].copy()
    # Teacher FF connectivity (for full-connectivity init; we overwrite later)
    teacher_ff_mask_full = np.ones_like(teacher_ff_weights, dtype=bool)

    n_neurons_original = teacher_weights.shape[0]

    # ================================================================
    # Student-mismatch: structural masking (removes neurons entirely)
    # ================================================================
    structural_hidden_indices = np.array([], dtype=int)
    structural_visible_indices = np.arange(n_neurons_original)

    if missing_unit_fraction > 0:
        masking_rng = np.random.default_rng(seed)
        (
            teacher_weights,
            teacher_ff_weights,
            recurrent_mask,
            teacher_ff_mask_full,
            cell_type_indices,
            structural_visible_indices,
            structural_hidden_indices,
        ) = mask_hidden_units(
            teacher_weights,
            teacher_ff_weights,
            recurrent_mask,
            teacher_ff_mask_full,
            cell_type_indices,
            missing_unit_fraction,
            masking_rng,
        )
        print(
            f"\nStructural masking: {len(structural_hidden_indices)} neurons removed "
            f"(remaining {len(structural_visible_indices)} / {n_neurons_original})"
        )

    n_neurons = teacher_weights.shape[0]
    exc_mask = cell_type_indices == 0  # excitatory neurons

    # Snapshot the pristine teacher matrices (post structural-masking,
    # pre noise/shuffle/densify) so stats_computer can log teacher
    # reference values against the learned weights.
    teacher_ff_pristine = teacher_ff_weights.copy()
    teacher_rec_pristine = teacher_weights.copy()

    # ================================================================
    # No-connectome control: replace the teacher's recurrent weight
    # matrix with a fully-connected, per-cell-type-pair uniform matrix.
    # Each (in_type, out_type) block is filled with the teacher's mean
    # weight for that pair. The lognormal scaling-factor perturbation
    # below is then applied as usual.
    # ================================================================
    if no_connectome:
        n_rec_types = int(cell_type_indices.max()) + 1
        uniform_weights = np.zeros_like(teacher_weights)
        print("\nNo-connectome control: building fully-connected uniform matrix")
        for in_type in range(n_rec_types):
            for out_type in range(n_rec_types):
                in_m = cell_type_indices == in_type
                out_m = cell_type_indices == out_type
                pair_mask = np.outer(in_m, out_m)
                pair_w = teacher_weights[pair_mask]
                nz = pair_w[pair_w != 0]
                pair_mean = float(nz.mean()) if nz.size > 0 else 0.0
                uniform_weights[pair_mask] = pair_mean
                in_name = recurrent.cell_types.names[in_type]
                out_name = recurrent.cell_types.names[out_type]
                print(f"  {in_name} -> {out_name}: {pair_mean:.6f}")
        teacher_weights = uniform_weights
        recurrent_mask = np.ones_like(recurrent_mask, dtype=bool)

    # ================================================================
    # Student-mismatch: shuffle recurrent connectome within each
    # cell-type-pair block (preserves per-pair distribution and sparsity)
    # ================================================================
    if shuffle_connectome:
        shuffle_rng = np.random.default_rng((seed if seed is not None else 0) + 2)
        print(
            "\nShuffling recurrent connectome within each cell-type-pair block "
            "(preserves distribution & sparsity)"
        )
        teacher_weights, recurrent_mask = shuffle_connectome_by_cell_type(
            teacher_weights, recurrent_mask, cell_type_indices, shuffle_rng
        )

    # ================================================================
    # Student-mismatch: shuffle nonzero weight values within each
    # cell-type-pair block while keeping the binary connectome fixed
    # ================================================================
    if shuffle_weights:
        weight_shuffle_rng = np.random.default_rng(
            (seed if seed is not None else 0) + 3
        )
        print(
            "\nShuffling recurrent weight values within the existing "
            "connectome (per cell-type pair; topology preserved)"
        )
        teacher_weights = shuffle_weights_within_connectome_by_cell_type(
            teacher_weights, recurrent_mask, cell_type_indices, weight_shuffle_rng
        )

    # ================================================================
    # Student-mismatch: weight noise on recurrent weights (per cell-type pair)
    # ================================================================
    if noise_frac > 0:
        noise_rng = np.random.default_rng((seed if seed is not None else 0) + 1)
        n_rec_types = int(cell_type_indices.max()) + 1
        print(
            f"\nApplying {noise_frac * 100:.1f}% multiplicative noise to "
            f"recurrent weights (per cell-type pair)..."
        )
        for in_type in range(n_rec_types):
            for out_type in range(n_rec_types):
                in_m = cell_type_indices == in_type
                out_m = cell_type_indices == out_type
                pair_mask = np.outer(in_m, out_m)
                pair_weights = teacher_weights[pair_mask]
                if (pair_weights != 0).sum() == 0:
                    continue
                teacher_weights[pair_mask] = apply_weight_noise(
                    pair_weights.reshape(-1),
                    noise_frac,
                    rng=noise_rng,
                    preserve_statistics=True,
                )

    # Save neuron classification so the analysis notebook can map back
    # to the original teacher indices.
    class_dir = output_dir / "targets"
    class_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        class_dir / "neuron_classification.npz",
        structural_visible_indices=structural_visible_indices,
        structural_hidden_indices=structural_hidden_indices,
        n_neurons_original=n_neurons_original,
        n_neurons=n_neurons,
        missing_unit_fraction=missing_unit_fraction,
        noise_frac=noise_frac,
        input_type=input_type,
        shuffle_connectome=shuffle_connectome,
        shuffle_weights=shuffle_weights,
        no_connectome=no_connectome,
    )

    # ================================================================
    # Load FF dataset (switch on input_type)
    # ================================================================

    if input_type == "exact_spikes":
        spike_dataset = ExactFFDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=training.recurrent_smoothing_tau,
        )
        n_feedforward = spike_dataset.n_input_neurons
    elif input_type == "constant":
        spike_dataset = HomogeneousPoissonFFDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
            firing_rate_override=firing_rate_hz,
            recurrent_smoothing_tau=training.recurrent_smoothing_tau,
        )
        n_feedforward = spike_dataset.n_input_neurons
    else:  # "latents"
        spike_dataset = LatentOUDataset(
            spike_data_path=input_dir / "spike_data.zarr",
            chunk_size=chunk_size,
            device=device,
            recurrent_smoothing_tau=training.recurrent_smoothing_tau,
        )
        n_feedforward = spike_dataset.n_latents

    batch_size = spike_dataset.batch_size

    print(f"\nLoaded {spike_dataset.num_chunks} chunks x {batch_size} batch size")
    print(f"  - {n_feedforward} feedforward (mitral) neurons")
    print(f"  - {n_neurons} recurrent neurons")

    # ================================================================
    # Build weight matrix and input structure
    # ================================================================

    n_total_inputs = n_feedforward + n_neurons

    # Feedforward weights: per-output-cell-type constant initialisation
    # Calibrate so total FF input per neuron ~ teacher's total FF input,
    # separately for excitatory and inhibitory output neurons.
    ff_weights = np.zeros((n_feedforward, n_neurons), dtype=np.float32)
    ff_mask = np.ones((n_feedforward, n_neurons), dtype=bool)  # full connectivity

    output_cell_type_names = recurrent.cell_types.names
    for type_idx, type_name in enumerate(output_cell_type_names):
        type_mask = cell_type_indices == type_idx
        mean_ff_type = teacher_ff_weights[:, type_mask].sum(axis=0).mean()
        constant_val = mean_ff_type / n_feedforward
        ff_weights[:, type_mask] = constant_val
        print(f"  FF weight init -> {type_name}: {constant_val:.6f}")

    print(
        f"  (teacher mean FF weight sum per neuron: "
        f"{teacher_ff_weights.sum(axis=0).mean():.4f})"
    )

    # is_feedforward_input: first n_feedforward are feedforward, rest are recurrent
    is_feedforward_input = np.zeros(n_total_inputs, dtype=bool)
    is_feedforward_input[:n_feedforward] = True

    # Cell type indices for inputs:
    # - Mitral inputs get cell type 0 ("mitral")
    # - Recurrent inputs get cell types offset by n_ff_cell_types
    ff_cell_type_indices = np.zeros(n_feedforward, dtype=np.int64)
    recurrent_cell_type_indices = cell_type_indices + n_ff_cell_types

    # ================================================================
    # RESUME PATH: Load saved initial state
    # ================================================================
    if resume_from is not None:
        header = "RESUMING — phase 2 from checkpoint"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        initial_state_dir = output_dir / "initial_state"
        print(f"\nLoading initial state from {initial_state_dir}...")

        initial_state = np.load(initial_state_dir / "network_structure.npz")
        perturbed_weights = initial_state["feedforward_weights"]
        concatenated_mask = initial_state["feedforward_connectivity"]
        concatenated_cell_type_indices = initial_state["feedforward_cell_type_indices"]

        print(f"  Loaded perturbed weights: {perturbed_weights.shape}")
        print(f"  Active connections: {concatenated_mask.sum():,}")

        # Load saved targets (recurrent scaling factors only)
        targets_dir = output_dir / "targets"
        saved_targets = np.load(targets_dir / "target_scaling_factors.npz")
        if "recurrent_scaling_factors" in saved_targets:
            target_scaling_factors_rec = saved_targets["recurrent_scaling_factors"]
        else:
            full_sf = saved_targets["feedforward_scaling_factors"]
            target_scaling_factors_rec = full_sf[n_ff_cell_types:, :]
        print(f"  Loaded target scaling factors: {target_scaling_factors_rec.shape}")

    # ================================================================
    # FRESH START PATH: Apply perturbation and save initial state
    # ================================================================
    else:
        header = "Perturbing Scaling Factors"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # Concatenate weights: [ff_weights, recurrent_weights]
        concatenated_weights = np.concatenate([ff_weights, teacher_weights], axis=0)
        concatenated_mask = np.concatenate([ff_mask, recurrent_mask], axis=0)
        concatenated_cell_type_indices = np.concatenate(
            [ff_cell_type_indices, recurrent_cell_type_indices]
        )

        # ============================================
        # Apply Scaling Factor Perturbation
        # ============================================
        # One lognormal scalar per (input_ct, output_ct) pair, applied to
        # both FF and recurrent blocks. Without an FF perturbation the
        # student starts at the teacher's mean FF weight per output type,
        # which the rank-1 init can match for free — that's a cheat.

        sigma = np.sqrt(weight_perturbation_variance)
        mu = -(sigma**2) / 2.0  # Ensures E[target] = 1

        target_scaling_factors_rec = np.random.lognormal(
            mean=mu, sigma=sigma, size=sf_recurrent.shape
        )
        target_scaling_factors_ff = np.random.lognormal(
            mean=mu, sigma=sigma, size=sf_feedforward.shape
        )

        # Perturbation is reciprocal of target (so target * perturbation = 1)
        perturbation_factors_rec = 1.0 / target_scaling_factors_rec
        perturbation_factors_ff = 1.0 / target_scaling_factors_ff

        perturbed_weights = concatenated_weights.copy()
        for input_idx in range(n_feedforward):
            for output_idx in range(n_neurons):
                output_type = cell_type_indices[output_idx]
                perturbed_weights[input_idx, output_idx] *= perturbation_factors_ff[
                    0, output_type
                ]
        for input_idx in range(n_feedforward, n_total_inputs):
            input_type = concatenated_cell_type_indices[input_idx]
            rec_type = input_type - n_ff_cell_types  # index into recurrent SF matrix
            for output_idx in range(n_neurons):
                output_type = cell_type_indices[output_idx]
                perturbed_weights[input_idx, output_idx] *= perturbation_factors_rec[
                    rec_type, output_type
                ]

        normalized_initial_rec = sf_recurrent / target_scaling_factors_rec
        normalized_initial_ff = sf_feedforward / target_scaling_factors_ff
        print(
            f"\nRecurrent scaling factors / target (should recover to 1.0):\n{normalized_initial_rec}"
        )
        print(
            f"FF scaling factors / target (should recover to 1.0):\n{normalized_initial_ff}"
        )

        # ============================================
        # Save Initial State (for resume capability)
        # ============================================

        initial_state_dir = output_dir / "initial_state"
        initial_state_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            initial_state_dir / "network_structure.npz",
            feedforward_weights=perturbed_weights,
            feedforward_connectivity=concatenated_mask,
            cell_type_indices=cell_type_indices,
            feedforward_cell_type_indices=concatenated_cell_type_indices,
            is_feedforward_input=is_feedforward_input,
        )

        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            targets_dir / "target_scaling_factors.npz",
            recurrent_scaling_factors=target_scaling_factors_rec,
            feedforward_scaling_factors=target_scaling_factors_ff,
        )

        print(f"\nSaved initial perturbed state to {initial_state_dir}")

    # ================================================================
    # Common setup (both paths converge here)
    # ================================================================

    print("\nFeedforward network setup:")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - Total inputs per neuron: {n_total_inputs}")
    print(f"    ({n_feedforward} mitral + {n_neurons} recurrent)")
    print(f"  - Weight matrix shape: {perturbed_weights.shape}")
    print(f"  - Active connections: {concatenated_mask.sum():,}")

    # ==================================================
    # Construct Combined Input Cell Types and Parameters
    # ==================================================

    # Concatenate cell params: feedforward + recurrent (with offset cell_ids)
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cell_params in recurrent_cell_params:
        offset_cell_params = cell_params.copy()
        offset_cell_params["cell_id"] = cell_params["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cell_params)

    # Concatenate synapse params with offset indices
    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for syn_params in recurrent_synapse_params:
        offset_syn_params = syn_params.copy()
        offset_syn_params["cell_id"] = syn_params["cell_id"] + n_ff_cell_types
        offset_syn_params["synapse_id"] = syn_params["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_syn_params)

    print("\nCombined parameters:")
    print(
        f"  - Cell types: {n_ff_cell_types} FF + {n_rec_cell_types} rec = {len(combined_cell_params_FF)}"
    )
    print(
        f"  - Synapse types: {n_ff_synapse_types} FF + {len(recurrent_synapse_params)} rec = {len(combined_synapse_params_FF)}"
    )

    # ======================
    # Setup DataLoader
    # ======================

    if missing_unit_fraction > 0:
        # Target spikes come from the full (original) teacher — subset them
        # to the structurally-visible neurons the student still has.
        visible_tensor = torch.from_numpy(structural_visible_indices).long()
        collate_fn = VisibleSubsetCollate(visible_tensor)
    else:
        collate_fn = FeedforwardCollate(dt=spike_dataset.dt)
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    # ==============================================
    # Common model keyword arguments
    # ==============================================

    # Shared kwargs for the model (projections passed per-phase).
    model_kwargs = dict(
        dt=spike_dataset.dt,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=gradient_config.get("surrgrad_scale", 5.0),
        batch_size=batch_size,
        track_variables=False,
    )

    # ==============================
    # Setup Loss Functions
    # ==============================

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=spike_dataset.dt,
        window_size=chunk_size,
        device=device,
    )

    van_rossum_rate_loss_fn = (
        VanRossumLoss(
            tau_rise=van_rossum_rate_tau_rise,
            tau_decay=van_rossum_rate_tau_decay,
            dt=spike_dataset.dt,
            window_size=chunk_size,
            device=device,
        )
        if (
            van_rossum_rate_tau_rise
            and van_rossum_rate_tau_decay
            and loss_weight_van_rossum_rate > 0
        )
        else None
    )

    # ================================================
    # Initialize wandb (shared across both phases)
    # ================================================

    wandb_run = None
    if wandb_config:
        wandb_init_config = {
            "simulation": simulation.model_dump(),
            "training": training.model_dump(),
            "hyperparameters": hyperparameters.model_dump(),
            "recurrent": recurrent.model_dump(),
            "feedforward": feedforward.model_dump(),
            "scaling_factors": scaling_factors,
            "output_dir": str(output_dir),
            "device": device,
            "n_feedforward": n_feedforward,
            "ff_rank": ff_rank,
            "gradient": gradient_config,
        }

        run_name = wandb_config.pop("name", None) or (
            output_dir.name if output_dir else "inferring-inputs-learn-ff-connectivity"
        )

        # Resume wandb run if resuming from checkpoint
        wandb_resume_kwargs = {}
        if resume_from is not None:
            wandb_dir = output_dir / "wandb"
            if wandb_dir.exists():
                run_dirs = sorted(wandb_dir.glob("run-*"))
                if run_dirs:
                    last_run = run_dirs[-1].name
                    run_id = last_run.split("-")[-1]
                    wandb_resume_kwargs = {"id": run_id, "resume": "must"}
                    print(f"  Resuming wandb run: {run_id}")

        wandb_run = wandb.init(
            name=run_name,
            config=wandb_init_config,
            dir=str(output_dir) if output_dir else None,
            **wandb_resume_kwargs,
            **wandb_config,
        )

    # Initialize AsyncLogger for disk logging
    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # Initialise FF weights from config
    best_perturbed_weights = perturbed_weights

    rec_cell_type_names = recurrent.cell_types.names  # ["excitatory", "inhibitory"]
    ff_cell_type_names = feedforward.cell_types.names[:n_ff_cell_types]  # ["mitral"]

    # ================================================
    # Create Plotting Function for Spike Comparison
    # ================================================

    def plot_generator(
        spikes,
        input_spikes,
        target_spikes,
        **kwargs,
    ):
        """Generate spike train comparison plot for subset of neurons."""
        n_plot = min(10, spikes.shape[2])

        interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes[0, :, i]

        cell_type_indices_plot = np.array([0, 1] * n_plot)
        cell_type_names_plot = ["Target", "Trained"]

        fig = plot_spike_trains(
            spikes=interleaved,
            dt=spike_dataset.dt,
            cell_type_indices=cell_type_indices_plot,
            cell_type_names=cell_type_names_plot,
            n_neurons_plot=2 * n_plot,
            fraction=1.0,
            random_seed=None,
            title=f"FF-connectivity: Target vs Trained (first {n_plot} neurons)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        return {"spike_comparison": fig}

    # ================================================
    # Create Stats Computer Function
    # ================================================

    dt = spike_dataset.dt
    ff_cell_type_names = list(feedforward.cell_types.names[:n_ff_cell_types])

    def stats_computer(snapshot):
        """Compute summary statistics for all neurons (student and teacher)."""
        student_spikes = snapshot["spikes"][0, :, :]
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        teacher_spikes = snapshot["target_spikes"][0, :n_timesteps, :]
        input_data = snapshot["input_spikes"][0, :n_timesteps, :]
        ff_spikes = input_data[:, :n_feedforward]  # (time, n_feedforward)

        # Compute firing rates
        student_firing_rates = student_spikes.sum(axis=0) / duration_s
        teacher_firing_rates = teacher_spikes.sum(axis=0) / duration_s

        stats = {
            "firing_rate/student_mean": float(student_firing_rates.mean()),
            "firing_rate/student_std": float(student_firing_rates.std()),
            "firing_rate/student_min": float(student_firing_rates.min()),
            "firing_rate/student_max": float(student_firing_rates.max()),
            "firing_rate/teacher_mean": float(teacher_firing_rates.mean()),
            "firing_rate/teacher_std": float(teacher_firing_rates.std()),
            "firing_rate/teacher_min": float(teacher_firing_rates.min()),
            "firing_rate/teacher_max": float(teacher_firing_rates.max()),
        }

        # Cell-type-specific firing rates
        output_cell_type_names = recurrent.cell_types.names
        for type_idx, type_name in enumerate(output_cell_type_names):
            type_mask = cell_type_indices == type_idx
            if type_mask.sum() > 0:
                student_type_rates = student_firing_rates[type_mask]
                stats[f"firing_rate/student_{type_name}_mean"] = float(
                    student_type_rates.mean()
                )
                stats[f"firing_rate/student_{type_name}_std"] = float(
                    student_type_rates.std()
                )
                teacher_type_rates = teacher_firing_rates[type_mask]
                stats[f"firing_rate/teacher_{type_name}_mean"] = float(
                    teacher_type_rates.mean()
                )
                stats[f"firing_rate/teacher_{type_name}_std"] = float(
                    teacher_type_rates.std()
                )

        current_sf = snapshot["scaling_factors_FF"]
        rec_cell_type_names = recurrent.cell_types.names

        # Scaling factor tracking (recurrent only) — skipped for the
        # no-connectome control, where SF are not optimised.
        if not no_connectome:
            target_sf = target_scaling_factors_rec
            for source_idx in range(target_sf.shape[0]):
                source_name = rec_cell_type_names[source_idx]
                sf_row = source_idx + n_ff_cell_types
                for target_idx in range(target_sf.shape[1]):
                    target_name = rec_cell_type_names[target_idx]
                    synapse_name = f"{source_name}_to_{target_name}"
                    target_val = target_sf[source_idx, target_idx]
                    if target_val != 0:
                        normalized_value = current_sf[sf_row, target_idx] / target_val
                    else:
                        normalized_value = current_sf[sf_row, target_idx]
                    stats[f"scaling_factors/{synapse_name}_value"] = float(
                        normalized_value
                    )
                    stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        # ======================
        # Per-cell-type-pair weight stats
        # ======================
        weights_ff = snapshot["weights_FF"]
        # weights_FF already includes the per-pair scaling factor (each
        # ScalingFactorProjection.forward() returns exp(log_sf) * connectome),
        # so no additional SF multiplication is needed here.
        ff_w_effective = weights_ff[:n_feedforward, :]

        # Feedforward block: input cell type "mitral" -> recurrent output types.
        # Compare student's effective weights against the teacher's effective
        # weights (× teacher's configured SF) for fairness.
        for out_idx, out_name in enumerate(rec_cell_type_names):
            out_mask = cell_type_indices == out_idx
            if out_mask.sum() == 0:
                continue
            for in_idx, in_name in enumerate(ff_cell_type_names):
                w_pair = ff_w_effective[:, out_mask]
                t_pair = teacher_ff_pristine[:, out_mask] * float(
                    sf_feedforward[in_idx, out_idx]
                )
                synapse_name = f"{in_name}_to_{out_name}"
                stats[f"ff_weights/{synapse_name}_mean"] = float(w_pair.mean())
                stats[f"ff_weights/{synapse_name}_min"] = float(w_pair.min())
                stats[f"ff_weights/{synapse_name}_max"] = float(w_pair.max())
                stats[f"ff_weights/{synapse_name}_teacher_mean"] = float(t_pair.mean())
                stats[f"ff_weights/{synapse_name}_teacher_min"] = float(t_pair.min())
                stats[f"ff_weights/{synapse_name}_teacher_max"] = float(t_pair.max())

        rec_weights = weights_ff[n_feedforward:, :]

        # Recurrent block: only logged for the no-connectome control,
        # where the recurrent matrix is itself a learned parameter.
        # Teacher's effective recurrent weights = pristine × teacher rec SF.
        if no_connectome:
            for out_idx, out_name in enumerate(rec_cell_type_names):
                out_mask = cell_type_indices == out_idx
                if out_mask.sum() == 0:
                    continue
                for in_idx, in_name in enumerate(rec_cell_type_names):
                    in_mask = cell_type_indices == in_idx
                    if in_mask.sum() == 0:
                        continue
                    w_pair = rec_weights[in_mask][:, out_mask]
                    t_pair = teacher_rec_pristine[in_mask][:, out_mask] * float(
                        sf_recurrent[in_idx, out_idx]
                    )
                    synapse_name = f"{in_name}_to_{out_name}"
                    stats[f"recurrent_weights/{synapse_name}_mean"] = float(
                        w_pair.mean()
                    )
                    stats[f"recurrent_weights/{synapse_name}_min"] = float(w_pair.min())
                    stats[f"recurrent_weights/{synapse_name}_max"] = float(w_pair.max())
                    stats[f"recurrent_weights/{synapse_name}_teacher_mean"] = float(
                        t_pair.mean()
                    )
                    stats[f"recurrent_weights/{synapse_name}_teacher_min"] = float(
                        t_pair.min()
                    )
                    stats[f"recurrent_weights/{synapse_name}_teacher_max"] = float(
                        t_pair.max()
                    )

        # ======================
        # Feedforward vs recurrent drive comparison
        # ======================
        rec_sf = current_sf[n_ff_cell_types:, :]
        exc_outgoing_w = np.zeros(int(exc_mask.sum()), dtype=np.float64)
        for target_type in range(rec_sf.shape[1]):
            target_mask = cell_type_indices == target_type
            exc_to_type = rec_weights[exc_mask][:, target_mask].sum(axis=1)
            exc_outgoing_w += exc_to_type * rec_sf[0, target_type]

        # FF: total spike count * outgoing weight per FF neuron
        ff_spike_counts = ff_spikes.sum(axis=0)  # (n_feedforward,)
        ff_outgoing_w = ff_w_effective.sum(axis=1)  # (n_feedforward,)
        ff_drive = (ff_spike_counts * ff_outgoing_w).sum()

        # Rec exc: spike count * outgoing weight per exc neuron
        rec_spike_counts = teacher_spikes.sum(axis=0)
        rec_exc_drive = (rec_spike_counts[exc_mask] * exc_outgoing_w).sum()

        total_drive = ff_drive + rec_exc_drive
        stats["drive/ff_fraction"] = float(ff_drive / max(total_drive, 1e-10))
        stats["drive/rec_exc_fraction"] = float(rec_exc_drive / max(total_drive, 1e-10))
        stats["drive/ratio_rec_exc_over_ff"] = float(
            rec_exc_drive / max(ff_drive, 1e-10)
        )

        return stats

    # ========================================
    # Common setup for gradient-based training
    # ========================================

    # Update feedforward cell type names for trainer labeling
    feedforward.cell_types.names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )

    # Gradient phase uses Van Rossum loss(es)
    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}
    if van_rossum_rate_loss_fn is not None:
        gradient_loss_functions["van_rossum_rate"] = van_rossum_rate_loss_fn
        gradient_loss_weights["van_rossum_rate"] = loss_weight_van_rossum_rate

    phase2_epochs = gradient_config.get("epochs", 100)
    phase2_chunks = phase2_epochs * spike_dataset.num_chunks

    # Resume detection: only phase-2 checkpoints are produced.
    if resume_from is not None:
        _check_resume_checkpoint(output_dir)

    # Recurrent ScalingFactorProjection starts at SF=1.0 (no phase-1 prior).
    best_rec_sf = np.ones((n_rec_cell_types, n_rec_cell_types), dtype=np.float32)

    # ================================================
    # Build projection registry
    # ================================================
    # FF pairs: LowRank, initialised via truncated SVD of the per-pair FF
    # block (rank-1 in practice, since FF init is uniform per output type).
    # Recurrent pairs: ScalingFactor with init_sf=1.0 (baseline) or
    # FullRank initialised from the perturbed rec block (no-connectome).
    print(f"\nBuilding Phase 2 projections (rank={ff_rank} for FF)...")

    phase2_projections: dict = {}
    rec_idx_per_ct = {
        name: np.flatnonzero(cell_type_indices == ct_id)
        for ct_id, name in enumerate(rec_cell_type_names)
    }
    ff_block_baked = best_perturbed_weights[:n_feedforward, :]
    rec_block = best_perturbed_weights[n_feedforward:, :]

    for tgt_name in rec_cell_type_names:
        tgt_idx = rec_idx_per_ct[tgt_name]
        # FF pair (mitral, tgt): low-rank from SVD of baked sub-block
        ff_pair_block = ff_block_baked[:, tgt_idx]  # (n_feedforward, n_tgt)
        log_block = np.log(np.maximum(ff_pair_block, 1e-8)).astype(np.float32)
        # Mask is all-ones for FF (full mitral connectivity).
        log_block_t = torch.from_numpy(log_block)
        rank_eff = min(ff_rank, *log_block_t.shape)
        U_full, S_full, Vh_full = torch.linalg.svd(log_block_t, full_matrices=False)
        # Zero out singular values that are float32-noise relative to the
        # leading one. The FF block is effectively rank-1 (uniform per
        # output type), so without this the SVD reconstruction picks up
        # a spurious ±0.5-in-log-space perturbation from tail singular
        # values before any optimiser step has run.
        S_eff = S_full[:rank_eff].clone()
        S_eff[S_eff < S_full[0] * 1e-4] = 0.0
        U_init = (U_full[:, :rank_eff] * S_eff.sqrt().unsqueeze(0)).numpy()
        V_init = (S_eff.sqrt().unsqueeze(1) * Vh_full[:rank_eff, :]).numpy()
        var_explained = float(
            (S_full[:rank_eff] ** 2).sum() / max((S_full**2).sum(), 1e-10)
        )
        print(
            f"  FF (mitral->{tgt_name}): rank {rank_eff}, "
            f"top-{rank_eff} variance explained {var_explained:.1%}"
        )
        phase2_projections[("mitral", tgt_name)] = LowRankProjection(
            n_source=n_feedforward,
            n_target=len(tgt_idx),
            rank=rank_eff,
            U_init=U_init,
            V_init=V_init,
            mask=None,  # full mitral connectivity
        )

    for src_name in rec_cell_type_names:
        src_idx = rec_idx_per_ct[src_name]
        for tgt_name in rec_cell_type_names:
            tgt_idx = rec_idx_per_ct[tgt_name]
            rec_pair_block = rec_block[np.ix_(src_idx, tgt_idx)]
            if no_connectome:
                # Trainable per-connection rec weights (full connectivity).
                phase2_projections[(src_name, tgt_name)] = FullRankProjection(
                    init_weights=rec_pair_block.astype(np.float32),
                    mask=None,
                )
            else:
                # Trainable per-pair recurrent scaling factor (init = 1.0).
                src_id = next(
                    i for i, n in enumerate(rec_cell_type_names) if n == src_name
                )
                tgt_id = next(
                    i for i, n in enumerate(rec_cell_type_names) if n == tgt_name
                )
                init_sf = float(best_rec_sf[src_id, tgt_id])
                # Connectome from rec_block (unscaled in baseline path).
                phase2_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                    connectome=rec_pair_block.astype(np.float32),
                    init_sf=init_sf,
                )

    # Phase 2 surrgrad scale set on model_kwargs before construction.
    p2_surrgrad = gradient_config.get("surrgrad_scale", 5.0)
    phase2_model_kwargs = {**model_kwargs, "surrgrad_scale": p2_surrgrad}
    model = FeedforwardConductanceLIFNetwork(
        **phase2_model_kwargs,
        projections=phase2_projections,
    )
    model.to(device)

    print("\nPhase 2 model initialized:")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - FF weight rank: {ff_rank}")
    print(
        f"  - Recurrent mode: {'weights (full-rank)' if no_connectome else 'scaling factors'}"
    )

    # ==============================
    # Setup Optimizer for Phase 2
    # ==============================

    p2_lr_weights = gradient_config.get("lr_weights", 1e-3)
    p2_lr_scaling = gradient_config.get("lr_scaling", 5e-2)
    p2_lr_min_weights = gradient_config.get("lr_min_weights", None)
    p2_lr_min_scaling = gradient_config.get("lr_min_scaling", None)
    p2_grad_clip_weights = gradient_config.get("grad_clip_weights", 50.0)
    p2_grad_clip_scaling = gradient_config.get("grad_clip_scaling", 0.5)

    # Group parameters by projection type. U/V from LowRank and log_weights
    # from FullRank go into the "weights" group. log_sf from ScalingFactor
    # goes into the "scaling" group.
    weights_params: list[nn.Parameter] = []
    scaling_params: list[nn.Parameter] = []
    for proj in model.projections.values():
        if isinstance(proj, LowRankProjection):
            weights_params.append(proj.U)
            weights_params.append(proj.V)
        elif isinstance(proj, FullRankProjection):
            weights_params.append(proj.log_weights)
        elif isinstance(proj, ScalingFactorProjection):
            scaling_params.append(proj.log_sf)

    param_groups = [{"params": weights_params, "lr": p2_lr_weights}]
    print(
        f"  LR (weights/low-rank/full-rank): {p2_lr_weights}, "
        f"{len(weights_params)} param tensors"
    )
    if scaling_params:
        param_groups.append({"params": scaling_params, "lr": p2_lr_scaling})
        print(
            f"  LR (scaling factors): {p2_lr_scaling}, "
            f"{len(scaling_params)} param tensors"
        )

    optimiser = torch.optim.Adam(param_groups, betas=(beta1, beta2))

    # Per-group cosine annealing for Phase 2
    p2_lr_pairs = []  # (lr_init, lr_min) per param group
    for group in param_groups:
        lr_init = group["lr"]
        if lr_init == p2_lr_weights and p2_lr_min_weights is not None:
            p2_lr_pairs.append((lr_init, p2_lr_min_weights))
        elif lr_init == p2_lr_scaling and p2_lr_min_scaling is not None:
            p2_lr_pairs.append((lr_init, p2_lr_min_scaling))
        else:
            p2_lr_pairs.append((lr_init, lr_init))

    if any(lr_min < lr_init for lr_init, lr_min in p2_lr_pairs):

        def make_cosine_lambda(lr_init, lr_min, n_epochs):
            frac = lr_min / lr_init

            def cosine_lr_lambda(epoch):
                return (
                    frac + (1 - frac) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
                )

            return cosine_lr_lambda

        lambdas = [make_cosine_lambda(li, lm, phase2_epochs) for li, lm in p2_lr_pairs]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambdas)
        print(f"  LR schedule: cosine annealing over {phase2_epochs} epochs")
        for li, lm in p2_lr_pairs:
            print(f"    {li:.1e} -> {lm:.1e}")
    else:
        scheduler = None

    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    def _clip_grads_per_group(m):
        if weights_params:
            torch.nn.utils.clip_grad_norm_(
                weights_params, max_norm=p2_grad_clip_weights
            )
        if scaling_params:
            torch.nn.utils.clip_grad_norm_(
                scaling_params, max_norm=p2_grad_clip_scaling
            )

    print(f"  Surrogate gradient scale: {p2_surrgrad} (fixed)")

    epoch_callbacks = []
    total_gradient_chunks = phase2_chunks

    phase2_pbar = tqdm(
        range(total_gradient_chunks),
        desc="Training (low-rank)",
        unit="chunk",
        total=total_gradient_chunks,
    )

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions=gradient_loss_functions,
        loss_weights=gradient_loss_weights,
        device=device,
        num_epochs=total_gradient_chunks,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_clip_fn=_clip_grads_per_group,
        wandb_config=None,
        progress_bar=phase2_pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        connectome_mask=None,
        chunks_per_data_epoch=spike_dataset.num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
        epoch_callbacks=epoch_callbacks,
    )
    trainer.metrics_logger = metrics_logger
    if wandb_run is not None:
        trainer.wandb_logger = wandb_run
        # Skip wandb.watch — it causes torch.compile graph breaks and
        # recompilations on every parameter name. Gradient norms are
        # already logged via _extract_current_gradients.
        wandb.define_metric("epoch")

    # Handle checkpoint resuming
    if resume_from is not None:
        phase2_checkpoint = sorted((output_dir / "checkpoints").glob("*.pt"))[-1]
        print(f"\nResuming from checkpoint: {phase2_checkpoint}")
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path=phase2_checkpoint,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )
        trainer.set_checkpoint_state(start_epoch, best_loss)
        phase2_pbar.n = start_epoch
        if wandb_run is not None:
            wandb.log(
                {"epoch": start_epoch / spike_dataset.num_chunks},
                step=start_epoch,
            )
        phase2_pbar.refresh()

    # =================
    # Run Phase 2
    # =================

    print(f"\nStarting Phase 2 from chunk {trainer.current_epoch}...")
    print(f"  Epochs: {phase2_epochs}")
    print(f"  Batch size: {batch_size}")
    print(
        f"  Training {n_neurons} neurons: low-rank FF weights (rank {ff_rank}) + recurrent scaling factors"
    )

    model.reset_state(batch_size=batch_size)
    model.track_variables = False

    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    header = f"Training complete! Best loss achieved: {best_loss:.6f}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Save final state
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    # `model.weights_FF` already returns the effective weight matrix —
    # `_assemble_weights_FF` calls `proj()` on each projection, and
    # `ScalingFactorProjection.forward()` returns `exp(log_sf) * connectome`,
    # so SF is already baked in for scaling-factor projections. The FF
    # (LowRank) block has no per-pair SF (mitral row of scaling_factors_FF
    # stays at 1.0), so it needs no adjustment either.
    final_weights = model.weights_FF.detach().cpu().numpy()
    final_sf = model.scaling_factors_FF.detach().cpu().numpy()

    print("\nSaving final network structure (scaling factors baked into weights)...")
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=final_weights,
        recurrent_weights=final_weights[n_feedforward:, :],
        feedforward_connectivity=concatenated_mask,
        recurrent_connectivity=recurrent_mask,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=np.ones_like(final_sf),
        is_feedforward_input=is_feedforward_input,
    )

    # Close shared loggers
    metrics_logger.close()

    if wandb_run is not None:
        wandb.finish()

    print(f"\nCheckpoints: {output_dir / 'checkpoints'}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"Metrics: {output_dir / 'training_metrics.csv'}")
    print(f"Final state: {final_state_dir / 'network_structure.npz'}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
