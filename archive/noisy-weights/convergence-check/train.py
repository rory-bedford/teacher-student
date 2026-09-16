"""
Training all neurons with feedforward dynamics to match target activity.

This script is identical to train_feedforward.py, except that log-normal noise
is added to the weight matrix before training. The noise is multiplicative:
    w_noisy = w * exp(noise_frac * N(0,1) - noise_frac^2 / 2)

The multiplier has mean 1 (mean-preserving) and is strictly positive, so
non-negativity of weights is guaranteed.

Noise is applied SEPARATELY to each cell-type pair (e.g., exc->exc, inh->exc,
mitral->inh), with an affine transformation applied per pair to preserve
the mean and std of that specific connection type. This maintains E/I balance
and other dynamics-critical properties.

TODO: Log teacher firing rate metrics (mean, std, min, max) to the training CSV.
      Currently we only log learned firing rates, which makes post-hoc analysis harder
      since we need to load the zarr file to compute teacher rates.
"""

import gc

import numpy as np
import matplotlib.pyplot as plt
from connectome_snns.dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    feedforward_collate_fn,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    make_chunked_ff_projections,
)
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
from connectome_snns.training_utils.losses import (
    VanRossumLoss,
    FiringRateLoss,
    AsymmetricSilencePenalty,
)
from connectome_snns.training_utils import load_checkpoint, AsyncLogger
from connectome_snns.configs import (
    DATALOADER_KWARGS,
    StudentSimulationConfig,
    StudentTrainingConfig,
    StudentHyperparameters,
)
from connectome_snns.configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from connectome_snns.snn_runners import SNNTrainer, EvolutionarySearch
import toml
import wandb
from connectome_snns.visualization.neuronal_dynamics import plot_spike_trains


def apply_weight_noise(weights, noise_frac, rng=None, preserve_statistics=True):
    """Apply multiplicative log-normal noise to weights with statistics preservation.

    Applies noise: w * exp(noise_frac * N(0,1) - noise_frac^2 / 2), then uses an
    affine transformation to exactly match the original mean and std of non-zero weights.
    The multiplier has E[m] = 1 and is strictly positive, guaranteeing non-negativity.

    Args:
        weights: Weight matrix (numpy array, may be sparse with zeros)
        noise_frac: Fraction of weight magnitude for noise (e.g., 0.1 for 10%)
        rng: Optional numpy random generator for reproducibility
        preserve_statistics: If True, apply affine transform to match original
            mean and std of non-zero weights (default: True)

    Returns:
        Noisy weights with same shape as input. Zero weights remain zero.
        Non-zero weights have the same mean and std as the original.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Identify non-zero weights (sparse connectivity)
    nonzero_mask = weights != 0

    # Original statistics on non-zero weights only
    orig_mean = weights[nonzero_mask].mean()
    orig_std = weights[nonzero_mask].std()

    # Log-normal multiplicative noise: w * exp(noise_frac * N(0,1) - noise_frac^2 / 2)
    # E[multiplier] = 1 (mean-preserving), multiplier > 0 always
    multiplier = np.exp(
        noise_frac * rng.standard_normal(weights.shape) - noise_frac**2 / 2
    )

    noisy_weights = weights * multiplier

    if preserve_statistics and orig_std > 0:
        # Compute noisy statistics on non-zero positions only
        noisy_nz = noisy_weights[nonzero_mask]
        noisy_mean = noisy_nz.mean()
        noisy_std = noisy_nz.std()

        # Affine transform to exactly match original mean and std
        # new = orig_mean + (old - old_mean) * (orig_std / old_std)
        if noisy_std > 0:
            noisy_weights[nonzero_mask] = orig_mean + (noisy_nz - noisy_mean) * (
                orig_std / noisy_std
            )
            # Clip to non-negative: for high-CV pairs the affine rescale can push
            # near-zero weights just below 0 by finite-sample chance. Magnitudes
            # are negligible (< 1e-4) relative to typical weights (~0.1).
            noisy_weights[nonzero_mask] = np.maximum(noisy_weights[nonzero_mask], 0)

    return noisy_weights


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train all neurons to match target spike train activity using feedforward dynamics.

    Args:
        input_dir (Path): Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir (Path): Directory where training outputs will be saved
        params_file (Path): Path to the file containing training parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to checkpoint to resume from
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)

    # Load configuration sections
    simulation = StudentSimulationConfig(**data["simulation"])
    training = StudentTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})
    cma_es_config = data.get("cma_es", {})

    # Load weight noise configuration
    weight_noise_config = data.get("weight_noise", {})
    noise_frac = weight_noise_config.get("noise_frac", 0.0)

    # Extract parameters into plain Python variables
    chunk_size = simulation.chunk_size
    seed = simulation.seed
    epochs = training.epochs
    chunks_per_update = training.chunks_per_update
    log_interval = training.log_interval
    checkpoint_interval = training.checkpoint_interval
    plot_size = training.plot_size
    mixed_precision = training.mixed_precision
    grad_norm_clip = hyperparameters.grad_norm_clip
    burn_in_chunks = getattr(training, "burn_in_chunks", 0)
    weight_perturbation_variance = training.weight_perturbation_variance
    optimisable = training.optimisable
    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
    learning_rate = hyperparameters.learning_rate
    surrgrad_scale = hyperparameters.surrgrad_scale
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay
    loss_weight_van_rossum = hyperparameters.loss_weight.van_rossum
    loss_weight_firing_rate = getattr(hyperparameters.loss_weight, "firing_rate", 0.0)
    loss_weight_silence_penalty = getattr(
        hyperparameters.loss_weight, "silence_penalty", 0.0
    )

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # Get cell and synapse parameters (needed for both paths)
    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    n_ff_cell_types = len(feedforward_cell_params)
    n_rec_cell_types = len(recurrent_cell_params)
    n_ff_synapse_types = len(feedforward_synapse_params)

    # Get base scaling factors from config
    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # ================================================================
    # RESUME PATH: Load saved initial state (perturbed weights + targets)
    # ================================================================
    if resume_from is not None:
        print("\n" + "=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)

        # Load the saved initial state (perturbed weights from original run)
        initial_state_dir = output_dir / "initial_state"
        print(f"\nLoading initial state from {initial_state_dir}...")

        initial_state = np.load(initial_state_dir / "network_structure.npz")
        perturbed_weights = initial_state["feedforward_weights"]
        concatenated_mask = initial_state["feedforward_connectivity"]
        cell_type_indices = initial_state["cell_type_indices"]
        concatenated_cell_type_indices = initial_state["feedforward_cell_type_indices"]

        n_neurons = perturbed_weights.shape[1]
        n_total_inputs = perturbed_weights.shape[0]

        print(f"  ✓ Loaded perturbed weights: {perturbed_weights.shape}")
        print(f"  ✓ Active connections: {concatenated_mask.sum():,}")

        # Load saved targets
        targets_dir = output_dir / "targets"
        saved_targets = np.load(targets_dir / "target_scaling_factors.npz")
        target_scaling_factors_FF = saved_targets["feedforward_scaling_factors"]
        print(f"  ✓ Loaded target scaling factors: {target_scaling_factors_FF.shape}")

    # ================================================================
    # FRESH START PATH: Apply noise, perturbation, and save initial state
    # ================================================================
    else:
        print("\n" + "=" * 60)
        print("FRESH START: Applying perturbations")
        print("=" * 60)

        # Create RNG for weight noise
        weight_noise_rng = np.random.default_rng(seed)

        # Load original network structure from teacher
        network_structure = np.load(input_dir / "network_structure.npz")

        weights = network_structure["recurrent_weights"]
        feedforward_weights = network_structure["feedforward_weights"]
        cell_type_indices = network_structure["cell_type_indices"]
        feedforward_cell_type_indices = network_structure[
            "feedforward_cell_type_indices"
        ]
        recurrent_mask = network_structure["recurrent_connectivity"]
        feedforward_mask = network_structure["feedforward_connectivity"]

        n_neurons = weights.shape[0]
        n_feedforward = feedforward_weights.shape[0]
        n_total_inputs = n_feedforward + n_neurons

        # Concatenate weights and masks
        concatenated_weights = np.concatenate([feedforward_weights, weights], axis=0)
        concatenated_mask = np.concatenate([feedforward_mask, recurrent_mask], axis=0)

        # Build concatenated cell type indices
        concatenated_cell_type_indices = np.concatenate(
            [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
        )
        n_input_types = concatenated_cell_type_indices.max() + 1
        n_output_types = cell_type_indices.max() + 1

        # ===============================================
        # Apply Gaussian Noise to Weights (Per Cell-Type Pair)
        # ===============================================

        if noise_frac > 0:
            print(
                f"\nApplying {noise_frac * 100:.1f}% multiplicative Gaussian noise (per cell-type pair)..."
            )

            original_weights = concatenated_weights.copy()

            # Apply noise separately to each (input_type, output_type) pair
            pair_stats = []
            for in_type in range(n_input_types):
                for out_type in range(n_output_types):
                    in_mask = concatenated_cell_type_indices == in_type
                    out_mask = cell_type_indices == out_type
                    pair_mask = np.outer(in_mask, out_mask)

                    pair_weights = concatenated_weights[pair_mask]
                    nonzero = pair_weights != 0

                    if nonzero.sum() == 0:
                        continue

                    orig_mean = pair_weights[nonzero].mean()
                    orig_std = pair_weights[nonzero].std()

                    noisy_pair = apply_weight_noise(
                        pair_weights.reshape(-1),
                        noise_frac,
                        rng=weight_noise_rng,
                        preserve_statistics=True,
                    )

                    concatenated_weights[pair_mask] = noisy_pair

                    noisy_nz = noisy_pair[nonzero]
                    pair_stats.append(
                        {
                            "in_type": in_type,
                            "out_type": out_type,
                            "n_weights": nonzero.sum(),
                            "orig_mean": orig_mean,
                            "orig_std": orig_std,
                            "noisy_mean": noisy_nz.mean(),
                            "noisy_std": noisy_nz.std(),
                        }
                    )

            # Print per-pair statistics
            print("\n  Per cell-type pair statistics:")
            for ps in pair_stats:
                print(
                    f"    [{ps['in_type']}→{ps['out_type']}] n={ps['n_weights']:,}: "
                    f"μ {ps['orig_mean']:.4f}→{ps['noisy_mean']:.4f}, "
                    f"σ {ps['orig_std']:.4f}→{ps['noisy_std']:.4f}"
                )

            # Global stats
            nonzero_mask = concatenated_weights != 0
            original_weights_nz = original_weights[nonzero_mask]
            noisy_weights_nz = concatenated_weights[nonzero_mask]
            print(
                f"\n  - Global original (non-zero): mean={original_weights_nz.mean():.6f}, std={original_weights_nz.std():.6f}"
            )
            print(
                f"  - Global noisy (non-zero):    mean={noisy_weights_nz.mean():.6f}, std={noisy_weights_nz.std():.6f}"
            )

            # Create and save weight noise comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            ax = axes[0]
            max_points = 50000
            if len(original_weights_nz) > max_points:
                idx = np.random.default_rng(42).choice(
                    len(original_weights_nz), max_points, replace=False
                )
            else:
                idx = np.arange(len(original_weights_nz))

            ax.scatter(original_weights_nz[idx], noisy_weights_nz[idx], alpha=0.2, s=1)
            lims = [
                min(0, noisy_weights_nz.min()),
                np.percentile(
                    np.concatenate([original_weights_nz, noisy_weights_nz]), 95
                ),
            ]
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Original Weight")
            ax.set_ylabel("Noisy Weight")
            ax.set_aspect("equal")

            corr = np.corrcoef(original_weights_nz, noisy_weights_nz)[0, 1]
            stats_text = (
                f"Original: μ={original_weights_nz.mean():.6f}, σ={original_weights_nz.std():.6f}\n"
                f"Noisy:    μ={noisy_weights_nz.mean():.6f}, σ={noisy_weights_nz.std():.6f}\n"
                f"Corr: r={corr:.4f}"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.set_title("Original vs Noisy Weights")

            ax = axes[1]
            bins = np.linspace(
                min(0, noisy_weights_nz.min()),
                np.percentile(
                    np.concatenate([original_weights_nz, noisy_weights_nz]), 99
                ),
                100,
            )
            ax.hist(
                original_weights_nz,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Original",
                color="blue",
            )
            ax.hist(
                noisy_weights_nz,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Noisy",
                color="orange",
            )
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Weight")
            ax.set_ylabel("Density")
            ax.set_title("Distribution Comparison")
            ax.legend()

            fig.suptitle(
                f"Weight Noise: {noise_frac * 100:.1f}% (Per Cell-Type Pair, Clip [0, ∞) + Affine Rescale)",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()

            plot_path = output_dir / "weight_noise_comparison.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  - Saved weight noise comparison plot to {plot_path}")
        else:
            print("\nNo weight noise applied (noise_frac = 0)")

        # ============================================
        # Apply Scaling Factor Perturbation
        # ============================================

        # Sample TARGET scaling factors from log-normal distribution
        sigma = np.sqrt(weight_perturbation_variance)
        mu = -(sigma**2) / 2.0  # Ensures E[target] = 1

        target_scaling_factors_FF = np.random.lognormal(
            mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
        )

        # Perturbation is reciprocal of target (so target * perturbation = 1)
        perturbation_factors = 1.0 / target_scaling_factors_FF

        # Apply perturbation to weights
        perturbed_weights = concatenated_weights.copy()
        for input_idx in range(n_total_inputs):
            input_type = concatenated_cell_type_indices[input_idx]
            for output_idx in range(n_neurons):
                output_type = cell_type_indices[output_idx]
                perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                    input_type, output_type
                ]

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
        )

        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            targets_dir / "target_scaling_factors.npz",
            feedforward_scaling_factors=target_scaling_factors_FF,
        )

        print(f"\n✓ Saved initial perturbed state to {initial_state_dir}")
        print(f"✓ Saved target scaling factors to {targets_dir}")

    # ================================================================
    # Common setup (both paths converge here)
    # ================================================================

    print("\n✓ Feedforward network setup:")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - Total inputs per neuron: {n_total_inputs}")
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

    print("\n✓ Combined parameters:")
    print(
        f"  - Cell types: {n_ff_cell_types} FF + {n_rec_cell_types} rec = {len(combined_cell_params_FF)}"
    )
    print(
        f"  - Synapse types: {n_ff_synapse_types} FF + {len(recurrent_synapse_params)} rec = {len(combined_synapse_params_FF)}"
    )

    # ======================
    # Load Dataset from Disk
    # ======================

    # Load precomputed spike data from zarr
    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )

    batch_size = spike_dataset.batch_size

    print(f"\n✓ Loaded {spike_dataset.num_chunks} chunks × {batch_size} batch size")

    # DataLoader with custom collate function for concatenation
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=feedforward_collate_fn,
    )

    # ==============================================
    # Common Model Keyword Arguments
    # ==============================================

    model_kwargs = dict(
        dt=spike_dataset.dt,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    # Combined input cell-type names (for projection keys): FF names + rec names.
    combined_input_cell_type_names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )
    rec_output_cell_type_names = recurrent.cell_types.names

    # Split perturbed_weights into the FF block (rows for FF inputs) and
    # the recurrent block (rows for recurrent inputs, treated as FF rows by
    # the chunked-FF model). ``make_chunked_ff_projections`` and the frozen
    # builder expect them passed separately.
    n_feedforward_inputs = int((concatenated_cell_type_indices < n_ff_cell_types).sum())
    perturbed_ff_block = perturbed_weights[:n_feedforward_inputs, :]
    perturbed_rec_block = perturbed_weights[n_feedforward_inputs:, :]
    ff_cell_type_indices_only = concatenated_cell_type_indices[:n_feedforward_inputs]

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

    firing_rate_loss_fn = FiringRateLoss(dt=spike_dataset.dt)
    silence_penalty_fn = AsymmetricSilencePenalty()

    cma_loss_weights = {
        "van_rossum": loss_weight_van_rossum,
        "firing_rate": loss_weight_firing_rate,
        "silence_penalty": loss_weight_silence_penalty,
    }

    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}

    # ================================================
    # Initialize Logging (shared across CMA-ES and gradient training)
    # ================================================

    # Cell type names for logging
    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names
    output_cell_type_names = recurrent.cell_types.names

    # Initialize AsyncLogger for disk logging
    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # Initialize wandb if config provided
    wandb_logger = None
    if wandb_config:
        wandb_config_dict = {
            **wandb_config,
            "config": {
                "simulation": simulation.model_dump(),
                "training": training.model_dump(),
                "hyperparameters": hyperparameters.model_dump(),
                "recurrent": recurrent.model_dump(),
                "feedforward": feedforward.model_dump(),
                "scaling_factors": scaling_factors,
                "weight_noise": weight_noise_config,
                "output_dir": str(output_dir),
                "device": device,
            },
        }
        run_name = wandb_config_dict.pop("name", None) or output_dir.name
        user_config = wandb_config_dict.pop("config", {})
        wandb_init_kwargs = {
            "name": run_name,
            "config": user_config,
            "dir": str(output_dir),
            **wandb_config_dict,
        }
        wandb_logger = wandb.init(**wandb_init_kwargs)
        wandb.define_metric("epoch")

    # ================================================
    # Phase 1: CMA-ES Evolutionary Search
    # ================================================

    best_scaling_factors_FF = concatenated_scaling_factors

    # Check if CMA-ES already completed (resume loads saved results)
    cma_state_path = output_dir / "cma_es_state" / "scaling_factors.npz"
    if resume_from is not None and cma_state_path.exists():
        print("\nLoading CMA-ES results from previous run...")
        cma_saved = np.load(cma_state_path)
        best_scaling_factors_FF = cma_saved["scaling_factors_FF"]
        best_cma_loss = float(cma_saved["best_loss"])
        print(f"  Scaling factors: {best_scaling_factors_FF}")
        print(f"  Best CMA-ES loss: {best_cma_loss:.6f}")

    elif resume_from is None and cma_es_config:
        header = "PHASE 1: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # ScalingFactorProjections for CMA-ES: connectome is the (already
        # noise- and perturbation-baked) per-pair block; init SF = teacher's
        # configured value. The CMA update_fn writes new SFs into each
        # projection's ``log_sf`` buffer.
        cma_projections = make_chunked_ff_projections(
            rec_weights=perturbed_rec_block,
            ff_weights=perturbed_ff_block,
            cell_type_indices=cell_type_indices,
            ff_cell_type_indices=ff_cell_type_indices_only,
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=feedforward.cell_types.names,
            init_sf=1.0,
        )
        # Bake teacher's configured SFs into each pair's initial log_sf so
        # the CMA initial point matches ``concatenated_scaling_factors``.
        with torch.no_grad():
            for (src_name, tgt_name), proj in cma_projections.items():
                src_id = combined_input_cell_type_names.index(src_name)
                tgt_id = rec_output_cell_type_names.index(tgt_name)
                proj.log_sf.copy_(
                    torch.tensor(
                        float(np.log(concatenated_scaling_factors[src_id, tgt_id])),
                        dtype=proj.log_sf.dtype,
                    )
                )

        cma_model = FeedforwardConductanceLIFNetwork(
            **model_kwargs,
            projections=cma_projections,
        )
        cma_model.to(device)

        sf_shape = concatenated_scaling_factors.shape

        def cma_update_fn(flat_log_params: np.ndarray) -> None:
            sf_matrix = np.exp(flat_log_params).reshape(sf_shape).astype(np.float32)
            with torch.no_grad():
                for (src_name, tgt_name), proj in cma_projections.items():
                    src_id = combined_input_cell_type_names.index(src_name)
                    tgt_id = rec_output_cell_type_names.index(tgt_name)
                    proj.log_sf.copy_(
                        torch.tensor(
                            float(np.log(sf_matrix[src_id, tgt_id])),
                            dtype=proj.log_sf.dtype,
                            device=proj.log_sf.device,
                        )
                    )

        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=spike_dataset.dt,
            window_size=chunk_size,
            device=device,
        )
        cma_loss_functions = {
            "van_rossum": cma_van_rossum_fn,
            "firing_rate": firing_rate_loss_fn,
            "silence_penalty": silence_penalty_fn,
        }

        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            **DATALOADER_KWARGS,
            collate_fn=feedforward_collate_fn,
        )

        dt = spike_dataset.dt

        def cma_es_callback(metrics):
            """Log CMA-ES generation metrics to disk and wandb."""
            log_dict = {
                "cma_es/best_loss": metrics["best_loss"],
                "cma_es/sigma": metrics["sigma"],
                "cma_es/n_evals": metrics["n_evals"],
            }
            for loss_name, loss_val in metrics["mean_losses"].items():
                log_dict[f"cma_es_loss/{loss_name}"] = loss_val

            # Scaling factors vs targets
            current_sf = cma_model.scaling_factors_FF.detach().cpu().numpy()
            for source_idx in range(current_sf.shape[0]):
                source_name = input_cell_type_names[source_idx]
                for target_idx in range(current_sf.shape[1]):
                    target_name = output_cell_type_names[target_idx]
                    synapse_name = f"{source_name}_to_{target_name}"
                    target_val = target_scaling_factors_FF[source_idx, target_idx]
                    if target_val != 0:
                        normalized = current_sf[source_idx, target_idx] / target_val
                    else:
                        normalized = current_sf[source_idx, target_idx]
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_value"] = float(
                        normalized
                    )
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_target"] = 1.0

            # Firing rates from output spikes
            output_spikes = metrics["output_spikes"].numpy()
            student_spikes = output_spikes[0]  # (time, n_neurons)
            duration_s = student_spikes.shape[0] * dt / 1000.0
            student_rates = student_spikes.sum(axis=0) / duration_s
            for type_idx, type_name in enumerate(output_cell_type_names):
                mask = cell_type_indices == type_idx
                if mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_{type_name}_mean"] = float(
                        student_rates[mask].mean()
                    )

            # Log to disk (negative epoch to distinguish CMA-ES from gradient training)
            metrics_logger.log(epoch=-metrics["generation"], **log_dict)

            # Log to wandb
            if wandb_logger:
                wandb.log(log_dict)

        searcher = EvolutionarySearch(
            model=cma_model,
            initial_params=np.log(concatenated_scaling_factors).ravel(),
            update_fn=cma_update_fn,
            dataloader=cma_dataloader,
            loss_functions=cma_loss_functions,
            loss_weights=cma_loss_weights,
            device=device,
            n_chunks=cma_es_config.get("n_chunks", 10),
            initial_sigma=cma_es_config.get("initial_sigma", 0.5),
            max_evaluations=cma_es_config.get("max_evaluations", 200),
            popsize=cma_es_config.get("popsize", None),
            batch_size=cma_es_config.get("batch_size", None),
            seed=cma_es_config.get("seed", None),
            callback=cma_es_callback,
            burn_in_chunks=burn_in_chunks,
        )

        best_cma_loss = searcher.search()
        best_scaling_factors_FF = cma_model.scaling_factors_FF.detach().cpu().numpy()

        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")

        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "scaling_factors.npz",
            scaling_factors_FF=best_scaling_factors_FF,
            best_loss=best_cma_loss,
        )

        # Free all CMA-ES GPU memory before gradient phase
        searcher.update_fn = None
        searcher.callback = None
        searcher._chunks = []
        searcher.model = None
        searcher.loss_functions = {}
        del cma_model, searcher
        del cma_van_rossum_fn, cma_loss_functions
        del cma_dataloader, cma_update_fn, cma_es_callback
        gc.collect()
        torch.cuda.empty_cache()

    # ==============================================
    # Initialize Gradient Training Model (Phase 2)
    # ==============================================

    # Gradient phase: ScalingFactorProjection per (input_ct, output_ct).
    # ``best_scaling_factors_FF`` (shape (n_ff_ct + n_rec_ct, n_rec_ct))
    # provides the per-pair initial SF. ``optimisable`` from the original
    # API ("scaling_factors") is implicit here — the only trainable params
    # are the per-pair ``log_sf`` scalars.
    if optimisable != "scaling_factors":
        raise NotImplementedError(
            f"convergence-check supports optimisable='scaling_factors' only; "
            f"got {optimisable!r}"
        )

    gradient_projections = make_chunked_ff_projections(
        rec_weights=perturbed_rec_block,
        ff_weights=perturbed_ff_block,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=ff_cell_type_indices_only,
        cell_type_names=rec_output_cell_type_names,
        ff_cell_type_names=feedforward.cell_types.names,
        init_sf=1.0,
    )
    with torch.no_grad():
        for (src_name, tgt_name), proj in gradient_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_output_cell_type_names.index(tgt_name)
            proj.log_sf.copy_(
                torch.tensor(
                    float(np.log(best_scaling_factors_FF[src_id, tgt_id])),
                    dtype=proj.log_sf.dtype,
                )
            )

    model = FeedforwardConductanceLIFNetwork(
        **model_kwargs,
        projections=gradient_projections,
    )
    model.to(device)

    print("\n✓ Model initialized:")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - Scaling factors shape: {model.scaling_factors_FF.shape}")

    # ==============================
    # Setup Optimizer
    # ==============================

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

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
        # spikes: (batch, time, n_neurons) - trained network
        # target_spikes: (batch, time, n_neurons) - target spikes

        # Take first batch, plot first few neurons
        n_plot = min(10, spikes.shape[2])

        # Interleave target and trained for comparison
        # Shape: (1, time, 2*n_plot) alternating [target0, trained0, target1, trained1, ...]
        interleaved = np.zeros((1, spikes.shape[1], 2 * n_plot))
        for i in range(n_plot):
            interleaved[0, :, 2 * i] = target_spikes[0, :, i]
            interleaved[0, :, 2 * i + 1] = spikes[0, :, i]

        # Create cell type indices for coloring (0=target, 1=trained)
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
            title=f"Feedforward Network (noisy): Target vs Trained (first {n_plot} neurons)",
            ylabel="Neuron",
            figsize=(14, 8),
        )

        return {"spike_comparison": fig}

    # ================================================
    # Create Stats Computer Function
    # ================================================

    dt = spike_dataset.dt

    def stats_computer(snapshot):
        """Compute summary statistics for all neurons (student and teacher)."""
        spikes = snapshot["spikes"]
        # spikes shape: (batch, time, n_neurons)
        student_spikes = spikes[0, :, :]  # (time, n_neurons)
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        # Teacher spikes come from the trainer's accumulated batch data.
        teacher_spikes = snapshot["target_spikes"][0, :n_timesteps, :]

        # Compute student firing rates
        student_spike_counts = student_spikes.sum(axis=0)
        student_firing_rates = student_spike_counts / duration_s

        # Compute teacher firing rates
        teacher_spike_counts = teacher_spikes.sum(axis=0)
        teacher_firing_rates = teacher_spike_counts / duration_s

        # Use flattened metric names so wandb glob pattern firing_rate/* matches all
        stats = {
            # Student aggregate stats
            "firing_rate/student_mean": float(student_firing_rates.mean()),
            "firing_rate/student_std": float(student_firing_rates.std()),
            "firing_rate/student_min": float(student_firing_rates.min()),
            "firing_rate/student_max": float(student_firing_rates.max()),
            # Teacher aggregate stats
            "firing_rate/teacher_mean": float(teacher_firing_rates.mean()),
            "firing_rate/teacher_std": float(teacher_firing_rates.std()),
            "firing_rate/teacher_min": float(teacher_firing_rates.min()),
            "firing_rate/teacher_max": float(teacher_firing_rates.max()),
        }

        # Add cell-type-specific firing rates (handles arbitrary cell types from teacher)
        output_cell_type_names = recurrent.cell_types.names
        for type_idx, type_name in enumerate(output_cell_type_names):
            type_mask = cell_type_indices == type_idx
            if type_mask.sum() > 0:
                # Student by cell type
                student_type_rates = student_firing_rates[type_mask]
                stats[f"firing_rate/student_{type_name}_mean"] = float(
                    student_type_rates.mean()
                )
                stats[f"firing_rate/student_{type_name}_std"] = float(
                    student_type_rates.std()
                )
                # Teacher by cell type
                teacher_type_rates = teacher_firing_rates[type_mask]
                stats[f"firing_rate/teacher_{type_name}_mean"] = float(
                    teacher_type_rates.mean()
                )
                stats[f"firing_rate/teacher_{type_name}_std"] = float(
                    teacher_type_rates.std()
                )

        # Add scaling factor tracking with proper cell type names
        current_sf = snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_FF

        # Get cell type names for proper labeling
        # Combined input types: feedforward names + recurrent names
        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

        # Log all scaling factor elements normalized so target=1
        # This makes it easy to see convergence (value should approach 1)
        # Use flattened names so wandb glob pattern scaling_factors/* matches all
        for source_idx in range(current_sf.shape[0]):
            source_type_name = input_cell_type_names[source_idx]
            for target_idx in range(current_sf.shape[1]):
                target_type_name = output_cell_type_names[target_idx]
                synapse_name = f"{source_type_name}_to_{target_type_name}"
                target_val = target_sf[source_idx, target_idx]
                if target_val != 0:
                    normalized_value = current_sf[source_idx, target_idx] / target_val
                else:
                    normalized_value = current_sf[source_idx, target_idx]
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized_value)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        return stats

    # ===================
    # Setup Training Loop
    # ===================

    # Update params to reflect concatenated cell types for trainer
    # The trainer reads from feedforward.cell_types.names for labeling
    # Since we concatenated recurrent types as additional inputs, update the config
    feedforward.cell_types.names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )

    # Calculate total number of training iterations
    num_epochs = epochs * spike_dataset.num_chunks

    lr_min = getattr(hyperparameters, "lr_min", None)
    if lr_min is not None:
        t_max = epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=t_max, eta_min=lr_min
        )
        print(f"  LR schedule: cosine {learning_rate} → {lr_min} over {epochs} epochs")
    else:
        scheduler = None

    pbar = tqdm(
        range(num_epochs),
        desc="Training",
        unit="chunk",
        total=num_epochs,
    )

    # Create trainer (reuse pre-initialized wandb and metrics logger)
    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions=gradient_loss_functions,
        loss_weights=gradient_loss_weights,
        device=device,
        num_epochs=num_epochs,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_norm_clip=grad_norm_clip,
        wandb_config=None,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=spike_dataset.num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
    )

    # Share the pre-initialized loggers with the trainer
    trainer.metrics_logger = metrics_logger
    trainer.wandb_logger = wandb_logger

    # Handle checkpoint resuming
    if resume_from is not None:
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )
        trainer.set_checkpoint_state(start_epoch, best_loss)
        # Update progress bar to reflect resume point
        pbar.initial = start_epoch
        pbar.refresh()

    # =================
    # Run Training Loop
    # =================

    print(f"\nStarting training from chunk {trainer.current_epoch}...")
    chunk_duration_s = chunk_size * spike_dataset.dt / 1000.0
    total_duration_s = num_epochs * chunk_duration_s
    print(f"Chunk size: {chunk_size} timesteps ({chunk_duration_s:.1f}s)")
    print(f"Total chunks: {num_epochs} ({total_duration_s:.1f}s total)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training all {n_neurons} neurons with Van Rossum loss")
    if noise_frac > 0:
        print(f"Weight noise: {noise_frac * 100:.1f}%")

    model.reset_state(batch_size=batch_size)
    model.track_variables = False

    best_loss = trainer.train(output_dir=output_dir)

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss achieved: {best_loss:.6f}")
    print("=" * 60)

    # Save final state
    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving final network structure...")
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=model.weights_FF.detach().cpu().numpy(),
        feedforward_connectivity=concatenated_mask,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=model.scaling_factors_FF.detach().cpu().numpy(),
    )

    # ==============================================
    # Generate and Save Final Spike Train Plot Data
    # ==============================================

    print(f"\nGenerating final spike train data ({plot_size} chunks)...")

    model.eval()
    model.reset_state(batch_size=batch_size)

    student_chunks = []
    target_chunks = []
    spike_dataloader_iter = iter(spike_dataloader)

    with torch.no_grad():
        for _ in range(plot_size):
            batch = next(spike_dataloader_iter)
            input_spikes = batch["input_spikes"].to(device)
            target_spikes = batch["output_spikes"].to(device)
            student_spikes = model.forward(input_spikes)
            student_chunks.append(student_spikes[0].detach().cpu().numpy())
            target_chunks.append(target_spikes[0].detach().cpu().numpy())

    # Concatenate along time axis: (total_time, n_neurons)
    all_student = np.concatenate(student_chunks, axis=0)
    all_target = np.concatenate(target_chunks, axis=0)

    n_neurons_save = all_student.shape[1]

    print(f"Saving spike data for {n_neurons_save} neurons...")
    np.savez(
        final_state_dir / "plot_data.npz",
        student_spikes=all_student,
        teacher_spikes=all_target,
        dt=spike_dataset.dt,
        cell_type_indices=cell_type_indices,
        n_neurons=n_neurons_save,
    )

    # Close shared loggers
    metrics_logger.close()

    if wandb_logger is not None:
        wandb.finish()

    print(f"\n✓ Checkpoints: {output_dir / 'checkpoints'}")
    print(f"✓ Figures: {output_dir / 'figures'}")
    print(f"✓ Metrics: {output_dir / 'training_metrics.csv'}")
    print(f"✓ Final state: {final_state_dir / 'network_structure.npz'}")
    print(f"✓ Plot data: {final_state_dir / 'plot_data.npz'}")
