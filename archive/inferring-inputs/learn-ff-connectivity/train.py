"""
Inferring-inputs: learn-ff-connectivity training.

Feedforward inputs are actual mitral cell spike trains (not latents).
The network learns a low-rank FF weight matrix (U @ V decomposition)
from scratch, while simultaneously recovering perturbed recurrent scaling
factors via gradient-based optimisation (Adam with separate learning rates
per parameter group).
"""

import gc
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from connectome_snns.dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    FeedforwardCollate,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    LowRankProjection,
    ScalingFactorProjection,
)
from connectome_snns.training_utils.losses import VanRossumLoss
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


def _detect_resume_phase(output_dir):
    """Detect which training phase to resume from based on saved artifacts.

    Returns one of:
        "phase3"             — phase 3 checkpoint exists, resume within phase 3
        "phase2_done"        — phase 2 completed (result saved), start phase 3 fresh
        "phase2_in_progress" — phase 2 checkpoint exists but not completed
        "cma_done"           — CMA-ES completed, no phase 2 progress yet
    """
    phase2_result = output_dir / "phase2_state" / "phase2_result.npz"
    cma_result = output_dir / "cma_es_state" / "cma_es_result.npz"
    checkpoint_dir = output_dir / "checkpoints"
    has_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))

    if phase2_result.exists() and has_checkpoints:
        # Peek at the latest checkpoint to check if it's from phase 3
        latest = sorted(checkpoint_dir.glob("*.pt"))[-1]
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        keys = ckpt["model_state_dict"].keys()
        # Phase 3 introduces LowRankProjection (U, V parameters) on FF pairs.
        if any(k.endswith(".U") or k.endswith(".V") for k in keys):
            return "phase3"
        # Checkpoints are stale phase 2 leftovers; start phase 3 fresh
        return "phase2_done"
    elif phase2_result.exists():
        return "phase2_done"
    elif has_checkpoints:
        return "phase2_in_progress"
    elif cma_result.exists():
        return "cma_done"
    else:
        raise RuntimeError(
            f"Cannot determine resume phase from {output_dir}. "
            "No CMA-ES results, checkpoints, or phase 2 state found."
        )


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train network with learn-ff-connectivity: low-rank FF weights + scaling factors.

    Feedforward inputs are mitral cell spike trains from spike_data.zarr.
    Phase 3 learns a low-rank FF weight matrix (U @ V) while recovering
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
    cma_es_config = data.get("cma_es", {})
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
    phase2_config = data.get("phase2", {})
    phase3_config = data.get("phase3", {})

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

    # Get base scaling factors from config (recurrent only — FF scaling is
    # absorbed into the learned FF weight constants).
    sf_recurrent = np.array(scaling_factors["recurrent"], dtype=np.float32)

    # ================================================================
    # Load teacher network structure
    # ================================================================

    network_structure = np.load(input_dir / "network_structure.npz")
    teacher_weights = network_structure["recurrent_weights"]
    teacher_ff_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]

    n_neurons = teacher_weights.shape[0]
    n_feedforward_teacher = teacher_ff_weights.shape[0]
    exc_mask = cell_type_indices == 0  # excitatory neurons

    # ================================================================
    # Load exact FF spike dataset
    # ================================================================

    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
        recurrent_smoothing_tau=training.recurrent_smoothing_tau,
    )
    n_feedforward = spike_dataset.n_input_neurons
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
    ff_weight_constants = []  # one constant per output cell type
    for type_idx, type_name in enumerate(output_cell_type_names):
        type_mask = cell_type_indices == type_idx
        mean_ff_type = teacher_ff_weights[:, type_mask].sum(axis=0).mean()
        constant_val = mean_ff_type / n_feedforward
        ff_weights[:, type_mask] = constant_val
        ff_weight_constants.append(constant_val)
        print(f"  FF weight init -> {type_name}: {constant_val:.6f}")

    ff_weight_constants = np.array(ff_weight_constants, dtype=np.float32)
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
    # Detect resume phase
    resume_phase = None
    if resume_from is not None:
        resume_phase = _detect_resume_phase(output_dir)

        header = f"RESUMING — detected phase: {resume_phase}"
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
        # Apply Scaling Factor Perturbation (recurrent only)
        # ============================================
        # Feedforward scaling factors are meaningless here — any scaling is
        # absorbed into the learned FF weights. Only perturb recurrent.

        sigma = np.sqrt(weight_perturbation_variance)
        mu = -(sigma**2) / 2.0  # Ensures E[target] = 1

        target_scaling_factors_rec = np.random.lognormal(
            mean=mu, sigma=sigma, size=sf_recurrent.shape
        )

        # Perturbation is reciprocal of target (so target * perturbation = 1)
        perturbation_factors_rec = 1.0 / target_scaling_factors_rec

        # Apply perturbation to recurrent weights only (not FF weights)
        perturbed_weights = concatenated_weights.copy()
        for input_idx in range(n_feedforward, n_total_inputs):
            input_type = concatenated_cell_type_indices[input_idx]
            rec_type = input_type - n_ff_cell_types  # index into recurrent SF matrix
            for output_idx in range(n_neurons):
                output_type = cell_type_indices[output_idx]
                perturbed_weights[input_idx, output_idx] *= perturbation_factors_rec[
                    rec_type, output_type
                ]

        normalized_initial = sf_recurrent / target_scaling_factors_rec
        print(
            f"\nRecurrent scaling factors / target (should recover to 1.0):\n{normalized_initial}"
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

    model_kwargs = dict(
        dt=spike_dataset.dt,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=phase2_config.get("surrgrad_scale", 1.0),
        batch_size=batch_size,
        track_variables=False,
    )

    # Cell-type-name spaces for projection keys
    rec_output_cell_type_names = recurrent.cell_types.names
    ff_input_cell_type_names = list(feedforward.cell_types.names)

    # Indices for slicing concatenated weights into FF and recurrent blocks
    rec_idx_per_ct = {
        name: np.flatnonzero(cell_type_indices == ct_id)
        for ct_id, name in enumerate(rec_output_cell_type_names)
    }
    ff_idx_per_ct = {
        name: np.flatnonzero(ff_cell_type_indices == ct_id)
        for ct_id, name in enumerate(ff_input_cell_type_names)
    }

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
            "cma_es": cma_es_config,
            "output_dir": str(output_dir),
            "device": device,
            "n_feedforward": n_feedforward,
            "ff_rank": ff_rank,
            "phase2": phase2_config,
            "phase3": phase3_config,
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

    # ================================================
    # Phase 1: CMA-ES Evolutionary Search
    # ================================================

    # Helper: build the dict of ScalingFactorProjections covering all 6 pairs
    # (4 recurrent + 2 mitral->rec). FF connectome = uniform 1.0 so that the
    # mitral log_sf carries the FF weight constant directly; recurrent
    # connectome = perturbed rec block, with log_sf carrying sf_recurrent[s,t].
    perturbed_rec_block = perturbed_weights[n_feedforward:, :]

    def _build_sf_projections(rec_log_sf_2x2, ff_log_constants):
        """rec_log_sf_2x2: (n_rec_ct, n_rec_ct) log-SFs.
        ff_log_constants: (n_ff_ct, n_rec_ct) log-FF-constants."""
        projs: dict = {}
        for src_id, src_name in enumerate(ff_input_cell_type_names):
            src_idx = ff_idx_per_ct[src_name]
            for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
                tgt_idx = rec_idx_per_ct[tgt_name]
                # Uniform 1.0 connectome over (mitral_src, rec_tgt) — full FF.
                block = np.ones((len(src_idx), len(tgt_idx)), dtype=np.float32)
                proj = ScalingFactorProjection(
                    block, init_sf=float(np.exp(ff_log_constants[src_id, tgt_id]))
                )
                projs[(src_name, tgt_name)] = proj
        for src_id, src_name in enumerate(rec_output_cell_type_names):
            src_idx = rec_idx_per_ct[src_name]
            for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
                tgt_idx = rec_idx_per_ct[tgt_name]
                block = perturbed_rec_block[np.ix_(src_idx, tgt_idx)].astype(np.float32)
                proj = ScalingFactorProjection(
                    block, init_sf=float(np.exp(rec_log_sf_2x2[src_id, tgt_id]))
                )
                projs[(src_name, tgt_name)] = proj
        return projs

    # Track best parameters from CMA-ES (or use initial values)
    best_rec_log_sf = np.log(sf_recurrent).astype(np.float32)  # (2, 2)
    best_ff_log_constants = (
        np.log(ff_weight_constants).astype(np.float32).reshape(1, -1)
    )  # (1, 2) — one mitral source, 2 targets
    best_ff_weight_constants = ff_weight_constants

    # Check if CMA-ES already completed (resume skips directly to gradient phase)
    cma_state_path = output_dir / "cma_es_state" / "cma_es_result.npz"
    if resume_from is not None and cma_state_path.exists():
        print("\nLoading CMA-ES results from previous run...")
        cma_saved = np.load(cma_state_path)
        best_scaling_factors_FF = cma_saved["scaling_factors_FF"]
        best_ff_weight_constants = cma_saved["ff_weight_constants"]
        # Reconstruct per-pair log-SF/log-FF-constants from saved state.
        best_rec_log_sf = np.log(
            best_scaling_factors_FF[n_ff_cell_types:].astype(np.float32)
        )
        best_ff_log_constants = np.log(
            best_ff_weight_constants.astype(np.float32)
        ).reshape(1, -1)
        print(f"  Scaling factors: {best_scaling_factors_FF}")
        print(f"  FF weight constants: {best_ff_weight_constants}")
        print(f"  Best CMA-ES loss: {cma_saved['best_loss']:.6f}")

    elif resume_from is None and cma_es_config:
        header = "PHASE 1: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # Build ScalingFactorProjections for all 6 pairs.
        cma_projections = _build_sf_projections(best_rec_log_sf, best_ff_log_constants)

        cma_model = FeedforwardConductanceLIFNetwork(
            **model_kwargs,
            projections=cma_projections,
        )
        cma_model.to(device)

        print("\nCMA-ES model initialized:")
        print("  - Searching over 6 parameters:")
        print("    - 4 recurrent scaling factors (2x2)")
        print("    - 2 FF weight constants (mitral->exc, mitral->inh)")

        # Build initial parameter vector in log space:
        # [4 recurrent scaling factors, 2 FF weight constants]
        initial_log_sf = best_rec_log_sf.ravel()  # (4,)
        initial_log_ff = best_ff_log_constants.ravel()  # (2,)
        initial_params = np.concatenate([initial_log_sf, initial_log_ff])
        print(f"  - Initial params (log): {initial_params}")

        # Ordered list of (pair_key, role) corresponding to flat_log_params:
        # first 4 entries -> recurrent pairs in row-major order,
        # last 2 entries -> mitral -> (exc, inh).
        cma_pair_order: list = []
        for src_name in rec_output_cell_type_names:
            for tgt_name in rec_output_cell_type_names:
                cma_pair_order.append((src_name, tgt_name))
        for tgt_name in rec_output_cell_type_names:
            cma_pair_order.append(("mitral", tgt_name))

        def cma_update_fn(flat_log_params: np.ndarray) -> None:
            """Write flat log-params into each projection's log_sf in place."""
            with torch.no_grad():
                for i, key in enumerate(cma_pair_order):
                    proj = cma_projections[key]
                    proj.log_sf.copy_(
                        torch.tensor(
                            float(flat_log_params[i]),
                            dtype=proj.log_sf.dtype,
                            device=proj.log_sf.device,
                        )
                    )

        # CMA-ES needs its own loss instances (they have internal state)
        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=spike_dataset.dt,
            window_size=chunk_size,
            device=device,
        )
        cma_loss_functions = {"van_rossum": cma_van_rossum_fn}
        cma_loss_weights = {"van_rossum": loss_weight_van_rossum}

        if van_rossum_rate_loss_fn is not None:
            cma_van_rossum_rate_fn = VanRossumLoss(
                tau_rise=van_rossum_rate_tau_rise,
                tau_decay=van_rossum_rate_tau_decay,
                dt=spike_dataset.dt,
                window_size=chunk_size,
                device=device,
            )
            cma_loss_functions["van_rossum_rate"] = cma_van_rossum_rate_fn
            cma_loss_weights["van_rossum_rate"] = loss_weight_van_rossum_rate

        # Separate dataloader for CMA-ES
        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            **DATALOADER_KWARGS,
            collate_fn=collate_fn,
        )

        # Preload one chunk for per-generation evaluation
        cma_eval_iter = iter(
            DataLoader(
                spike_dataset,
                batch_size=None,
                sampler=CyclicSampler(spike_dataset),
                **DATALOADER_KWARGS,
                collate_fn=collate_fn,
            )
        )
        cma_eval_batch = next(cma_eval_iter)
        cma_eval_input = cma_eval_batch.input_spikes[:1].to(device)
        cma_eval_target = getattr(cma_eval_batch, "target_spikes", None)
        if cma_eval_target is not None:
            cma_eval_target = cma_eval_target[:1].to(device)
        cma_eval_dt = spike_dataset.dt

        def cma_es_callback(metrics):
            log_dict = {
                "cma_es/best_loss": metrics["best_loss"],
                "cma_es/sigma": metrics["sigma"],
                "cma_es/n_evals": metrics["n_evals"],
            }

            for loss_name, loss_val in metrics["mean_losses"].items():
                log_dict[f"cma_es_loss/{loss_name}"] = loss_val

            # Log recurrent scaling factors normalized by target
            current_sf = cma_model.scaling_factors_FF.detach().cpu().numpy()
            rec_sf = current_sf[n_ff_cell_types:, :]
            rec_cell_type_names = recurrent.cell_types.names
            for source_idx in range(target_scaling_factors_rec.shape[0]):
                source_name = rec_cell_type_names[source_idx]
                for target_idx in range(target_scaling_factors_rec.shape[1]):
                    target_name = rec_cell_type_names[target_idx]
                    synapse_name = f"{source_name}_to_{target_name}"
                    target_val = target_scaling_factors_rec[source_idx, target_idx]
                    if target_val != 0:
                        normalized = rec_sf[source_idx, target_idx] / target_val
                    else:
                        normalized = rec_sf[source_idx, target_idx]
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_value"] = float(
                        normalized
                    )
                    log_dict[f"cma_es_scaling_factors/{synapse_name}_target"] = 1.0

            # Log FF weight constants
            current_weights = cma_model.weights_FF.detach().cpu().numpy()
            current_ff_w = current_weights[:n_feedforward, :]
            for type_idx, type_name in enumerate(output_cell_type_names):
                type_mask = cell_type_indices == type_idx
                log_dict[f"cma_es_ff_weights/{type_name}_constant"] = float(
                    current_ff_w[:, type_mask].mean()
                )

            # Forward pass on best candidate for firing rate stats
            cma_model.reset_state(batch_size=1)
            cma_model.track_variables = False
            cma_model.track_batch_idx = None
            with torch.inference_mode():
                spikes = cma_model.forward(input_spikes=cma_eval_input)

            spikes_np = spikes[0].detach().cpu().numpy()
            n_timesteps = spikes_np.shape[0]
            duration_s = n_timesteps * cma_eval_dt / 1000.0
            student_rates = spikes_np.sum(axis=0) / duration_s
            log_dict["cma_es_firing_rate/student_mean"] = float(student_rates.mean())
            log_dict["cma_es_firing_rate/student_std"] = float(student_rates.std())

            if cma_eval_target is not None:
                teacher_np = cma_eval_target[0].detach().cpu().numpy()
                teacher_rates = teacher_np.sum(axis=0) / duration_s
                log_dict["cma_es_firing_rate/teacher_mean"] = float(
                    teacher_rates.mean()
                )

            for type_idx, type_name in enumerate(output_cell_type_names):
                type_mask = cell_type_indices == type_idx
                if type_mask.sum() > 0:
                    log_dict[f"cma_es_firing_rate/student_{type_name}_mean"] = float(
                        student_rates[type_mask].mean()
                    )
                    if cma_eval_target is not None:
                        log_dict[f"cma_es_firing_rate/teacher_{type_name}_mean"] = (
                            float(teacher_rates[type_mask].mean())
                        )

            # Synaptic drive comparison (FF vs recurrent)
            ff_chunk_eval = (
                cma_eval_input[0, :, :n_feedforward].detach().cpu().numpy()
            )  # (time, n_feedforward)
            rec_spikes_eval = spikes_np  # (time, n_neurons)
            rec_weights_eval = current_weights[n_feedforward:, :]

            # FF: total spike count * outgoing weight per FF neuron
            ff_spike_counts = ff_chunk_eval.sum(axis=0)  # (n_feedforward,)
            ff_outgoing_w = current_ff_w.sum(axis=1)  # (n_feedforward,)
            ff_drive = (ff_spike_counts * ff_outgoing_w).sum()

            # Rec exc: spike count * outgoing weight per exc neuron
            rec_spike_counts = rec_spikes_eval.sum(axis=0)
            exc_outgoing_w = rec_weights_eval[exc_mask, :].sum(axis=1)
            rec_exc_drive = (rec_spike_counts[exc_mask] * exc_outgoing_w).sum()

            total_drive = ff_drive + rec_exc_drive
            log_dict["cma_es_drive/ff_fraction"] = float(
                ff_drive / max(total_drive, 1e-10)
            )
            log_dict["cma_es_drive/rec_exc_fraction"] = float(
                rec_exc_drive / max(total_drive, 1e-10)
            )
            log_dict["cma_es_drive/ratio_rec_exc_over_ff"] = float(
                rec_exc_drive / max(ff_drive, 1e-10)
            )

            metrics_logger.log(epoch=-metrics["generation"], **log_dict)
            if wandb_run is not None:
                wandb.log(log_dict)

        searcher = EvolutionarySearch(
            model=cma_model,
            initial_params=initial_params,
            update_fn=cma_update_fn,
            dataloader=cma_dataloader,
            loss_functions=cma_loss_functions,
            loss_weights=cma_loss_weights,
            device=device,
            n_chunks=cma_es_config.get("n_chunks", 10),
            initial_sigma=cma_es_config.get("initial_sigma", 0.1),
            max_evaluations=cma_es_config.get("max_evaluations", 200),
            popsize=cma_es_config.get("popsize", None),
            batch_size=cma_es_config.get("batch_size", None),
            seed=cma_es_config.get("seed", None),
            callback=cma_es_callback,
            burn_in_chunks=burn_in_chunks,
        )

        best_cma_loss = searcher.search()

        # Extract best parameters from CMA-ES model.
        # scaling_factors_FF property assembles a (n_ff_ct + n_rec_ct, n_rec_ct)
        # matrix from each ScalingFactorProjection's exp(log_sf).
        best_scaling_factors_FF = cma_model.scaling_factors_FF.detach().cpu().numpy()
        # Per-pair log_sf values, stored back into our running "best" arrays.
        for src_id, src_name in enumerate(rec_output_cell_type_names):
            for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
                best_rec_log_sf[src_id, tgt_id] = float(
                    cma_projections[(src_name, tgt_name)].log_sf.detach().cpu()
                )
        for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
            best_ff_log_constants[0, tgt_id] = float(
                cma_projections[("mitral", tgt_name)].log_sf.detach().cpu()
            )
        best_ff_weight_constants = np.exp(best_ff_log_constants.ravel()).astype(
            np.float32
        )

        normalized_sf = np.exp(best_rec_log_sf) / target_scaling_factors_rec
        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")
        print(f"Recurrent SF / target (should be 1): {normalized_sf}")
        print(f"FF weight constants: {best_ff_weight_constants}")

        # Save CMA-ES state
        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "cma_es_result.npz",
            scaling_factors_FF=best_scaling_factors_FF,
            ff_weight_constants=best_ff_weight_constants,
            best_loss=best_cma_loss,
        )
        print(f"Saved CMA-ES state to {cma_state_dir}")

        # Free all CMA-ES GPU memory before gradient phase
        # Clear searcher internals first — closures and preloaded chunks hold
        # GPU tensors alive even after `del searcher` until GC runs.
        searcher.update_fn = None
        searcher.callback = None
        searcher._chunks = []
        searcher.model = None
        searcher.loss_functions = {}
        del cma_model, cma_projections, searcher
        del cma_eval_input, cma_eval_target, cma_eval_batch, cma_eval_iter
        del cma_dataloader, cma_update_fn, cma_es_callback
        del cma_loss_functions, cma_loss_weights, cma_van_rossum_fn
        try:
            del cma_van_rossum_rate_fn
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

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

        # Scaling factor tracking (recurrent only)
        current_sf = snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_rec
        rec_cell_type_names = recurrent.cell_types.names

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
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized_value)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        # ======================
        # Learned FF weight matrix stats
        # ======================
        weights_ff = snapshot["weights_FF"]
        # ``weights_FF`` already includes the per-pair scaling factor
        # (ScalingFactorProjection.forward() returns ``exp(log_sf) * connectome``),
        # and LowRankProjection has no per-pair SF — so the FF block is
        # already the effective weight matrix.
        ff_w_effective = weights_ff[:n_feedforward, :]

        stats["ff_weights/mean"] = float(ff_w_effective.mean())
        stats["ff_weights/std"] = float(ff_w_effective.std())
        stats["ff_weights/min"] = float(ff_w_effective.min())
        stats["ff_weights/max"] = float(ff_w_effective.max())

        # Per output cell type
        for type_idx, type_name in enumerate(recurrent.cell_types.names):
            type_mask = cell_type_indices == type_idx
            if type_mask.sum() > 0:
                w_type = ff_w_effective[:, type_mask]
                stats[f"ff_weights/{type_name}_mean"] = float(w_type.mean())
                stats[f"ff_weights/{type_name}_std"] = float(w_type.std())

        # Correlation with teacher FF weights
        # Reshape learned weights to match teacher shape if needed
        if n_feedforward == n_feedforward_teacher:
            teacher_ff_flat = teacher_ff_weights.ravel()
            learned_ff_flat = ff_w_effective.ravel()
            if len(teacher_ff_flat) == len(learned_ff_flat):
                corr = np.corrcoef(teacher_ff_flat, learned_ff_flat)[0, 1]
                stats["ff_weights/teacher_correlation"] = float(corr)

        # Effective rank (SVD participation ratio)
        try:
            s = np.linalg.svd(ff_w_effective, compute_uv=False)
            # Participation ratio: (sum s_i)^2 / sum(s_i^2)
            effective_rank = (s.sum() ** 2) / (s**2).sum()
            stats["ff_weights/effective_rank"] = float(effective_rank)
            # Fraction of variance in top ff_rank components
            total_var = (s**2).sum()
            top_var = (s[:ff_rank] ** 2).sum()
            stats["ff_weights/top_rank_variance_fraction"] = float(
                top_var / max(total_var, 1e-10)
            )
        except np.linalg.LinAlgError:
            pass

        # ======================
        # Feedforward vs recurrent drive comparison
        # ======================
        # As with the FF block, the recurrent block of ``weights_FF`` already
        # has each pair's scaling factor baked in via the projection forward.
        rec_weights = weights_ff[n_feedforward:, :]

        exc_outgoing_w = np.zeros(int(exc_mask.sum()), dtype=np.float64)
        for target_type in range(current_sf.shape[1]):
            target_mask = cell_type_indices == target_type
            exc_to_type = rec_weights[exc_mask][:, target_mask].sum(axis=1)
            exc_outgoing_w += exc_to_type

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

    phase2_epochs = phase2_config.get("epochs", 0)
    phase3_epochs = phase3_config.get("epochs", 100)
    phase2_chunks = phase2_epochs * spike_dataset.num_chunks
    phase3_chunks = phase3_epochs * spike_dataset.num_chunks

    # =============================================
    # Phase 2: Gradient on Scaling Factors Only (6 params)
    # =============================================

    # Decide whether to run phase 2
    run_phase2 = False
    resume_phase2_checkpoint = None
    if resume_from is None:
        run_phase2 = phase2_epochs > 0
    elif resume_phase == "phase2_in_progress":
        run_phase2 = True
        resume_phase2_checkpoint = sorted((output_dir / "checkpoints").glob("*.pt"))[-1]
        print(f"\nResuming Phase 2 from checkpoint: {resume_phase2_checkpoint}")
    elif resume_phase == "cma_done":
        run_phase2 = phase2_epochs > 0

    if run_phase2:
        header = "PHASE 2: Gradient Training (Scaling Factors Only)"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        print(
            f"  {phase2_epochs} epochs, 6 params: FF scaling (1x2) + rec scaling (2x2)"
        )

        # Phase 2 uses ScalingFactorProjection on every (input_ct, output_ct)
        # pair (same as CMA). The optimisable params are each projection's
        # scalar log_sf.
        phase2_projections = _build_sf_projections(
            best_rec_log_sf, best_ff_log_constants
        )

        phase2_model = FeedforwardConductanceLIFNetwork(
            **model_kwargs,
            projections=phase2_projections,
        )
        phase2_model.to(device)

        p2_lr_scaling = phase2_config.get("lr_scaling", 5e-2)
        p2_lr_min_scaling = phase2_config.get("lr_min_scaling", None)
        p2_grad_clip_scaling = phase2_config.get("grad_clip_scaling", 0.5)

        phase2_scaling_params = [proj.log_sf for proj in phase2_projections.values()]
        phase2_param_groups = [{"params": phase2_scaling_params, "lr": p2_lr_scaling}]
        phase2_optimiser = torch.optim.Adam(phase2_param_groups, betas=(beta1, beta2))

        if p2_lr_min_scaling is not None:
            p2_frac = p2_lr_min_scaling / p2_lr_scaling

            def p2_cosine_lambda(epoch):
                return (
                    p2_frac
                    + (1 - p2_frac)
                    * (1 + math.cos(math.pi * epoch / phase2_epochs))
                    / 2
                )

            phase2_scheduler = torch.optim.lr_scheduler.LambdaLR(
                phase2_optimiser, [p2_cosine_lambda]
            )
            print(
                f"  LR schedule: cosine {p2_lr_scaling:.1e} -> {p2_lr_min_scaling:.1e}"
            )
        else:
            phase2_scheduler = None

        phase2_scaler = GradScaler(
            "cuda", enabled=training.mixed_precision and device == "cuda"
        )

        def _clip_grads_phase2(m):
            if phase2_scaling_params:
                torch.nn.utils.clip_grad_norm_(
                    phase2_scaling_params, max_norm=p2_grad_clip_scaling
                )

        # Surrogate gradient annealing (Phase 2 only)
        p2_surrgrad_init = phase2_config.get("surrgrad_scale", 1.0)
        p2_surrgrad_final = phase2_config.get("surrgrad_scale_final", None)
        phase2_epoch_callbacks = []
        if p2_surrgrad_final is not None:

            def _anneal_surrgrad_p2(data_epoch):
                frac = min(data_epoch / max(phase2_epochs - 1, 1), 1.0)
                cosine_frac = (1 - math.cos(math.pi * frac)) / 2
                new_scale = p2_surrgrad_init + cosine_frac * (
                    p2_surrgrad_final - p2_surrgrad_init
                )
                phase2_model.surrgrad_scale.fill_(new_scale)

            phase2_epoch_callbacks.append(_anneal_surrgrad_p2)

        phase2_pbar = tqdm(
            range(phase2_chunks),
            desc="Phase 2 (scaling only)",
            unit="chunk",
            total=phase2_chunks,
        )

        phase2_trainer = SNNTrainer(
            model=phase2_model,
            optimizer=phase2_optimiser,
            scaler=phase2_scaler,
            dataloader=spike_dataloader,
            loss_functions=gradient_loss_functions,
            loss_weights=gradient_loss_weights,
            device=device,
            num_epochs=phase2_chunks,
            chunks_per_update=chunks_per_update,
            log_interval=log_interval,
            checkpoint_interval=checkpoint_interval,
            plot_size=plot_size,
            mixed_precision=mixed_precision,
            grad_clip_fn=_clip_grads_phase2,
            wandb_config=None,
            progress_bar=phase2_pbar,
            plot_generator=plot_generator,
            stats_computer=stats_computer,
            chunks_per_data_epoch=spike_dataset.num_chunks,
            burn_in_chunks=burn_in_chunks,
            scheduler=phase2_scheduler,
            epoch_callbacks=phase2_epoch_callbacks,
        )
        phase2_trainer.metrics_logger = metrics_logger
        if wandb_run is not None:
            phase2_trainer.wandb_logger = wandb_run

        # Resume within phase 2 if we have a checkpoint
        if resume_phase2_checkpoint is not None:
            p2_epoch, _, _, _, p2_best = load_checkpoint(
                checkpoint_path=resume_phase2_checkpoint,
                model=phase2_model,
                optimiser=phase2_optimiser,
                scaler=phase2_scaler,
                device=device,
            )
            phase2_trainer.set_checkpoint_state(p2_epoch, p2_best)
            phase2_pbar.n = p2_epoch
            phase2_pbar.refresh()
            print(f"  Resumed Phase 2 at chunk {p2_epoch}, best loss {p2_best:.6f}")

        phase2_model.reset_state(batch_size=batch_size)
        phase2_model.track_variables = False

        phase2_best = phase2_trainer.train(output_dir=output_dir)
        print(f"\nPhase 2 complete. Best loss: {phase2_best:.6f}")

        # Extract learned per-pair log_sf back into our running "best" arrays.
        # Mitral pairs' log_sf carries the FF weight constant directly; the
        # FF connectome is uniform 1.0, so the effective FF block is
        # ``exp(best_ff_log_constants[0, ct])`` per output column.
        for src_id, src_name in enumerate(rec_output_cell_type_names):
            for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
                best_rec_log_sf[src_id, tgt_id] = float(
                    phase2_projections[(src_name, tgt_name)].log_sf.detach().cpu()
                )
        for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
            best_ff_log_constants[0, tgt_id] = float(
                phase2_projections[("mitral", tgt_name)].log_sf.detach().cpu()
            )
        best_ff_weight_constants = np.exp(best_ff_log_constants.ravel()).astype(
            np.float32
        )
        best_scaling_factors_FF = phase2_model.scaling_factors_FF.detach().cpu().numpy()

        print(f"  FF weight constants after Phase 2: {best_ff_weight_constants}")
        print(f"  Recurrent log-SF after Phase 2: {best_rec_log_sf}")

        # Save Phase 2 results for resume
        phase2_state_dir = output_dir / "phase2_state"
        phase2_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            phase2_state_dir / "phase2_result.npz",
            scaling_factors_FF=best_scaling_factors_FF,
            ff_weight_constants=best_ff_weight_constants,
            rec_log_sf=best_rec_log_sf,
            ff_log_constants=best_ff_log_constants,
            best_loss=phase2_best,
        )
        print(f"  Phase 2 results saved to {phase2_state_dir}")

        del phase2_model, phase2_projections, phase2_trainer
        torch.cuda.empty_cache()

    # ==============================================
    # Initialize Phase 3 Model (Low-Rank FF Weights)
    # ==============================================

    # Load post-phase-2 state from disk when resuming after phase 2.
    if resume_phase in ("phase2_done", "phase3"):
        p2_result = np.load(output_dir / "phase2_state" / "phase2_result.npz")
        if "rec_log_sf" in p2_result and "ff_log_constants" in p2_result:
            best_rec_log_sf = p2_result["rec_log_sf"].astype(np.float32)
            best_ff_log_constants = p2_result["ff_log_constants"].astype(np.float32)
        else:
            # Backwards-compat path for old phase2 results.
            saved_sf = p2_result["scaling_factors_FF"]
            best_rec_log_sf = np.log(saved_sf[n_ff_cell_types:].astype(np.float32))
            best_ff_log_constants = np.log(
                p2_result["ff_weight_constants"].astype(np.float32)
            ).reshape(1, -1)
        best_ff_weight_constants = np.exp(best_ff_log_constants.ravel()).astype(
            np.float32
        )
        best_scaling_factors_FF = p2_result["scaling_factors_FF"]
        print("\n  Loaded phase 2 state")
        print(f"  Scaling factors: {best_scaling_factors_FF}")

    # Build Phase 3 projections:
    #   - LowRankProjection for each FF pair (mitral -> rec_ct), initialised
    #     via truncated SVD of the per-pair effective log-FF block.
    #   - ScalingFactorProjection for each recurrent pair (rec_ct -> rec_ct).
    print(f"\nBuilding Phase 3 projections (rank={ff_rank} for FF)...")

    phase3_projections: dict = {}

    # FF pairs: log of the effective FF block per output type is constant
    # (= best_ff_log_constants[0, tgt_id]) since the connectome was uniform.
    for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
        tgt_idx = rec_idx_per_ct[tgt_name]
        n_tgt = len(tgt_idx)
        log_block = np.full(
            (n_feedforward, n_tgt),
            float(best_ff_log_constants[0, tgt_id]),
            dtype=np.float32,
        )
        log_block_t = torch.from_numpy(log_block)
        rank_eff = min(ff_rank, *log_block_t.shape)
        U_full, S_full, Vh_full = torch.linalg.svd(log_block_t, full_matrices=False)
        # Zero out singular values that are float32-noise relative to the
        # leading one. The FF block is rank-1 (uniform per output type),
        # so without this the SVD reconstruction picks up a spurious
        # ±0.5-in-log-space perturbation from tail singular values before
        # any optimiser step has run.
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
        phase3_projections[("mitral", tgt_name)] = LowRankProjection(
            n_source=n_feedforward,
            n_target=n_tgt,
            rank=rank_eff,
            U_init=U_init,
            V_init=V_init,
            mask=None,  # full mitral connectivity
        )

    # Recurrent pairs: per-pair perturbed connectome with trainable log_sf.
    for src_id, src_name in enumerate(rec_output_cell_type_names):
        src_idx = rec_idx_per_ct[src_name]
        for tgt_id, tgt_name in enumerate(rec_output_cell_type_names):
            tgt_idx = rec_idx_per_ct[tgt_name]
            block = perturbed_rec_block[np.ix_(src_idx, tgt_idx)].astype(np.float32)
            phase3_projections[(src_name, tgt_name)] = ScalingFactorProjection(
                block, init_sf=float(np.exp(best_rec_log_sf[src_id, tgt_id]))
            )

    model = FeedforwardConductanceLIFNetwork(
        **model_kwargs,
        projections=phase3_projections,
    )
    model.to(device)

    print("\nPhase 3 model initialized (LowRank FF + ScalingFactor rec):")
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - FF weight rank: {ff_rank}")
    print(f"  - Scaling factors shape: {model.scaling_factors_FF.shape}")

    # ==============================
    # Setup Optimizer for Phase 3
    # ==============================

    p3_lr_weights = phase3_config.get("lr_weights", 1e-3)
    p3_lr_scaling = phase3_config.get("lr_scaling", 5e-2)
    p3_lr_min_weights = phase3_config.get("lr_min_weights", None)
    p3_lr_min_scaling = phase3_config.get("lr_min_scaling", None)
    p3_grad_clip_weights = phase3_config.get("grad_clip_weights", 50.0)
    p3_grad_clip_scaling = phase3_config.get("grad_clip_scaling", 0.5)

    # Group parameters: U/V from LowRank go into the "weights" group;
    # log_sf from each recurrent ScalingFactor goes into "scaling".
    weights_params: list[nn.Parameter] = []
    scaling_params: list[nn.Parameter] = []
    for proj in phase3_projections.values():
        if isinstance(proj, LowRankProjection):
            weights_params.append(proj.U)
            weights_params.append(proj.V)
        elif isinstance(proj, ScalingFactorProjection):
            scaling_params.append(proj.log_sf)

    param_groups = [{"params": weights_params, "lr": p3_lr_weights}]
    print(f"  LR (weights U, V): {p3_lr_weights}, {len(weights_params)} tensors")
    if scaling_params:
        param_groups.append({"params": scaling_params, "lr": p3_lr_scaling})
        print(f"  LR (scaling factors): {p3_lr_scaling}, {len(scaling_params)} tensors")

    optimiser = torch.optim.Adam(param_groups, betas=(beta1, beta2))

    # Per-group cosine annealing for Phase 3
    p3_lr_pairs = []  # (lr_init, lr_min) per param group
    for group in param_groups:
        lr_init = group["lr"]
        if lr_init == p3_lr_weights and p3_lr_min_weights is not None:
            p3_lr_pairs.append((lr_init, p3_lr_min_weights))
        elif lr_init == p3_lr_scaling and p3_lr_min_scaling is not None:
            p3_lr_pairs.append((lr_init, p3_lr_min_scaling))
        else:
            p3_lr_pairs.append((lr_init, lr_init))

    if any(lr_min < lr_init for lr_init, lr_min in p3_lr_pairs):

        def make_cosine_lambda(lr_init, lr_min, n_epochs):
            frac = lr_min / lr_init

            def cosine_lr_lambda(epoch):
                return (
                    frac + (1 - frac) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
                )

            return cosine_lr_lambda

        lambdas = [make_cosine_lambda(li, lm, phase3_epochs) for li, lm in p3_lr_pairs]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambdas)
        print(f"  LR schedule: cosine annealing over {phase3_epochs} epochs")
        for li, lm in p3_lr_pairs:
            print(f"    {li:.1e} -> {lm:.1e}")
    else:
        scheduler = None

    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    def _clip_grads_per_group(m):
        # Clip low-rank factors U, V together
        if weights_params:
            torch.nn.utils.clip_grad_norm_(
                weights_params, max_norm=p3_grad_clip_weights
            )
        if scaling_params:
            torch.nn.utils.clip_grad_norm_(
                scaling_params, max_norm=p3_grad_clip_scaling
            )

    # Set Phase 3 surrogate gradient scale (fixed, no annealing)
    p3_surrgrad = phase3_config.get("surrgrad_scale", 5.0)
    model.surrgrad_scale.fill_(p3_surrgrad)
    print(f"  Surrogate gradient scale: {p3_surrgrad} (fixed)")

    epoch_callbacks = []
    num_phase3_chunks = phase3_chunks

    total_gradient_chunks = phase2_chunks + num_phase3_chunks

    phase3_pbar = tqdm(
        range(total_gradient_chunks),
        desc="Phase 3 (low-rank)",
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
        progress_bar=phase3_pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=spike_dataset.num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
        epoch_callbacks=epoch_callbacks,
    )
    # Offset Phase 3 chunk counter so logging is continuous with Phase 2
    trainer.set_checkpoint_state(phase2_chunks, float("inf"))
    phase3_pbar.n = phase2_chunks
    phase3_pbar.refresh()

    trainer.metrics_logger = metrics_logger
    if wandb_run is not None:
        trainer.wandb_logger = wandb_run
        # Skip wandb.watch — it causes torch.compile graph breaks and
        # recompilations on every parameter name. Gradient norms are
        # already logged via _extract_current_gradients.
        wandb.define_metric("epoch")

    # Handle checkpoint resuming within Phase 3
    if resume_phase == "phase3":
        phase3_checkpoint = sorted((output_dir / "checkpoints").glob("*.pt"))[-1]
        print(f"\nResuming Phase 3 from checkpoint: {phase3_checkpoint}")
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path=phase3_checkpoint,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )
        trainer.set_checkpoint_state(start_epoch, best_loss)
        phase3_pbar.n = start_epoch
        if wandb_run is not None:
            wandb.log(
                {"epoch": start_epoch / spike_dataset.num_chunks},
                step=start_epoch,
            )
        phase3_pbar.refresh()

    # =================
    # Run Phase 3
    # =================

    print(f"\nStarting Phase 3 from chunk {trainer.current_epoch}...")
    print(f"  Epochs: {phase3_epochs}")
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

    # ``model.weights_FF`` already returns the effective weight matrix —
    # ScalingFactorProjection.forward() returns ``exp(log_sf) * connectome``
    # and LowRankProjection has no per-pair SF, so each projection's per-pair
    # block already has its scaling baked in. The reported scaling_factors_FF
    # property mirrors that scaling on the kernel side and equals 1.0 for
    # LowRank pairs, so we write ones for the saved scaling factors.
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
