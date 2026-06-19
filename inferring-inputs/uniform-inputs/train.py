"""
Inferring-inputs: uniform-inputs training with CMA-ES initialisation.

Identical to fully-observed training except that feedforward inputs are
replaced by uniform 6 Hz Poisson spike trains — no structured input signal.
Recurrent (output) spikes are loaded exactly from disk as training targets.

Training proceeds in two phases:
1. CMA-ES: Gradient-free search over scaling factors from initial values
2. Gradient-based: Adam optimisation from CMA-ES solution
"""

import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from dataloaders.supervised import (
    HomogeneousPoissonFFDataset,
    CyclicSampler,
    FeedforwardCollate,
)
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.projections import (
    make_chunked_ff_projections,
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
from snn_runners import SNNTrainer, EvolutionarySearch
import toml
import wandb
from visualization.neuronal_dynamics import plot_spike_trains


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train all neurons to match target spike train activity with uniform Poisson inputs.

    Feedforward inputs are uniform Poisson spike trains at a fixed rate (no
    structured input signal).  Recurrent targets are loaded exactly from disk.
    Two-phase training:
    1. CMA-ES evolutionary search over scaling factors
    2. Gradient-based optimisation from CMA-ES solution

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
    firing_rate_hz = float(data["inputs"]["firing_rate_hz"])

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
        header = "RESUMING FROM CHECKPOINT"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        initial_state_dir = output_dir / "initial_state"
        print(f"\nLoading initial state from {initial_state_dir}...")

        initial_state = np.load(initial_state_dir / "network_structure.npz")
        perturbed_weights = initial_state["feedforward_weights"]
        concatenated_mask = initial_state["feedforward_connectivity"]
        cell_type_indices = initial_state["cell_type_indices"]
        concatenated_cell_type_indices = initial_state["feedforward_cell_type_indices"]

        n_neurons = perturbed_weights.shape[1]
        n_total_inputs = perturbed_weights.shape[0]

        print(f"  Loaded perturbed weights: {perturbed_weights.shape}")
        print(f"  Active connections: {concatenated_mask.sum():,}")

        # Load saved targets
        targets_dir = output_dir / "targets"
        saved_targets = np.load(targets_dir / "target_scaling_factors.npz")
        target_scaling_factors_FF = saved_targets["feedforward_scaling_factors"]
        print(f"  Loaded target scaling factors: {target_scaling_factors_FF.shape}")

    # ================================================================
    # FRESH START PATH: Apply perturbation and save initial state
    # ================================================================
    else:
        header = "Perturbing Scaling Factors"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

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

        # Print in normalized space (current/target), matching wandb convention
        normalized_initial = concatenated_scaling_factors / target_scaling_factors_FF
        print(
            f"\nInitial scaling factors / target (should recover to 1.0):\n{normalized_initial}"
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
        )

        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            targets_dir / "target_scaling_factors.npz",
            feedforward_scaling_factors=target_scaling_factors_FF,
        )

        print(f"\nSaved initial perturbed state to {initial_state_dir}")

    # ================================================================
    # Common setup (both paths converge here)
    # ================================================================

    print("\nFeedforward network setup:")
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

    print("\nCombined parameters:")
    print(
        f"  - Cell types: {n_ff_cell_types} FF + {n_rec_cell_types} rec = {len(combined_cell_params_FF)}"
    )
    print(
        f"  - Synapse types: {n_ff_synapse_types} FF + {len(recurrent_synapse_params)} rec = {len(combined_synapse_params_FF)}"
    )

    # ======================
    # Load Dataset from Disk
    # ======================

    spike_dataset = HomogeneousPoissonFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
        firing_rate_override=firing_rate_hz,
        recurrent_smoothing_tau=training.recurrent_smoothing_tau,
    )

    batch_size = spike_dataset.batch_size

    print(f"\nLoaded {spike_dataset.num_chunks} chunks x {batch_size} batch size")
    print(f"  Uniform Poisson FF rate: {firing_rate_hz} Hz")

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
    # the chunked-FF model). ``make_chunked_ff_projections`` expects them
    # passed separately.
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

    # CMA-ES loss weights (used for evolutionary search)
    cma_loss_weights = {
        "van_rossum": loss_weight_van_rossum,
    }

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
            "inputs": {"firing_rate_hz": firing_rate_hz},
            "output_dir": str(output_dir),
            "device": device,
        }

        run_name = wandb_config.pop("name", None) or (
            output_dir.name if output_dir else "inferring-inputs-uniform"
        )
        wandb_run = wandb.init(
            name=run_name,
            config=wandb_init_config,
            dir=str(output_dir) if output_dir else None,
            **wandb_config,
        )

    # Initialize AsyncLogger for disk logging (shared across CMA-ES and gradient training)
    metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)

    # ================================================
    # Phase 1: CMA-ES Evolutionary Search
    # ================================================

    # Track best scaling factors from CMA-ES (or use initial ones)
    best_scaling_factors_FF = concatenated_scaling_factors

    # Check if CMA-ES already completed (resume loads saved results)
    cma_state_path = output_dir / "cma_es_state" / "scaling_factors.npz"
    if resume_from is not None and cma_state_path.exists():
        print("\nLoading CMA-ES results from previous run...")
        cma_saved = np.load(cma_state_path)
        best_scaling_factors_FF = cma_saved["scaling_factors_FF"]
        best_cma_loss = float(cma_saved["best_loss"])
        normalized_sf = best_scaling_factors_FF / target_scaling_factors_FF
        print(f"  Scaling factors / target: {normalized_sf}")
        print(f"  Best CMA-ES loss: {best_cma_loss:.6f}")

    elif resume_from is None and cma_es_config:
        header = "PHASE 1: CMA-ES Evolutionary Search"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))

        # ScalingFactorProjections for CMA-ES: connectome is the per-pair
        # (already perturbation-baked) block; init SF = teacher's configured
        # value. The CMA update_fn writes new SFs into each projection's
        # ``log_sf`` buffer.
        cma_projections = make_chunked_ff_projections(
            rec_weights=perturbed_rec_block,
            ff_weights=perturbed_ff_block,
            cell_type_indices=cell_type_indices,
            ff_cell_type_indices=ff_cell_type_indices_only,
            cell_type_names=rec_output_cell_type_names,
            ff_cell_type_names=feedforward.cell_types.names,
            init_sf=1.0,
        )
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

        print("\nCMA-ES model initialized:")
        print(f"  - Output neurons: {n_neurons}")
        print(f"  - {n_total_inputs} feedforward inputs per neuron")
        print(f"  - Scaling factors shape: {cma_model.scaling_factors_FF.shape}")

        sf_shape = concatenated_scaling_factors.shape

        # Search in log space so scaling factors stay positive
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

        # CMA-ES needs its own VanRossumLoss instance (has internal state)
        cma_van_rossum_fn = VanRossumLoss(
            tau_rise=van_rossum_tau_rise,
            tau_decay=van_rossum_tau_decay,
            dt=spike_dataset.dt,
            window_size=chunk_size,
            device=device,
        )
        cma_loss_functions = {
            "van_rossum": cma_van_rossum_fn,
        }
        # Create a separate dataloader for CMA-ES
        cma_dataloader = DataLoader(
            spike_dataset,
            batch_size=None,
            sampler=CyclicSampler(spike_dataset),
            **DATALOADER_KWARGS,
            collate_fn=collate_fn,
        )

        # Build cell type name lists for scaling factor logging
        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

        # Preload one chunk for computing stats on best candidate each generation
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

        # Wandb + disk callback for CMA-ES generation logging
        def cma_es_callback(metrics):
            log_dict = {
                "cma_es/best_loss": metrics["best_loss"],
                "cma_es/sigma": metrics["sigma"],
                "cma_es/n_evals": metrics["n_evals"],
            }

            # Log per-loss means across the population
            for loss_name, loss_val in metrics["mean_losses"].items():
                log_dict[f"cma_es_loss/{loss_name}"] = loss_val

            # Log scaling factors normalized by target (should approach 1)
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

            # Run forward pass on best candidate to compute firing rate stats
            cma_model.reset_state(batch_size=1)
            cma_model.track_variables = False
            cma_model.track_batch_idx = None
            with torch.inference_mode():
                spikes = cma_model.forward(input_spikes=cma_eval_input)

            spikes_np = spikes[0].detach().cpu().numpy()  # (time, n_neurons)
            n_timesteps = spikes_np.shape[0]
            duration_s = n_timesteps * cma_eval_dt / 1000.0

            student_rates = spikes_np.sum(axis=0) / duration_s
            log_dict["cma_es_firing_rate/student_mean"] = float(student_rates.mean())
            log_dict["cma_es_firing_rate/student_std"] = float(student_rates.std())
            log_dict["cma_es_firing_rate/student_min"] = float(student_rates.min())
            log_dict["cma_es_firing_rate/student_max"] = float(student_rates.max())

            if cma_eval_target is not None:
                teacher_np = cma_eval_target[0].detach().cpu().numpy()
                teacher_rates = teacher_np.sum(axis=0) / duration_s
                log_dict["cma_es_firing_rate/teacher_mean"] = float(
                    teacher_rates.mean()
                )
                log_dict["cma_es_firing_rate/teacher_std"] = float(teacher_rates.std())
                log_dict["cma_es_firing_rate/teacher_min"] = float(teacher_rates.min())
                log_dict["cma_es_firing_rate/teacher_max"] = float(teacher_rates.max())

            # Per-cell-type firing rates
            for type_idx, type_name in enumerate(output_cell_type_names):
                type_mask = cell_type_indices == type_idx
                if type_mask.sum() > 0:
                    student_type_rates = student_rates[type_mask]
                    log_dict[f"cma_es_firing_rate/student_{type_name}_mean"] = float(
                        student_type_rates.mean()
                    )
                    log_dict[f"cma_es_firing_rate/student_{type_name}_std"] = float(
                        student_type_rates.std()
                    )
                    if cma_eval_target is not None:
                        teacher_type_rates = teacher_rates[type_mask]
                        log_dict[f"cma_es_firing_rate/teacher_{type_name}_mean"] = (
                            float(teacher_type_rates.mean())
                        )
                        log_dict[f"cma_es_firing_rate/teacher_{type_name}_std"] = float(
                            teacher_type_rates.std()
                        )

            # Log to disk (negative epoch to distinguish CMA-ES from gradient training)
            metrics_logger.log(epoch=-metrics["generation"], **log_dict)

            # Log to wandb
            if wandb_run is not None:
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
            initial_sigma=cma_es_config.get("initial_sigma", 0.1),
            max_evaluations=cma_es_config.get("max_evaluations", 200),
            popsize=cma_es_config.get("popsize", None),
            batch_size=cma_es_config.get("batch_size", None),
            seed=cma_es_config.get("seed", None),
            callback=cma_es_callback,
            burn_in_chunks=burn_in_chunks,
        )

        best_cma_loss = searcher.search()

        # Extract best scaling factors from CMA-ES model
        best_scaling_factors_FF = cma_model.scaling_factors_FF.detach().cpu().numpy()

        normalized_sf = best_scaling_factors_FF / target_scaling_factors_FF
        print(f"\nCMA-ES best loss: {best_cma_loss:.6f}")
        print(f"Scaling factors / target (should be 1): {normalized_sf}")

        # Save post-CMA-ES state
        cma_state_dir = output_dir / "cma_es_state"
        cma_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cma_state_dir / "scaling_factors.npz",
            scaling_factors_FF=best_scaling_factors_FF,
            best_loss=best_cma_loss,
        )
        print(f"Saved CMA-ES state to {cma_state_dir}")

        # Free all CMA-ES GPU memory before gradient phase
        searcher.update_fn = None
        searcher.callback = None
        searcher._chunks = []
        searcher.model = None
        searcher.loss_functions = {}
        del cma_model, searcher
        del cma_van_rossum_fn, cma_loss_functions
        del cma_dataloader, cma_update_fn, cma_es_callback
        del cma_eval_input, cma_eval_target, cma_eval_batch, cma_eval_iter
        gc.collect()
        torch.cuda.empty_cache()

    # ==============================================
    # Initialize Gradient Training Model (Phase 2)
    # ==============================================

    if optimisable != "scaling_factors":
        raise NotImplementedError(
            f"uniform-inputs supports optimisable='scaling_factors' only; "
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

    print("\nGradient model initialized (optimisable=%s):" % optimisable)
    print(f"  - Output neurons: {n_neurons}")
    print(f"  - {n_total_inputs} feedforward inputs per neuron")
    print(f"  - Scaling factors shape: {model.scaling_factors_FF.shape}")

    # ==============================
    # Setup Optimizer for Phase 2
    # ==============================

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )
    lr_min = getattr(hyperparameters, "lr_min", None)
    if lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr_min
        )
        print(f"  LR schedule: cosine {learning_rate} → {lr_min} over {epochs} epochs")
    else:
        scheduler = None
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
            title=f"Uniform-inputs: Target vs Trained (first {n_plot} neurons)",
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

        # Teacher spikes come from the trainer's accumulated batch data.
        teacher_spikes = snapshot["target_spikes"][0, :n_timesteps, :]

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

        # Scaling factor tracking
        current_sf = snapshot["scaling_factors_FF"]
        target_sf = target_scaling_factors_FF

        input_cell_type_names = (
            feedforward.cell_types.names + recurrent.cell_types.names
        )
        output_cell_type_names = recurrent.cell_types.names

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
    # Phase 2: Gradient-Based Training
    # ===================

    header = "PHASE 2: Gradient-Based Training"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Update feedforward cell type names for trainer labeling
    feedforward.cell_types.names = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )

    num_epochs = epochs * spike_dataset.num_chunks

    pbar = tqdm(
        range(num_epochs),
        desc="Training",
        unit="chunk",
        total=num_epochs,
    )

    # Gradient phase uses only Van Rossum loss
    gradient_loss_functions = {"van_rossum": van_rossum_loss_fn}
    gradient_loss_weights = {"van_rossum": loss_weight_van_rossum}

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
        wandb_config=None,  # Already initialized above
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=spike_dataset.num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
    )

    # Share pre-initialized loggers with the trainer
    trainer.metrics_logger = metrics_logger
    if wandb_run is not None:
        trainer.wandb_logger = wandb_run
        wandb.watch(model, log="parameters", log_freq=log_interval)
        wandb.define_metric("epoch")

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

    print("\nSaving final network structure...")
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=model.weights_FF.detach().cpu().numpy(),
        feedforward_connectivity=concatenated_mask,
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=model.scaling_factors_FF.detach().cpu().numpy(),
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
