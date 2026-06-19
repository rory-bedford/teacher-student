"""
Minimal single-neuron fully-observed training with Van Rossum loss only.

No CMA-ES, no firing rate loss, no silence penalty. Trains a single output
neuron's scaling factors using Van Rossum distance. Useful as a fast baseline
to verify the training pipeline works before adding complexity.

Uses the same parameters.toml as train.py (extra fields are ignored).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

from dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    single_neuron_collate_fn,
)
from network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from network_simulators.projections import ScalingFactorProjection
from training_utils.losses import VanRossumLoss
from training_utils import load_checkpoint
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


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Train scaling factors for a single neuron using Van Rossum loss only.

    Args:
        input_dir (Path): Directory containing teacher data
        output_dir (Path): Directory where training outputs will be saved
        params_file (Path): Path to the parameters TOML file
        wandb_config (dict, optional): W&B configuration
        resume_from (Path, optional): Path to checkpoint to resume from
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(params_file, "r") as f:
        data = toml.load(f)

    simulation = StudentSimulationConfig(**data["simulation"])
    training = StudentTrainingConfig(**data["training"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    scaling_factors = data.get("scaling_factors", {})

    chunk_size = simulation.chunk_size
    seed = simulation.seed
    epochs = training.epochs
    chunks_per_update = training.chunks_per_update
    log_interval = training.log_interval
    checkpoint_interval = training.checkpoint_interval
    plot_size = training.plot_size
    mixed_precision = training.mixed_precision
    grad_norm_clip = hyperparameters.grad_norm_clip
    weight_perturbation_variance = training.weight_perturbation_variance
    beta1 = hyperparameters.beta1
    beta2 = hyperparameters.beta2
    learning_rate = hyperparameters.learning_rate
    surrgrad_scale = hyperparameters.surrgrad_scale
    van_rossum_tau_rise = hyperparameters.van_rossum_tau_rise
    van_rossum_tau_decay = hyperparameters.van_rossum_tau_decay
    burn_in_chunks = training.burn_in_chunks

    # Single neuron index to train
    neuron_idx = 0

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    # Get cell and synapse parameters
    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()

    n_ff_cell_types = len(feedforward_cell_params)
    n_ff_synapse_types = len(feedforward_synapse_params)

    # Get base scaling factors from config
    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # ================================================================
    # RESUME PATH
    # ================================================================
    if resume_from is not None:
        print("\n" + "=" * 26)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 26)

        initial_state_dir = output_dir / "initial_state"
        initial_state = np.load(initial_state_dir / "network_structure.npz")
        single_weights = initial_state["feedforward_weights"]
        single_mask = initial_state["feedforward_connectivity"]
        cell_type_indices = initial_state["cell_type_indices"]
        single_cell_type_indices = initial_state["single_cell_type_indices"]
        concatenated_cell_type_indices = initial_state["feedforward_cell_type_indices"]
        neuron_idx = int(initial_state["neuron_idx"])

        n_total_inputs = single_weights.shape[0]

        targets_dir = output_dir / "targets"
        saved_targets = np.load(targets_dir / "target_scaling_factors.npz")
        target_scaling_factors_FF = saved_targets["feedforward_scaling_factors"]

    # ================================================================
    # FRESH START PATH
    # ================================================================
    else:
        print("\n" + "=" * 27)
        print("Perturbing Scaling Factors")
        print("=" * 27)

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

        concatenated_weights = np.concatenate([feedforward_weights, weights], axis=0)
        concatenated_mask = np.concatenate([feedforward_mask, recurrent_mask], axis=0)
        concatenated_cell_type_indices = np.concatenate(
            [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
        )

        # Sample target scaling factors and apply perturbation (full network)
        sigma = np.sqrt(weight_perturbation_variance)
        mu = -(sigma**2) / 2.0

        target_scaling_factors_FF = np.random.lognormal(
            mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
        )
        perturbation_factors = 1.0 / target_scaling_factors_FF

        perturbed_weights = concatenated_weights.copy()
        for input_idx in range(n_total_inputs):
            input_type = concatenated_cell_type_indices[input_idx]
            for output_idx in range(n_neurons):
                output_type = cell_type_indices[output_idx]
                perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                    input_type, output_type
                ]

        # Slice to single neuron
        single_weights = perturbed_weights[:, neuron_idx : neuron_idx + 1]
        single_mask = concatenated_mask[:, neuron_idx : neuron_idx + 1]
        single_cell_type_indices = cell_type_indices[neuron_idx : neuron_idx + 1]

        neuron_type = cell_type_indices[neuron_idx]
        neuron_type_name = recurrent.cell_types.names[neuron_type]
        print(f"\nSelected neuron {neuron_idx} (type: {neuron_type_name})")

        normalized_initial = concatenated_scaling_factors / target_scaling_factors_FF
        print(
            f"Initial scaling factors / target (should recover to 1.0):\n"
            f"{normalized_initial}"
        )

        # Save initial state
        initial_state_dir = output_dir / "initial_state"
        initial_state_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            initial_state_dir / "network_structure.npz",
            feedforward_weights=single_weights,
            feedforward_connectivity=single_mask,
            cell_type_indices=cell_type_indices,
            single_cell_type_indices=single_cell_type_indices,
            feedforward_cell_type_indices=concatenated_cell_type_indices,
            neuron_idx=neuron_idx,
        )

        targets_dir = output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            targets_dir / "target_scaling_factors.npz",
            feedforward_scaling_factors=target_scaling_factors_FF,
        )

    # ================================================================
    # Common setup
    # ================================================================

    print(f"\nSingle-neuron feedforward: {n_total_inputs} inputs -> 1 neuron")
    print(f"Active connections: {single_mask.sum():,}")

    # Concatenate cell params: feedforward + recurrent (with offset cell_ids)
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cell_params in recurrent_cell_params:
        offset_cell_params = cell_params.copy()
        offset_cell_params["cell_id"] = cell_params["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(offset_cell_params)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for syn_params in recurrent_synapse_params:
        offset_syn_params = syn_params.copy()
        offset_syn_params["cell_id"] = syn_params["cell_id"] + n_ff_cell_types
        offset_syn_params["synapse_id"] = syn_params["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(offset_syn_params)

    # Load dataset
    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size = spike_dataset.batch_size
    print(f"Loaded {spike_dataset.num_chunks} chunks x {batch_size} batch size")

    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=single_neuron_collate_fn,
    )

    # ================================================================
    # Model (single neuron output)
    # ================================================================

    # Build per-pair ScalingFactorProjections. Inputs span the combined
    # cell-type space (FF cell types + recurrent cell types offset by
    # n_ff_cell_types). The single output neuron has exactly one cell
    # type — only pairs with that target are created.
    input_cell_type_names_combined = (
        feedforward.cell_types.names + recurrent.cell_types.names
    )
    target_id = int(single_cell_type_indices[0])
    target_name = recurrent.cell_types.names[target_id]
    # Combined-input id for each pair source: FF ids in [0, n_ff_cell_types),
    # recurrent ids offset by n_ff_cell_types. The scaling-factor matrix
    # `concatenated_scaling_factors` is indexed the same way.
    single_weights_masked = single_weights * single_mask
    projections: dict[tuple[str, str], ScalingFactorProjection] = {}
    for combined_src_id, src_name in enumerate(input_cell_type_names_combined):
        src_row_mask = concatenated_cell_type_indices == combined_src_id
        if not src_row_mask.any():
            continue
        block = single_weights_masked[src_row_mask, :].astype(np.float32)
        init_sf = float(concatenated_scaling_factors[combined_src_id, target_id])
        projections[(src_name, target_name)] = ScalingFactorProjection(
            connectome=block,
            init_sf=init_sf,
        )

    model = FeedforwardConductanceLIFNetwork(
        dt=spike_dataset.dt,
        projections=projections,
        cell_type_indices=single_cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    model.to(device)

    print("\nModel:")
    print(f"  Scaling factors shape: {model.scaling_factors_FF.shape}")
    print(f"  Weights shape: {model.weights_FF.shape}")

    # ================================================================
    # Loss (Van Rossum only)
    # ================================================================

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=van_rossum_tau_rise,
        tau_decay=van_rossum_tau_decay,
        dt=spike_dataset.dt,
        window_size=chunk_size,
        device=device,
    )

    loss_functions = {"van_rossum": van_rossum_loss_fn}
    loss_weights = {"van_rossum": hyperparameters.loss_weight.van_rossum}

    # ================================================================
    # Optimizer
    # ================================================================

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )
    scaler = GradScaler("cuda", enabled=mixed_precision and device == "cuda")

    # ================================================================
    # Plotting and stats
    # ================================================================

    def plot_generator(spikes, input_spikes, target_spikes, **kwargs):
        interleaved = np.zeros((1, spikes.shape[1], 2))
        interleaved[0, :, 0] = target_spikes[0, :, 0]
        interleaved[0, :, 1] = spikes[0, :, 0]

        fig = plot_spike_trains(
            spikes=interleaved,
            dt=spike_dataset.dt,
            cell_type_indices=np.array([0, 1]),
            cell_type_names=["Target", "Trained"],
            n_neurons_plot=2,
            fraction=1.0,
            random_seed=None,
            title=f"Single neuron {neuron_idx}: Target vs Trained",
            ylabel="",
            figsize=(14, 4),
        )
        return {"spike_comparison": fig}

    dt = spike_dataset.dt
    num_chunks = spike_dataset.num_chunks

    input_cell_type_names = feedforward.cell_types.names + recurrent.cell_types.names
    output_cell_type_names = recurrent.cell_types.names

    def stats_computer(snapshot):
        # spikes shape: (batch, time, 1) — single neuron
        student_spikes = snapshot["spikes"][0, :, 0]
        n_timesteps = student_spikes.shape[0]
        duration_s = n_timesteps * dt / 1000.0

        # target_spikes from single_neuron_collate_fn: (batch, time, 1)
        teacher_spikes = snapshot["target_spikes"][0, :n_timesteps, 0]

        student_rate = float(student_spikes.sum()) / duration_s
        teacher_rate = float(teacher_spikes.sum()) / duration_s

        stats = {
            "firing_rate/student": student_rate,
            "firing_rate/teacher": teacher_rate,
        }

        # Log scaling factors normalized by target
        current_sf = snapshot["scaling_factors_FF"]
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
                stats[f"scaling_factors/{synapse_name}_value"] = float(normalized)
                stats[f"scaling_factors/{synapse_name}_target"] = 1.0

        return stats

    # ================================================================
    # Training
    # ================================================================

    print("\n" + "=" * 42)
    print("Single-Neuron Training (Van Rossum only)")
    print("=" * 42)

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

    pbar = tqdm(range(num_epochs), desc="Training", unit="chunk", total=num_epochs)

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=spike_dataloader,
        loss_functions=loss_functions,
        loss_weights=loss_weights,
        device=device,
        num_epochs=num_epochs,
        chunks_per_update=chunks_per_update,
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
        plot_size=plot_size,
        mixed_precision=mixed_precision,
        grad_norm_clip=grad_norm_clip,
        progress_bar=pbar,
        plot_generator=plot_generator,
        stats_computer=stats_computer,
        chunks_per_data_epoch=spike_dataset.num_chunks,
        burn_in_chunks=burn_in_chunks,
        scheduler=scheduler,
    )

    # Wandb
    if wandb_config:
        wandb_init_config = {
            "simulation": simulation.model_dump(),
            "training": training.model_dump(),
            "hyperparameters": hyperparameters.model_dump(),
            "script": "train_single",
            "neuron_idx": neuron_idx,
            "output_dir": str(output_dir),
            "device": device,
        }
        run_name = wandb_config.pop("name", None) or (
            output_dir.name if output_dir else "fully-observed-single"
        )
        wandb_run = wandb.init(
            name=run_name,
            config=wandb_init_config,
            dir=str(output_dir) if output_dir else None,
            **wandb_config,
        )
        trainer.wandb_logger = wandb_run
        wandb.watch(model, log="parameters", log_freq=log_interval)
        wandb.define_metric("epoch")

    # ================================================================
    # Pre-training inference (baseline metrics)
    # ================================================================

    print("\nRunning pre-training inference...")
    model.reset_state(batch_size=batch_size)
    model.track_variables = False

    all_spikes = []
    all_inputs = []
    all_targets = []
    pre_train_losses = {}
    pre_train_iter = iter(spike_dataloader)

    with torch.no_grad():
        for chunk_idx in range(min(plot_size, num_chunks)):
            batch_data = next(pre_train_iter)
            input_spikes = batch_data.input_spikes.to(device)
            target_spikes = batch_data.target_spikes.to(device)

            with torch.amp.autocast(
                "cuda", enabled=mixed_precision and device == "cuda"
            ):
                output_spikes = model.forward(input_spikes=input_spikes)

                for loss_name, loss_fn in loss_functions.items():
                    loss_val = loss_fn(
                        output_spikes=output_spikes, target_spikes=target_spikes
                    )
                    if loss_name not in pre_train_losses:
                        pre_train_losses[loss_name] = []
                    pre_train_losses[loss_name].append(loss_val.item())

            all_spikes.append(output_spikes.detach().cpu().numpy())
            all_inputs.append(input_spikes.detach().cpu().numpy())
            all_targets.append(target_spikes.detach().cpu().numpy())

    spikes_arr = np.concatenate(all_spikes, axis=1)
    inputs_arr = np.concatenate(all_inputs, axis=1)
    targets_arr = np.concatenate(all_targets, axis=1)

    # Compute stats
    pre_snapshot = {
        "spikes": spikes_arr,
        "scaling_factors_FF": model.scaling_factors_FF.detach().cpu().numpy(),
        "target_spikes": targets_arr,
    }
    pre_stats = stats_computer(pre_snapshot)

    # Compute mean losses
    for loss_name, values in pre_train_losses.items():
        mean_val = np.mean(values)
        pre_stats[f"loss/{loss_name}"] = mean_val
        print(f"  {loss_name}: {mean_val:.6f}")

    # Generate plot
    pre_plots = plot_generator(
        spikes=spikes_arr, input_spikes=inputs_arr, target_spikes=targets_arr
    )

    # Log to wandb
    if wandb_config:
        wandb_log = {"epoch": 0.0}
        for k, v in pre_stats.items():
            wandb_log[k] = v
        for k, fig in pre_plots.items():
            wandb_log[k] = wandb.Image(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)
        wandb.log(wandb_log)
        print("  Logged pre-training metrics to wandb")

    # Save plot to disk
    for plot_name, fig in pre_plots.items():
        plot_path = output_dir / f"pre_training_{plot_name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(f"  Saved {plot_path}")

    # Reset model and loss states for training
    model.reset_state(batch_size=batch_size)
    for loss_fn in loss_functions.values():
        if hasattr(loss_fn, "reset_state"):
            loss_fn.reset_state()

    # Resume from checkpoint
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

    # Run
    chunk_duration_s = chunk_size * dt / 1000.0
    neuron_type_name = recurrent.cell_types.names[cell_type_indices[neuron_idx]]
    print(f"\nNeuron: {neuron_idx} ({neuron_type_name})")
    print(f"Chunk: {chunk_size} timesteps ({chunk_duration_s:.1f}s)")
    print(f"Total: {num_epochs} chunks, {epochs} epochs, batch_size={batch_size}")

    model.reset_state(batch_size=batch_size)
    model.track_variables = False

    best_loss = trainer.train(output_dir=output_dir)

    # ================================================================
    # Save final state
    # ================================================================

    print(f"\nTraining complete! Best loss: {best_loss:.6f}")

    final_state_dir = output_dir / "final_state"
    final_state_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        final_state_dir / "network_structure.npz",
        feedforward_weights=model.weights_FF.detach().cpu().numpy(),
        feedforward_connectivity=single_mask,
        cell_type_indices=cell_type_indices,
        single_cell_type_indices=single_cell_type_indices,
        feedforward_cell_type_indices=concatenated_cell_type_indices,
        scaling_factors_FF=model.scaling_factors_FF.detach().cpu().numpy(),
        neuron_idx=neuron_idx,
    )

    if trainer.metrics_logger:
        trainer.metrics_logger.close()
    if wandb_config:
        wandb.finish()

    print(f"\nOutputs: {output_dir}")
