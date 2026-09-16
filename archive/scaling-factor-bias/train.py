"""
Is the downward bias in learnt scaling factors a property of the loss, or of training?

The setup is deliberately the simplest one that can answer that:

    * The student uses the teacher's own synaptic weights. Nothing is perturbed.
    * Half the neurons' spikes are hidden, so the student must infer them, using
      the two-layer architecture (recurrent hidden layer -> feedforward visible
      layer).
    * Per-cell-type-pair scaling factors multiply those weights and are the only
      trainable parameters. They start at exactly 1.0.

Because the weights are already correct, **scaling factors of 1.0 are the true
model**, and we have measured that they reproduce the teacher spike for spike —
the loss there is zero. So the location of the loss minimum is not in question
here, which is what makes the experiment decisive:

    if the scaling factors drift below 1.0 during training, that is an
    optimisation bias, not a bias in the objective.

Contrast with the older setup, which perturbed the weights by 1/target and asked
training to recover the target. That asks the same question through an extra
layer of indirection, and the normalised-vs-absolute scaling factor bookkeeping it
requires has been a recurring source of confusion.
"""

from pathlib import Path

import numpy as np
import toml
import torch
import wandb
from connectome_snns.configs import (
    DATALOADER_KWARGS,
    StudentHyperparameters,
    StudentSimulationConfig,
)
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.dataloaders.supervised import (
    CyclicSampler,
    ExactFFDataset,
    VisibleDrivenCollate,
)
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    make_chunked_ff_projections,
    make_scaling_factor_projections,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN
from connectome_snns.snn_runners import SNNTrainer
from connectome_snns.training_utils import AsyncLogger
from connectome_snns.training_utils.losses import VanRossumLoss
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

#: The scaling factors are initialised here, and this is also their true value —
#: the weights are already the teacher's, so nothing needs scaling.
TRUE_SCALING_FACTOR = 1.0


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    """Train scaling factors on an otherwise-correct student and watch where they go.

    Args:
        input_dir (Path): Teacher data — ``network_structure.npz``, ``spike_data.zarr``.
        output_dir (Path): Where outputs are written.
        params_file (Path): Path to ``parameters.toml``.
        wandb_config (dict, optional): W&B configuration.
        resume_from (Path, optional): Unused.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data = toml.load(params_file)
    simulation = StudentSimulationConfig(**data["simulation"])
    hyperparameters = StudentHyperparameters(**data["hyperparameters"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])
    training = data["training"]

    chunk_size = simulation.chunk_size
    hidden_cell_fraction = data["simulation"]["hidden_cell_fraction"]
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)

    # ================================================================
    # Teacher connectome. The student's weights ARE these weights — the
    # scaling factors are what training is allowed to change.
    # ================================================================

    structure = np.load(input_dir / "network_structure.npz")
    recurrent_weights = structure["recurrent_weights"] * structure[
        "recurrent_connectivity"
    ].astype(np.float32)
    feedforward_weights = structure["feedforward_weights"] * structure[
        "feedforward_connectivity"
    ].astype(np.float32)
    cell_type_indices = structure["cell_type_indices"]
    ff_cell_type_indices = structure["feedforward_cell_type_indices"]

    n_neurons = recurrent_weights.shape[0]
    n_feedforward = feedforward_weights.shape[0]

    # ================================================================
    # Split into hidden (inferred) and visible (observed) neurons.
    # ================================================================

    n_hidden = round(n_neurons * hidden_cell_fraction)
    shuffled = np.random.permutation(n_neurons)
    hidden_indices = np.sort(shuffled[:n_hidden])
    visible_indices = np.sort(shuffled[n_hidden:])
    print(f"\n{n_hidden} hidden / {len(visible_indices)} visible of {n_neurons}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "neuron_split.npz",
        hidden_indices=hidden_indices,
        visible_indices=visible_indices,
        hidden_cell_fraction=hidden_cell_fraction,
    )

    # ================================================================
    # Cell and synapse parameters, in the combined input space the
    # simulators expect (feedforward types, then recurrent types).
    # ================================================================

    rec_cells = recurrent.get_cell_params()
    ff_cells = feedforward.get_cell_params()
    rec_synapses = recurrent.get_synapse_params()
    ff_synapses = feedforward.get_synapse_params()
    n_ff_types, n_ff_synapse_types = len(ff_cells), len(ff_synapses)

    combined_cells = ff_cells.copy()
    for cell in rec_cells:
        shifted = cell.copy()
        shifted["cell_id"] = cell["cell_id"] + n_ff_types
        combined_cells.append(shifted)

    combined_synapses = ff_synapses.copy()
    for synapse in rec_synapses:
        shifted = synapse.copy()
        shifted["cell_id"] = synapse["cell_id"] + n_ff_types
        shifted["synapse_id"] = synapse["synapse_id"] + n_ff_synapse_types
        combined_synapses.append(shifted)

    rec_names = [c["name"] for c in rec_cells]
    ff_names = [c["name"] for c in ff_cells]
    combined_names = [c["name"] for c in combined_cells]

    # ======================
    # Teacher spike data
    # ======================

    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size, dt = spike_dataset.batch_size, spike_dataset.dt
    num_chunks = spike_dataset.num_chunks
    print(f"Loaded {num_chunks} chunks x {batch_size} trials, dt {dt} ms")

    # ================================================================
    # Layer 1 — the hidden neurons, genuinely recurrent among themselves.
    # Observed input is [mitral, visible teacher spikes]; the hidden
    # neurons' input to each other is the part that must be inferred.
    # ================================================================

    layer1_input_weights = np.concatenate(
        [
            feedforward_weights[:, hidden_indices],
            recurrent_weights[np.ix_(visible_indices, hidden_indices)],
        ]
    )
    layer1_input_cell_types = np.concatenate(
        [ff_cell_type_indices, cell_type_indices[visible_indices] + n_ff_types]
    )
    layer1_rec_projections, layer1_ff_projections = make_scaling_factor_projections(
        rec_weights=recurrent_weights[np.ix_(hidden_indices, hidden_indices)],
        ff_weights=layer1_input_weights,
        cell_type_indices=cell_type_indices[hidden_indices],
        ff_cell_type_indices=layer1_input_cell_types,
        cell_type_names=rec_names,
        ff_cell_type_names=combined_names,
        init_sf=TRUE_SCALING_FACTOR,
    )
    layer1 = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=layer1_rec_projections,
        ff_projections=layer1_ff_projections,
        cell_type_indices=cell_type_indices[hidden_indices],
        cell_type_indices_FF=layer1_input_cell_types,
        cell_params=rec_cells,
        cell_params_FF=combined_cells,
        synapse_params=rec_synapses,
        synapse_params_FF=combined_synapses,
        surrgrad_scale=hyperparameters.surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    # ================================================================
    # Layer 2 — the visible neurons. Input is [mitral, layer 1's inferred
    # hidden spikes, visible teacher spikes], all feedforward.
    # ================================================================

    layer2_input_weights = np.concatenate(
        [
            feedforward_weights[:, visible_indices],
            recurrent_weights[np.ix_(hidden_indices, visible_indices)],
            recurrent_weights[np.ix_(visible_indices, visible_indices)],
        ]
    )
    layer2_input_cell_types = np.concatenate(
        [
            ff_cell_type_indices,
            cell_type_indices[hidden_indices] + n_ff_types,
            cell_type_indices[visible_indices] + n_ff_types,
        ]
    )
    layer2_projections = make_chunked_ff_projections(
        rec_weights=layer2_input_weights[n_feedforward:],
        ff_weights=layer2_input_weights[:n_feedforward],
        cell_type_indices=cell_type_indices[visible_indices],
        ff_cell_type_indices=ff_cell_type_indices,
        cell_type_names=rec_names,
        ff_cell_type_names=ff_names,
        init_sf=TRUE_SCALING_FACTOR,
        # Layer 2's recurrent input rows are the hidden neurons followed by the
        # visible ones — a different set from its outputs.
        rec_source_cell_type_indices=layer2_input_cell_types[n_feedforward:]
        - n_ff_types,
    )

    # Tie the scaling-factor parameter (not the projection object) across layers:
    # one scaling factor per cell-type pair for the whole network, as intended.
    # The two layers hold different connectome blocks for the same pair, so
    # sharing the object would give layer 2 layer 1's weights.
    if training.get("share_scaling_factors", True):
        for key, projection in layer1_ff_projections.items():
            if key in layer2_projections:
                layer2_projections[key].log_sf = projection.log_sf

    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=layer2_projections,
        cell_type_indices=cell_type_indices[visible_indices],
        cell_type_indices_FF=layer2_input_cell_types,
        cell_params=rec_cells,
        cell_params_FF=combined_cells,
        synapse_params_FF=combined_synapses,
        surrgrad_scale=hyperparameters.surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    model = TwoLayerSNN(layer1, layer2, n_ff=n_feedforward).to(device)
    n_parameters = len({id(p) for p in model.parameters()})
    print(f"Model: TwoLayerSNN, {n_parameters} trainable scaling factors, all at 1.0")

    # ==========================================================
    # Reading the scaling factors back out, for logging.
    # ==========================================================

    def current_scaling_factors() -> dict:
        """Absolute scaling factor per pathway. 1.0 is the truth, by construction."""
        values = {}
        for (source, target), projection in layer1_ff_projections.items():
            values[f"{source}_to_{target}"] = float(
                np.exp(projection.log_sf.detach().cpu().numpy())
            )
        for (source, target), projection in layer1_rec_projections.items():
            values[f"{source}_to_{target}"] = float(
                np.exp(projection.log_sf.detach().cpu().numpy())
            )
        return values

    def stats_computer(snapshot):
        """Log the scaling factors and the visible firing rates.

        Both the student's and the teacher's rates come from the snapshot's own
        tensors, so they always cover exactly the same timesteps. Reconstructing
        the teacher's window from a chunk index is what silently went wrong in the
        older experiments.
        """
        student_spikes = snapshot["spikes"][0]
        teacher_spikes = snapshot["target_spikes"][0, : student_spikes.shape[0]]
        duration_s = student_spikes.shape[0] * dt / 1000.0

        stats = {}
        visible_cell_types = cell_type_indices[visible_indices]
        for type_index, type_name in enumerate(rec_names):
            members = visible_cell_types == type_index
            if members.any():
                stats[f"firing_rate/student_{type_name}"] = float(
                    student_spikes[:, members].sum() / duration_s / members.sum()
                )
                stats[f"firing_rate/teacher_{type_name}"] = float(
                    teacher_spikes[:, members].sum() / duration_s / members.sum()
                )

        for pathway, value in current_scaling_factors().items():
            stats[f"scaling_factors/{pathway}_value"] = value
            stats[f"scaling_factors/{pathway}_target"] = TRUE_SCALING_FACTOR
        return stats

    # ==============================
    # Loss, optimiser, dataloader
    # ==============================

    van_rossum = VanRossumLoss(
        tau_rise=hyperparameters.van_rossum_tau_rise,
        tau_decay=hyperparameters.van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        betas=(hyperparameters.beta1, hyperparameters.beta2),
    )
    total_epochs = training["total_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_epochs, eta_min=hyperparameters.lr_min
    )
    scaler = GradScaler(
        "cuda", enabled=training["mixed_precision"] and device == "cuda"
    )

    dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=VisibleDrivenCollate(torch.from_numpy(visible_indices).long()),
    )

    wandb_logger = None
    if wandb_config:
        run_name = wandb_config.pop("name", None) or output_dir.name
        wandb_logger = wandb.init(
            name=run_name,
            dir=str(output_dir),
            **{
                **wandb_config,
                "config": {
                    "hidden_cell_fraction": hidden_cell_fraction,
                    "n_hidden": n_hidden,
                    "weights": "teacher (unperturbed)",
                    "true_scaling_factor": TRUE_SCALING_FACTOR,
                },
            },
        )
        wandb.define_metric("epoch")

    # ==============================
    # Train
    # ==============================

    total_chunks = total_epochs * num_chunks
    print(f"\nScaling factors at start: {current_scaling_factors()}")
    print(f"Training for {total_epochs} epochs ({total_chunks} chunks)\n")

    trainer = SNNTrainer(
        model=model,
        optimizer=optimiser,
        scaler=scaler,
        dataloader=dataloader,
        loss_functions={"van_rossum": van_rossum},
        loss_weights={"van_rossum": hyperparameters.loss_weight.van_rossum},
        device=device,
        num_epochs=total_chunks,
        chunks_per_update=training["chunks_per_update"],
        log_interval=training["log_interval"],
        checkpoint_interval=training["checkpoint_interval"],
        plot_size=training["plot_size"],
        mixed_precision=training["mixed_precision"],
        grad_norm_clip=hyperparameters.grad_norm_clip,
        progress_bar=tqdm(range(total_chunks), desc="Training", unit="chunk"),
        stats_computer=stats_computer,
        chunks_per_data_epoch=num_chunks,
        burn_in_chunks=training["burn_in_chunks"],
        scheduler=scheduler,
    )
    trainer.metrics_logger = AsyncLogger(log_dir=output_dir, max_queue_size=10)
    if wandb_logger:
        trainer.wandb_logger = wandb_logger

    best_loss = trainer.train(output_dir=output_dir)

    # ==============================
    # Report
    # ==============================

    final = current_scaling_factors()
    np.savez(
        output_dir / "final_scaling_factors.npz",
        **{pathway: value for pathway, value in final.items()},
        true_scaling_factor=TRUE_SCALING_FACTOR,
    )

    print(f"\n{'=' * 64}")
    print("Training complete. Scaling factors relative to the truth (1.0):")
    for pathway, value in sorted(final.items()):
        drift = value - TRUE_SCALING_FACTOR
        print(f"  {pathway:32s} {value:.4f}   ({drift:+.4f})")
    below = sum(value < TRUE_SCALING_FACTOR for value in final.values())
    print(f"\n  {below}/{len(final)} pathways ended BELOW the true value")
    print(f"  best loss: {best_loss:.6f}  (the truth scores 0)")
    print(f"{'=' * 64}")

    trainer.metrics_logger.close()
    if wandb_logger:
        wandb.finish()
    return best_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
