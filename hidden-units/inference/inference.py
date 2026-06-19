"""
Inference with scaling factors at target values for the hidden-units experiment.

Establishes baseline performance ("correct student") with hidden units removed.
No training — just runs inference with SF = target to see achievable performance
when the model architecture is mis-specified (missing hidden neurons).

Same hidden-unit masking logic as train.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_snns.dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    SpikeData,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import FrozenProjection
import torch
import functools
from torch.utils.data import DataLoader
from connectome_snns.training_utils.losses import VanRossumLoss
from connectome_snns.configs import (
    DATALOADER_KWARGS,
    StudentSimulationConfig,
    StudentTrainingConfig,
    StudentHyperparameters,
)
from connectome_snns.configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
import toml
from tqdm import tqdm
from connectome_snns.visualization.neuronal_dynamics import plot_spike_trains


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


def hidden_units_collate_fn(batch, visible_indices_tensor):
    """Collate that provides [FF, visible_rec] as input, visible-only targets."""
    visible_rec = batch.target_spikes[:, :, visible_indices_tensor]
    combined_input = torch.cat([batch.input_spikes, visible_rec], dim=2)
    return SpikeData(input_spikes=combined_input, target_spikes=visible_rec)


def main(
    input_dir,
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
):
    """Run inference with scaling factors at target values and hidden units removed.

    Args:
        input_dir (Path): Directory containing teacher data (network_structure.npz, spike_data.zarr)
        output_dir (Path): Directory where outputs will be saved
        params_file (Path): Path to the parameters TOML file
        wandb_config (dict, optional): W&B configuration (not used)
        resume_from (Path, optional): Not used
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

    hidden_units_config = data.get("hidden_units", {})
    hidden_unit_fraction = hidden_units_config.get("hidden_unit_fraction", 0.1)

    chunk_size = simulation.chunk_size
    seed = simulation.seed
    plot_size = training.plot_size
    weight_perturbation_variance = training.weight_perturbation_variance

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    hidden_units_rng = np.random.default_rng(seed)

    # Load network structure
    network_structure = np.load(input_dir / "network_structure.npz")

    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    feedforward_cell_type_indices = network_structure["feedforward_cell_type_indices"]
    recurrent_mask = network_structure["recurrent_connectivity"]
    feedforward_mask = network_structure["feedforward_connectivity"]

    n_neurons = weights.shape[0]

    recurrent_cell_params = recurrent.get_cell_params()
    feedforward_cell_params = feedforward.get_cell_params()
    n_ff_cell_types = len(feedforward_cell_params)
    recurrent_synapse_params = recurrent.get_synapse_params()
    feedforward_synapse_params = feedforward.get_synapse_params()
    n_ff_synapse_types = len(feedforward_synapse_params)

    # Apply hidden-unit masking
    print(
        f"\nApplying hidden-unit masking: hiding {hidden_unit_fraction * 100:.1f}% of {n_neurons} neurons..."
    )
    (
        vis_weights,
        vis_ff_weights,
        vis_rec_mask,
        vis_ff_mask,
        vis_cell_type_indices,
        visible_indices,
        hidden_indices,
    ) = mask_hidden_units(
        weights,
        feedforward_weights,
        recurrent_mask,
        feedforward_mask,
        cell_type_indices,
        hidden_unit_fraction,
        hidden_units_rng,
    )

    n_visible = len(visible_indices)
    n_hidden = len(hidden_indices)
    print(f"  Visible neurons: {n_visible},  Hidden neurons: {n_hidden}")

    # Concatenate weights: [FF, visible_rec] → visible_rec output
    concatenated_weights = np.concatenate([vis_ff_weights, vis_weights], axis=0)
    concatenated_mask = np.concatenate([vis_ff_mask, vis_rec_mask], axis=0)
    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, vis_cell_type_indices + n_ff_cell_types]
    )

    # Base scaling factors
    sf_feedforward = np.array(scaling_factors["feedforward"])
    sf_recurrent = np.array(scaling_factors["recurrent"])
    concatenated_scaling_factors = np.concatenate(
        [sf_feedforward, sf_recurrent], axis=0
    )

    # Generate target scaling factors (using same seed as training would)
    sigma = np.sqrt(weight_perturbation_variance)
    mu = -(sigma**2) / 2.0
    target_scaling_factors_FF = np.random.lognormal(
        mean=mu, sigma=sigma, size=concatenated_scaling_factors.shape
    )

    # Perturbation is reciprocal of target
    perturbation_factors = 1.0 / target_scaling_factors_FF

    # Apply perturbation to concatenated weights
    n_total_inputs = concatenated_weights.shape[0]
    perturbed_weights = concatenated_weights.copy()
    for input_idx in range(n_total_inputs):
        input_type = concatenated_cell_type_indices[input_idx]
        for output_idx in range(n_visible):
            output_type = vis_cell_type_indices[output_idx]
            perturbed_weights[input_idx, output_idx] *= perturbation_factors[
                input_type, output_type
            ]

    # Initialize scaling factors AT TARGET (so normalized = 1)
    scaling_factors_init = target_scaling_factors_FF.copy()

    # Save targets
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = output_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        targets_dir / "target_scaling_factors.npz",
        feedforward_scaling_factors=target_scaling_factors_FF,
    )

    # Build combined cell/synapse params
    combined_cell_params_FF = feedforward_cell_params.copy()
    for cp in recurrent_cell_params:
        ocp = cp.copy()
        ocp["cell_id"] = cp["cell_id"] + n_ff_cell_types
        combined_cell_params_FF.append(ocp)

    combined_synapse_params_FF = feedforward_synapse_params.copy()
    for sp in recurrent_synapse_params:
        osp = sp.copy()
        osp["cell_id"] = sp["cell_id"] + n_ff_cell_types
        osp["synapse_id"] = sp["synapse_id"] + n_ff_synapse_types
        combined_synapse_params_FF.append(osp)

    # Load dataset
    spike_dataset = ExactFFDataset(
        spike_data_path=input_dir / "spike_data.zarr",
        chunk_size=chunk_size,
        device=device,
    )
    batch_size = spike_dataset.batch_size
    dt = spike_dataset.dt

    visible_indices_tensor = torch.from_numpy(visible_indices).long()

    collate_fn = functools.partial(
        hidden_units_collate_fn, visible_indices_tensor=visible_indices_tensor
    )

    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=None,
        sampler=CyclicSampler(spike_dataset),
        **DATALOADER_KWARGS,
        collate_fn=collate_fn,
    )

    # Build per-pair frozen projections with mask applied and scaling factors
    # baked in. Source rows mix FF and visible-recurrent cell types in a
    # combined namespace; targets are visible-recurrent only.
    masked_weights = perturbed_weights * concatenated_mask.astype(np.float32)
    combined_src_names = [c["name"] for c in combined_cell_params_FF]
    rec_tgt_names = [c["name"] for c in recurrent_cell_params]
    projections: dict[tuple[str, str], FrozenProjection] = {}
    for src_id, src_name in enumerate(combined_src_names):
        src_rows = np.flatnonzero(concatenated_cell_type_indices == src_id)
        for tgt_id, tgt_name in enumerate(rec_tgt_names):
            tgt_cols = np.flatnonzero(vis_cell_type_indices == tgt_id)
            block = masked_weights[np.ix_(src_rows, tgt_cols)].astype(np.float32)
            block = block * float(scaling_factors_init[src_id, tgt_id])
            projections[(src_name, tgt_name)] = FrozenProjection(block)

    # Initialize model with scaling factors AT TARGET
    model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=projections,
        cell_type_indices=vis_cell_type_indices,
        cell_type_indices_FF=concatenated_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=hyperparameters.surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )
    model.to(device)
    model.eval()
    model.reset_state(batch_size=batch_size)

    van_rossum_loss_fn = VanRossumLoss(
        tau_rise=hyperparameters.van_rossum_tau_rise,
        tau_decay=hyperparameters.van_rossum_tau_decay,
        dt=dt,
        window_size=chunk_size,
        device=device,
    )

    # Run inference: burn-in + analysis
    total_chunks = 2 * plot_size
    print(
        f"\nRunning inference for {total_chunks} chunks ({plot_size} burn-in + {plot_size} analysis)..."
    )

    data_iter = iter(spike_dataloader)

    losses = []
    student_spike_counts = np.zeros(n_visible)
    teacher_spike_counts = np.zeros(n_visible)
    plot_output_spikes = []
    plot_target_spikes = []

    with torch.no_grad():
        for chunk_idx in tqdm(range(total_chunks), desc="Inference"):
            batch = next(data_iter)
            input_spikes = batch.input_spikes.to(device)
            target_spikes = batch.target_spikes.to(device)

            output = model(input_spikes)
            spikes = output["spikes"] if isinstance(output, dict) else output

            if chunk_idx >= plot_size:
                loss = van_rossum_loss_fn(spikes, target_spikes)
                losses.append(loss.item())

                student_spike_counts += spikes[0].sum(dim=0).cpu().numpy()
                teacher_spike_counts += target_spikes[0].sum(dim=0).cpu().numpy()

                plot_output_spikes.append(spikes[0].cpu().numpy())
                plot_target_spikes.append(target_spikes[0].cpu().numpy())

    mean_loss = np.mean(losses)
    duration_s = plot_size * chunk_size * dt / 1000.0
    student_fr = student_spike_counts / duration_s
    teacher_fr = teacher_spike_counts / duration_s

    output_spikes = np.concatenate(plot_output_spikes, axis=0)
    target_spikes_np = np.concatenate(plot_target_spikes, axis=0)

    exc_mask = vis_cell_type_indices == 0
    inh_mask = vis_cell_type_indices == 1

    metrics = {
        "loss": mean_loss,
        "hidden_unit_fraction": hidden_unit_fraction,
        "n_visible": n_visible,
        "n_hidden": n_hidden,
        "firing_rate/student_mean": float(student_fr.mean()),
        "firing_rate/student_std": float(student_fr.std()),
        "firing_rate/teacher_mean": float(teacher_fr.mean()),
        "firing_rate/teacher_std": float(teacher_fr.std()),
        "firing_rate/student_exc_mean": float(student_fr[exc_mask].mean()),
        "firing_rate/student_inh_mean": float(student_fr[inh_mask].mean()),
        "firing_rate/teacher_exc_mean": float(teacher_fr[exc_mask].mean()),
        "firing_rate/teacher_inh_mean": float(teacher_fr[inh_mask].mean()),
    }

    header = f"METRICS (SF at target, hidden_fraction={hidden_unit_fraction:.2f})"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    print("=" * len(header))

    # Plot
    n_plot = min(10, output_spikes.shape[1])
    interleaved = np.zeros((1, output_spikes.shape[0], 2 * n_plot))
    for i in range(n_plot):
        interleaved[0, :, 2 * i] = target_spikes_np[:, i]
        interleaved[0, :, 2 * i + 1] = output_spikes[:, i]

    cell_type_indices_plot = np.array([0, 1] * n_plot)
    cell_type_names_plot = ["Target", "Student"]

    fig = plot_spike_trains(
        spikes=interleaved,
        dt=dt,
        cell_type_indices=cell_type_indices_plot,
        cell_type_names=cell_type_names_plot,
        n_neurons_plot=2 * n_plot,
        fraction=1.0,
        random_seed=None,
        title=f"Hidden units {hidden_unit_fraction:.0%}: SF at target (loss={mean_loss:.4f})",
        ylabel="Neuron",
        figsize=(14, 8),
    )

    fig.savefig(output_dir / "spike_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "training_metrics.csv", index=False)

    np.savez(
        output_dir / "plot_data.npz",
        student_spikes=output_spikes,
        teacher_spikes=target_spikes_np,
        dt=dt,
        cell_type_indices=vis_cell_type_indices,
        visible_indices=visible_indices,
        hidden_indices=hidden_indices,
        n_neurons=n_visible,
    )

    print(f"\nSaved to {output_dir}")
    print(f"✓ Spike comparison: {output_dir / 'spike_comparison.png'}")
    print(f"✓ Metrics CSV:      {output_dir / 'training_metrics.csv'}")
    print(f"✓ Plot data:        {output_dir / 'plot_data.npz'}")
