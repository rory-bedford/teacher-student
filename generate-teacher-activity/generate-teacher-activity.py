"""
Generating teacher activity from a synthetic connectome

This script generates a fresh connectome from parameters and then generates
teacher spike train patterns using odour-modulated inputs.
These patterns serve as targets for student network training.
"""

import numpy as np
import torch
import toml
import zarr
import matplotlib.pyplot as plt
from synthetic_connectome import topology_generators, weight_assigners, cell_types
from dataloaders.unsupervised import (
    InhomogeneousPoissonSpikeDataLoader,
)
from dataloaders.rate_processes import OrnsteinUhlenbeckRateProcess
from dataloaders.odourants import (
    generate_odour_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from network_simulators.projections import make_frozen_projections
from snn_runners import SNNInference
from configs import SimulationConfig
from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig
from configs.odours import OdourInputConfig
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
    create_assembly_activity_dashboard,
)


def main(input_dir, output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
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
    simulation = SimulationConfig(**data["simulation"])
    recurrent = RecurrentLayerConfig(**data["recurrent"])
    feedforward = FeedforwardLayerConfig(**data["feedforward"])

    # Parse odour configs
    odours_data = data["odours"]
    tau = odours_data.pop("tau")
    temperature = odours_data.pop("temperature")
    sigma = odours_data.pop("sigma")
    odours = {name: OdourInputConfig(**config) for name, config in odours_data.items()}

    # Extract parameters into plain Python variables
    dt = simulation.dt
    chunk_size = simulation.chunk_size
    num_chunks = simulation.num_chunks
    batch_size = simulation.batch_size
    seed = simulation.seed
    plot_size = simulation.plot_size

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ===================================
    # Generate Recurrent Connectivity
    # ===================================

    print(
        f"Generating {recurrent.topology.num_neurons} neuron recurrent network with {recurrent.topology.num_assemblies} assemblies..."
    )

    # Assign cell types to recurrent layer
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=recurrent.topology.num_neurons,
        cell_type_proportions=recurrent.cell_types.proportion,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph, assembly_ids = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,
        num_assemblies=recurrent.topology.num_assemblies,
        conn_within=recurrent.topology.conn_within,
        conn_between=recurrent.topology.conn_between,
        method="configuration",
    )

    # Assign log-normal weights to connectivity graph
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,
        w_mu_matrix=recurrent.weights.w_mu,
        w_sigma_matrix=recurrent.weights.w_sigma,
        parameter_space="linear",
    )

    # ==========================================
    # Generate Feedforward Connections and Weights
    # ==========================================

    print(f"Generating {feedforward.topology.num_neurons} feedforward inputs...")

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=feedforward.topology.num_neurons,
        cell_type_proportions=feedforward.cell_types.proportion,
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,
        conn_matrix=feedforward.topology.conn_inputs,
        allow_self_loops=True,
        method="configuration",
    )

    # Assign log-normal weights to feedforward connectivity
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,
        w_mu_matrix=feedforward.weights.w_mu,
        w_sigma_matrix=feedforward.weights.w_sigma,
        parameter_space="linear",
    )

    # Derive connectivity masks from boolean connectivity graphs
    connectome_mask = connectivity_graph
    feedforward_mask = feedforward_connectivity_graph

    print(f"✓ Generated recurrent weights: {weights.shape}")
    print(f"✓ Generated feedforward weights: {feedforward_weights.shape}")
    print(
        f"✓ Assembly IDs: {len(assembly_ids)} neurons in {len(np.unique(assembly_ids[assembly_ids >= 0]))} assemblies"
    )

    # =========================
    # Create Feedforward Inputs
    # =========================

    # Calculate number of odour patterns from assemblies
    num_odours = len(np.unique(assembly_ids[assembly_ids >= 0]))

    # Generate odour-modulated firing rate patterns (one per assembly)
    # Target excitatory cells (index 0) when computing connection strengths
    input_firing_rates_odour = generate_odour_firing_rates(
        feedforward_weights=feedforward_weights,
        input_source_indices=input_source_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=feedforward.cell_types.names,
        odour_configs={name: cfg.to_dict() for name, cfg in odours.items()},
    )

    print(f"✓ Generated {num_odours} odour patterns")
    print(f"  Total patterns: {input_firing_rates_odour.shape[0]}")

    batch_size = batch_size

    # Create Ornstein-Uhlenbeck rate process in pattern space
    # This modulates the odour patterns dynamically over time
    rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_firing_rates_odour,  # Shape: (n_patterns, n_input_neurons)
        chunk_size=int(chunk_size),
        dt=dt,
        tau=tau,  # Time constant for mean reversion (ms)
        temperature=temperature,  # Softmax temperature for pattern mixing
        sigma=sigma,  # Noise amplitude
        a_init=None,  # Initialize to ones
        return_rates=True,  # Return rates for storage and diagnostics
        batch_size=batch_size,  # Each trial gets its own independent OU trajectory
        device=device,  # Run OU process on GPU
    )

    # Create inhomogeneous Poisson spike dataloader with OU-driven rates
    spike_dataloader = InhomogeneousPoissonSpikeDataLoader(
        rate_process=rate_process,
        batch_size=batch_size,
        device=device,
        return_rates=True,  # Enable rate output
    )

    print("✓ Created inhomogeneous Poisson spike dataloader with OU rate process")

    # ======================
    # Initialize LIF Network
    # ======================

    rec_projections, ff_projections = make_frozen_projections(
        rec_weights=weights,
        ff_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=input_source_indices,
        cell_type_names=recurrent.cell_types.names,
        ff_cell_type_names=feedforward.cell_types.names,
    )

    model = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projections,
        ff_projections=ff_projections,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=input_source_indices,
        cell_params=recurrent.get_cell_params(),
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        surrgrad_scale=1.0,
        batch_size=batch_size,
        track_variables=False,
        track_batch_idx=0,
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Compile the model for faster execution
    try:
        model = torch.compile(model)
        print("✓ Model compiled with torch.compile")
    except Exception as e:
        print(f"⚠ torch.compile failed ({e}), running in eager mode")

    # ======================================
    # Save Static Data and Run Inference
    # ======================================

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = results_dir / "spike_data.zarr"

    # ==============================
    # Run Inference with SNNInference
    # ==============================

    print("\n" + "=" * len("STARTING NETWORK INFERENCE"))
    print("STARTING NETWORK INFERENCE")
    print("=" * len("STARTING NETWORK INFERENCE"))

    total_duration_s = num_chunks * chunk_size * dt / 1000.0
    print(f"Running inference with {batch_size} independent OU-modulated trajectories")
    print(f"Processing {num_chunks} chunks...")
    print(f"Total simulation duration: {total_duration_s:.2f} s")

    # Run inference and save to zarr
    # Note: We're NOT saving tracked variables yet (too large), we'll do a separate run for visualization
    inference_runner = SNNInference(
        model=model,
        dataloader=spike_dataloader,
        device=device,
        output_mode="zarr",
        zarr_path=zarr_path,
        save_tracked_variables=False,  # Don't save tracked vars (too large for full simulation)
        max_chunks=num_chunks,
        progress_bar=True,
    )

    _ = inference_runner.run()  # Run inference and save to zarr

    print("\n✓ Network inference completed!")
    print(f"✓ Data saved to {zarr_path}")

    # Save static odourant patterns (base patterns before OU modulation).
    # Must be done AFTER inference_runner.run() since SNNInference opens zarr
    # with mode="w", which would overwrite anything saved beforehand.
    root = zarr.open_group(zarr_path, mode="a")
    root.create_dataset(
        "odourant_patterns",
        shape=input_firing_rates_odour.shape,
        dtype=input_firing_rates_odour.dtype,
        data=input_firing_rates_odour,
    )
    print("✓ Saved odourant patterns to zarr")
    print(f"  - odourant_patterns: {input_firing_rates_odour.shape}")

    # Save network structure for downstream experiments
    np.savez_compressed(
        results_dir / "network_structure.npz",
        recurrent_weights=weights.astype(np.float32),
        feedforward_weights=feedforward_weights.astype(np.float32),
        cell_type_indices=cell_type_indices,
        feedforward_cell_type_indices=input_source_indices,
        assembly_ids=assembly_ids,
        recurrent_connectivity=connectivity_graph,
        feedforward_connectivity=feedforward_connectivity_graph,
    )
    print(f"✓ Saved network structure to {results_dir / 'network_structure.npz'}")

    # Reopen zarr to inspect what was saved
    root = zarr.open_group(zarr_path, mode="r")
    for key in root.keys():
        print(f"  - {key}: {root[key].shape}")

    # =========================================================
    # Run Small Inference for Visualization (with tracking)
    # =========================================================

    print("\n" + "=" * len("GENERATING VISUALIZATION DATA"))
    print("GENERATING VISUALIZATION DATA")
    print("=" * len("GENERATING VISUALIZATION DATA"))
    print(f"Running {plot_size} chunks with variable tracking for visualization...")

    # Reset model state for clean inference
    model.reset_state(batch_size=batch_size)
    model.track_variables = True  # Enable tracking for visualization

    # Create a new dataloader for visualization chunks
    viz_rate_process = OrnsteinUhlenbeckRateProcess(
        patterns=input_firing_rates_odour,
        chunk_size=int(chunk_size),
        dt=dt,
        tau=tau,
        temperature=temperature,
        sigma=sigma,
        a_init=None,
        return_rates=True,
        batch_size=batch_size,
        device=device,
    )

    viz_dataloader = InhomogeneousPoissonSpikeDataLoader(
        rate_process=viz_rate_process,
        batch_size=batch_size,
        device=device,
        return_rates=True,
    )

    # Run inference in memory mode to get tracked variables
    viz_inference_runner = SNNInference(
        model=model,
        dataloader=viz_dataloader,
        device=device,
        output_mode="memory",
        save_tracked_variables=True,  # Save tracked variables for visualization
        max_chunks=plot_size,
        progress_bar=False,
    )

    viz_result = viz_inference_runner.run()

    # Extract visualization data (first batch only)
    viz_output_spikes = viz_result["output_spikes"][0:1, ...]  # (1, time, neurons)
    viz_input_spikes = viz_result["input_spikes"][0:1, ...]
    viz_ou_weights = viz_result["weights"][0:1, ...]
    viz_voltages = viz_result["voltages"][0:1, ...]
    viz_currents_leak = viz_result["currents_leak"][0:1, ...]

    # Split unified currents/conductances into recurrent vs feedforward
    rec_unified_ids = sorted(set(model.rec_synapse_to_unified.values()))
    ff_unified_ids = sorted(set(model.ff_synapse_to_unified.values()))

    viz_currents = viz_result["currents"][0:1, ..., rec_unified_ids]
    viz_currents_FF = viz_result["currents"][0:1, ..., ff_unified_ids]
    viz_conductances = viz_result["conductances"][0:1, ..., rec_unified_ids]
    viz_conductances_FF = viz_result["conductances"][0:1, ..., ff_unified_ids]

    print(f"✓ Visualization data generated ({plot_size} chunks)")

    # Free GPU memory
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("\n✓ Freed GPU memory")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("INFERENCE COMPLETE!"))
    print("INFERENCE COMPLETE!")
    print("=" * len("INFERENCE COMPLETE!"))
    print(f"✓ Spike data saved to {zarr_path}")
    print(f"✓ Data can be loaded with: zarr.open('{zarr_path}', mode='r')")
    print("=" * len("INFERENCE COMPLETE!"))

    # =====================================
    # Generate Dashboards for Visualization
    # =====================================

    print("\n" + "=" * len("GENERATING DASHBOARDS"))
    print("GENERATING DASHBOARDS")
    print("=" * len("GENERATING DASHBOARDS"))

    # Data is already in numpy format from inference runner (batch=1)
    # Convert to appropriate dtypes for visualization
    viz_voltages = viz_voltages.astype(np.float32)
    viz_currents = viz_currents.astype(np.float32)
    viz_currents_FF = viz_currents_FF.astype(np.float32)
    viz_currents_leak = viz_currents_leak.astype(np.float32)
    viz_conductances = viz_conductances.astype(np.float32)
    viz_conductances_FF = viz_conductances_FF.astype(np.float32)
    viz_output_spikes = viz_output_spikes.astype(np.int32)
    viz_input_spikes = viz_input_spikes.astype(np.int32)
    viz_ou_weights = viz_ou_weights.astype(np.float32)

    # Generate connectivity dashboard
    print("Generating connectivity dashboard...")
    connectivity_fig = create_connectivity_dashboard(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=recurrent.cell_types.names,
        input_cell_type_names=feedforward.cell_types.names,
        connectome_mask=connectome_mask,
        feedforward_mask=feedforward_mask,
    )

    # Generate activity dashboard
    print("Generating activity dashboard...")
    activity_fig = create_activity_dashboard(
        output_spikes=viz_output_spikes,
        input_spikes=viz_input_spikes,
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        dt=dt,
        voltages=viz_voltages,
        neuron_types=cell_type_indices,
        neuron_params=recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=viz_currents,
        feedforward_currents=viz_currents_FF,
        leak_currents=viz_currents_leak,
        recurrent_conductances=viz_conductances,
        feedforward_conductances=viz_conductances_FF,
        input_cell_type_names=feedforward.cell_types.names,
        recurrent_synapse_names=recurrent.get_synapse_names(),
        feedforward_synapse_names=feedforward.get_synapse_names(),
        window_size=50.0,
        n_neurons_plot=20,
        fraction=1.0,
        random_seed=42,
        assembly_ids=assembly_ids,
    )

    # Generate assembly activity dashboard
    print("Generating assembly activity dashboard...")
    assembly_fig = create_assembly_activity_dashboard(
        output_spikes=viz_output_spikes,
        ou_process_weights=viz_ou_weights,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        dt=dt,
        excitatory_idx=0,
    )

    # Save dashboards
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    connectivity_fig.savefig(
        figures_dir / "connectivity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(connectivity_fig)

    activity_fig.savefig(
        figures_dir / "activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(activity_fig)

    assembly_fig.savefig(
        figures_dir / "assembly_activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(assembly_fig)

    print(f"✓ Saved dashboard plots to {figures_dir}")
    print("=" * len("GENERATING DASHBOARDS"))
