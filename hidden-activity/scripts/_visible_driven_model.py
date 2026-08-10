"""
Shared construction of the visible-driven TwoLayerSNN.

Mirrors the model that ``train_visible_driven.py`` builds, so inference-mode
analyses (e.g. ``bias-check``) evaluate exactly the network that was trained
rather than a re-derived approximation of it.

Architecture:
  Layer 1 (hidden):  [FF spikes, teacher visible spikes] -> hidden spikes,
                     with hidden->hidden recurrence.
  Layer 2 (visible): [FF spikes, hidden spikes, teacher visible spikes] ->
                     visible spikes, no recurrence.

Scaling factors are indexed by the combined input cell-type space
``[feedforward types, recurrent types]`` (rows) x ``[recurrent types]``
(columns) — the same ``(n_ff_ct + n_rec_ct, n_rec_ct)`` layout saved in
``targets/target_scaling_factors.npz``.
"""

import numpy as np
import torch
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


def perturb_weights(
    feedforward_weights,
    recurrent_weights,
    feedforward_mask,
    recurrent_mask,
    cell_type_indices,
    feedforward_cell_type_indices,
    target_scaling_factors,
    n_ff_cell_types,
):
    """Apply the per-cell-type-pair perturbation and the connectivity masks.

    The perturbation is the reciprocal of the target scaling factors, so a
    student running at the target scaling factors exactly recovers the teacher.

    Returns:
        (perturbed_ff_weights, perturbed_rec_weights), both mask-applied.
    """
    concatenated_cell_type_indices = np.concatenate(
        [feedforward_cell_type_indices, cell_type_indices + n_ff_cell_types]
    )
    concatenated_weights = np.concatenate(
        [feedforward_weights, recurrent_weights], axis=0
    )

    perturbation = 1.0 / target_scaling_factors
    perturbed = concatenated_weights * perturbation[
        concatenated_cell_type_indices[:, None], cell_type_indices[None, :]
    ].astype(np.float32)

    n_feedforward = feedforward_weights.shape[0]
    perturbed_ff = perturbed[:n_feedforward, :] * feedforward_mask.astype(np.float32)
    perturbed_rec = perturbed[n_feedforward:, :] * recurrent_mask.astype(np.float32)
    return perturbed_ff, perturbed_rec


def build_visible_driven_model(
    *,
    dt,
    batch_size,
    perturbed_ff_weights,
    perturbed_rec_weights,
    cell_type_indices,
    feedforward_cell_type_indices,
    visible_indices,
    hidden_indices,
    recurrent_cell_params,
    feedforward_cell_params,
    recurrent_synapse_params,
    combined_cell_params_FF,
    combined_synapse_params_FF,
    scaling_factors,
    surrgrad_scale,
    share_ff_scaling=True,
    track_variables=False,
):
    """Build the two-layer visible-driven model at fixed scaling factors.

    Args:
        scaling_factors: ``(n_ff_ct + n_rec_ct, n_rec_ct)`` matrix written into
            every projection's ``log_sf``.
        share_ff_scaling: Share layer 1's feedforward scaling-factor parameters
            with the matching layer 2 pairs, as training does.

    Returns:
        (model, layer1_ff_projections, layer1_rec_projections) — the projection
        dicts are returned so callers can read the scaling factors back out.
    """
    n_feedforward = perturbed_ff_weights.shape[0]
    n_ff_cell_types = len(feedforward_cell_params)

    rec_cell_type_names = [cp["name"] for cp in recurrent_cell_params]
    ff_cell_type_names = [cp["name"] for cp in feedforward_cell_params]
    combined_input_cell_type_names = [cp["name"] for cp in combined_cell_params_FF]

    sf_recurrent = scaling_factors[n_ff_cell_types:]

    # --- Layer 1 (hidden): [FF, visible] -> hidden, hidden -> hidden ---
    layer1_ff_weights = np.concatenate(
        [
            perturbed_ff_weights[:, hidden_indices],
            perturbed_rec_weights[visible_indices][:, hidden_indices],
        ],
        axis=0,
    )
    layer1_rec_weights = perturbed_rec_weights[hidden_indices][:, hidden_indices]
    layer1_cell_type_indices = cell_type_indices[hidden_indices]
    layer1_ff_cell_type_indices = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    layer1_rec_projections, layer1_ff_projections = make_scaling_factor_projections(
        rec_weights=layer1_rec_weights,
        ff_weights=layer1_ff_weights,
        cell_type_indices=layer1_cell_type_indices,
        ff_cell_type_indices=layer1_ff_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=combined_input_cell_type_names,
        init_sf=1.0,
    )

    with torch.no_grad():
        for (src_name, tgt_name), proj in layer1_rec_projections.items():
            src_id = rec_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.fill_(float(np.log(sf_recurrent[src_id, tgt_id])))
        for (src_name, tgt_name), proj in layer1_ff_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.fill_(float(np.log(scaling_factors[src_id, tgt_id])))

    # --- Layer 2 (visible): [FF, hidden, visible] -> visible, no recurrence ---
    layer2_ff_weights = np.concatenate(
        [
            perturbed_ff_weights[:, visible_indices],
            perturbed_rec_weights[hidden_indices][:, visible_indices],
            perturbed_rec_weights[visible_indices][:, visible_indices],
        ],
        axis=0,
    )
    layer2_cell_type_indices = cell_type_indices[visible_indices]
    layer2_ff_cell_type_indices = np.concatenate(
        [
            feedforward_cell_type_indices,
            cell_type_indices[hidden_indices] + n_ff_cell_types,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )
    # Recurrent input rows of layer 2 span hidden followed by visible neurons,
    # so their cell types are not the layer's output cell types.
    layer2_rec_source_cell_type_indices = (
        layer2_ff_cell_type_indices[n_feedforward:] - n_ff_cell_types
    )

    layer2_projections = make_chunked_ff_projections(
        rec_weights=layer2_ff_weights[n_feedforward:, :],
        ff_weights=layer2_ff_weights[:n_feedforward, :],
        cell_type_indices=layer2_cell_type_indices,
        ff_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=rec_cell_type_names,
        ff_cell_type_names=ff_cell_type_names,
        init_sf=1.0,
        rec_source_cell_type_indices=layer2_rec_source_cell_type_indices,
    )

    with torch.no_grad():
        for (src_name, tgt_name), proj in layer2_projections.items():
            src_id = combined_input_cell_type_names.index(src_name)
            tgt_id = rec_cell_type_names.index(tgt_name)
            proj.log_sf.fill_(float(np.log(scaling_factors[src_id, tgt_id])))

    if share_ff_scaling:
        # Tie the scaling-factor *parameter* only. The two layers' projections
        # hold different connectome blocks for the same (src, tgt) pair —
        # layer 1's targets are the hidden neurons, layer 2's the visible ones —
        # so sharing the whole Projection object would give layer 2 layer 1's
        # weights (and mismatched shapes whenever the two subsets differ).
        for key, proj in layer1_ff_projections.items():
            if key in layer2_projections:
                layer2_projections[key].log_sf = proj.log_sf

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
        track_variables=track_variables,
    )
    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=layer2_projections,
        cell_type_indices=layer2_cell_type_indices,
        cell_type_indices_FF=layer2_ff_cell_type_indices,
        cell_params=recurrent_cell_params,
        cell_params_FF=combined_cell_params_FF,
        synapse_params_FF=combined_synapse_params_FF,
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=track_variables,
    )

    model = TwoLayerSNN(layer1, layer2, n_ff=n_feedforward)
    return model, layer1_ff_projections, layer1_rec_projections


def read_scaling_factors(
    layer1_ff_projections,
    layer1_rec_projections,
    combined_input_cell_type_names,
    rec_cell_type_names,
    n_ff_cell_types,
):
    """Read the ``(n_ff_ct + n_rec_ct, n_rec_ct)`` scaling-factor matrix back out."""
    sf = np.ones(
        (len(combined_input_cell_type_names), len(rec_cell_type_names)),
        dtype=np.float32,
    )
    for (src_name, tgt_name), proj in layer1_ff_projections.items():
        src_id = combined_input_cell_type_names.index(src_name)
        tgt_id = rec_cell_type_names.index(tgt_name)
        sf[src_id, tgt_id] = float(np.exp(proj.log_sf.detach().cpu().numpy()))
    for (src_name, tgt_name), proj in layer1_rec_projections.items():
        src_id = rec_cell_type_names.index(src_name) + n_ff_cell_types
        tgt_id = rec_cell_type_names.index(tgt_name)
        sf[src_id, tgt_id] = float(np.exp(proj.log_sf.detach().cpu().numpy()))
    return sf
