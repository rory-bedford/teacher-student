"""The teacher-forced student, built from a saved student structure.

Architecture (the archived visible-driven / ff-learnt two-layer model):

    Layer 1  unobserved neurons — genuinely recurrent among themselves. Input rows:
             [feedforward units, observed neurons, unreconstructed recurrent units]
    Layer 2  observed neurons — no recurrence among themselves. Input rows:
             [feedforward units, layer-1 spikes, observed neurons, unreconstructed units]

Observed neurons' and all injected sources' spikes are the teacher's (teacher forcing).
The two layers are one network: every weight comes from the same student matrices, and
each trainable parameter is shared between every block it appears in, so a scaling
factor is one parameter per (source type, target type) pair for the whole network.

A block from source type ``a`` to target type ``b`` is parametrised as

    connectome               exp(log_sf[a, b]) * student_weights * perturbation[a, b]
    learnt recurrence (Fig 2) exp(log_weights[a, b])   — full rank, all pairs, no connectome
    unreconstructed (Fig 6)  exp(U[a, b] @ V[a, b])    — low rank, from sources outside S
"""

import numpy as np
import torch
from connectome_snns.configs.conductance_based import (
    FeedforwardLayerConfig,
    RecurrentLayerConfig,
)
from connectome_snns.dataloaders.supervised import SpikeData
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    FullRankProjection,
    LowRankProjection,
    Projection,
    ScalingFactorProjection,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN
from torch import nn

# =====================================================================
# Physiology
# =====================================================================


def physiology(params):
    """Cell/synapse parameter lists in the namespaces the simulators expect."""
    recurrent = RecurrentLayerConfig(**params["recurrent"])
    feedforward = FeedforwardLayerConfig(**params["feedforward"])
    rec_cells = recurrent.get_cell_params()
    ff_cells = feedforward.get_cell_params()
    rec_synapses = recurrent.get_synapse_params()
    ff_synapses = feedforward.get_synapse_params()

    combined_cells = ff_cells.copy()
    for cell in rec_cells:
        shifted = cell.copy()
        shifted["cell_id"] = cell["cell_id"] + len(ff_cells)
        combined_cells.append(shifted)
    combined_synapses = ff_synapses.copy()
    for synapse in rec_synapses:
        shifted = synapse.copy()
        shifted["cell_id"] = synapse["cell_id"] + len(ff_cells)
        shifted["synapse_id"] = synapse["synapse_id"] + len(ff_synapses)
        combined_synapses.append(shifted)

    return {
        "rec_cells": rec_cells,
        "rec_synapses": rec_synapses,
        "combined_cells": combined_cells,
        "combined_synapses": combined_synapses,
        "rec_names": [c["name"] for c in rec_cells],
        "combined_names": [c["name"] for c in combined_cells],
        "n_ff_types": len(ff_cells),
    }


# =====================================================================
# Projections (experiment-local)
# =====================================================================
#
# Shared parameters are held by ``StudentParameters`` and referenced from the
# projections with ``object.__setattr__`` so ``nn.Module`` does not register them a
# second time on every projection (the archived SharedSlicedLowRankProjection
# pattern). The classes subclass the library's FullRank/LowRank projections so the
# simulator's mask extraction recognises them.


class SlicedFullRankProjection(FullRankProjection):
    """Rows/columns of a shared full-rank log-weight matrix (learnt recurrence)."""

    def __init__(self, log_weights, rows, cols):
        Projection.__init__(self, len(rows), len(cols))
        object.__setattr__(self, "shared_log_weights", log_weights)
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer("mask", torch.ones(len(rows), len(cols)))

    # Called once per chunk, not per timestep; compiling it per block shape only
    # exhausts torch.compile's recompile limit.
    @torch.compiler.disable
    def forward(self):
        return torch.exp(self.shared_log_weights[self.rows][:, self.cols])


class MixedProjection(LowRankProjection):
    """Known rows through the connectome, unreconstructed rows through learnt weights.

    Known rows:  exp(log_sf) * connectome   (connectome already perturbation-scaled)
    Learnt rows: exp(U[u_rows] @ V[:, v_cols]), fully connected
    """

    def __init__(
        self, connectome, known_rows, learnt_rows, log_sf, U, V, u_rows, v_cols
    ):
        n_source = len(known_rows) + len(learnt_rows)
        n_target = connectome.shape[1]
        Projection.__init__(self, n_source, n_target)
        object.__setattr__(self, "shared_log_sf", log_sf)
        object.__setattr__(self, "shared_U", U)
        object.__setattr__(self, "shared_V", V)
        self.register_buffer(
            "connectome", torch.as_tensor(connectome, dtype=torch.float32)
        )
        self.register_buffer(
            "known_rows", torch.as_tensor(known_rows, dtype=torch.long)
        )
        self.register_buffer(
            "learnt_rows", torch.as_tensor(learnt_rows, dtype=torch.long)
        )
        self.register_buffer("u_rows", torch.as_tensor(u_rows, dtype=torch.long))
        self.register_buffer("v_cols", torch.as_tensor(v_cols, dtype=torch.long))
        mask = torch.ones(n_source, n_target)
        if len(known_rows):
            mask[self.known_rows] = (self.connectome != 0).float()
        self.register_buffer("mask", mask)

    # Called once per chunk, not per timestep; compiling it per block shape only
    # exhausts torch.compile's recompile limit.
    @torch.compiler.disable
    def forward(self):
        out = torch.zeros(
            self.n_source, self.n_target, device=self.mask.device, dtype=torch.float32
        )
        if self.known_rows.numel():
            out = out.index_put(
                (self.known_rows,), torch.exp(self.shared_log_sf) * self.connectome
            )
        if self.learnt_rows.numel():
            learnt = torch.exp(
                self.shared_U[self.u_rows] @ self.shared_V[:, self.v_cols]
            )
            out = out.index_put((self.learnt_rows,), learnt)
        return out


class StudentParameters(nn.Module):
    """Every trainable parameter of the student, registered exactly once."""

    def __init__(self):
        super().__init__()
        self.log_sf = nn.ParameterDict()
        self.log_weights = nn.ParameterDict()
        self.U = nn.ParameterDict()
        self.V = nn.ParameterDict()

    def scaling_factor_parameters(self):
        return list(self.log_sf.values())

    def weight_parameters(self):
        return [*self.log_weights.values(), *self.U.values(), *self.V.values()]


class ObservedOnlySNN(nn.Module):
    """Layer 2 alone, for a student with every modelled neuron observed."""

    def __init__(self, layer2):
        super().__init__()
        self.layer2 = layer2

    def forward(self, input_spikes):
        out = self.layer2(input_spikes)
        # Layer 2 returns a dict of tracked variables when the trainer turns tracking on.
        result = dict(out) if isinstance(out, dict) else {"spikes": out}
        spikes = result["spikes"]
        result["hidden_spikes"] = spikes.new_zeros(spikes.shape[0], spikes.shape[1], 0)
        return result

    def reset_state(self, batch_size=None):
        self.layer2.reset_state(batch_size)

    def get_checkpoint_state(self):
        return {f"layer2_{k}": v for k, v in self.layer2.get_checkpoint_state().items()}

    def load_checkpoint_state(self, state):
        self.layer2.load_checkpoint_state(
            {k.removeprefix("layer2_"): v for k, v in state.items()}
        )

    @property
    def track_variables(self):
        return self.layer2.track_variables

    @track_variables.setter
    def track_variables(self, value):
        self.layer2.track_variables = value

    @property
    def device(self):
        return self.layer2.device

    @property
    def batch_size(self):
        # The trainer resets state with getattr(model, "batch_size", 1).
        return self.layer2.batch_size


# =====================================================================
# Model
# =====================================================================


def neuron_sets(structure):
    """Teacher ids of the observed, unobserved and unreconstructed recurrent units."""
    modelled = structure["modelled"]
    observed = structure["observed"]
    injected = (
        ~modelled if structure["inject_unreconstructed"] else np.zeros_like(modelled)
    )
    return {
        "observed": np.flatnonzero(observed),
        "unobserved": np.flatnonzero(modelled & ~observed),
        "unreconstructed": np.flatnonzero(injected),
    }


class StudentCollate:
    """Input ``[feedforward, observed, unreconstructed]``; target ``observed``."""

    def __init__(self, observed, unreconstructed):
        self.observed = torch.as_tensor(observed, dtype=torch.long)
        self.unreconstructed = torch.as_tensor(unreconstructed, dtype=torch.long)

    def __call__(self, batch):
        observed = batch.target_spikes[:, :, self.observed]
        injected = batch.target_spikes[:, :, self.unreconstructed]
        return SpikeData(
            input_spikes=torch.cat([batch.input_spikes, observed, injected], dim=2),
            target_spikes=observed,
        )


def build_student(structure, params, *, batch_size, dt, surrgrad_scale, low_rank=1):
    """Build the student network described by ``structure``.

    Returns:
        (model, student_parameters) — the parameters are also registered on the model.
    """
    phys = physiology(params)
    n_ff_types = phys["n_ff_types"]
    combined_names = phys["combined_names"]
    rec_names = phys["rec_names"]

    ct = structure["cell_type_indices"]
    ff_ct = structure["ff_cell_type_indices"]
    n_ff = ff_ct.size
    known_ff = structure["known_ff"]
    perturbation = structure["perturbation"]
    learnt_recurrence = structure["recurrent_model"] == "learnt"
    sets = neuron_sets(structure)
    observed, unobserved, unreconstructed = (
        sets["observed"],
        sets["unobserved"],
        sets["unreconstructed"],
    )
    modelled = np.flatnonzero(structure["modelled"])
    has_learnt_sources = bool(structure["inject_unreconstructed"])

    # One weight matrix over [feedforward units; recurrent neurons] x recurrent neurons.
    weights = np.concatenate([structure["ff_weights"], structure["rec_weights"]])
    row_types = np.concatenate([ff_ct, ct + n_ff_types])
    row_known = np.concatenate([known_ff, structure["modelled"]])

    parameters = StudentParameters()

    def log_sf(a, b):
        key = f"{combined_names[a]}__{rec_names[b]}"
        if key not in parameters.log_sf:
            parameters.log_sf[key] = nn.Parameter(torch.zeros(()))
        return parameters.log_sf[key]

    # Positions of each modelled neuron within the modelled neurons of its type
    # (columns of learnt matrices).
    type_position = np.full(ct.size, -1)
    for b in range(len(rec_names)):
        members = modelled[ct[modelled] == b]
        type_position[members] = np.arange(members.size)

    def learnt_recurrence_block(a, b, source_ids, target_ids):
        key = f"{combined_names[a]}__{rec_names[b]}"
        if key not in parameters.log_weights:
            # Archived no-connectome init: fully connected at the mean non-zero weight.
            src = modelled[ct[modelled] == a - n_ff_types]
            tgt = modelled[ct[modelled] == b]
            block = structure["rec_weights"][np.ix_(src, tgt)]
            nonzero = block[block != 0]
            mean = float(nonzero.mean()) if nonzero.size else 1e-8
            init = np.log(mean * perturbation[a, b])
            parameters.log_weights[key] = nn.Parameter(
                torch.full((src.size, tgt.size), init, dtype=torch.float32)
            )
        return SlicedFullRankProjection(
            parameters.log_weights[key],
            type_position[source_ids],
            type_position[target_ids],
        )

    # Learnt sources of each type, and their positions (rows of U).
    learnt_rows_global = np.flatnonzero(~row_known)
    learnt_position = np.full(row_types.size, -1)
    for a in range(len(combined_names)):
        members = learnt_rows_global[row_types[learnt_rows_global] == a]
        learnt_position[members] = np.arange(members.size)

    def low_rank_factors(a, b):
        key = f"{combined_names[a]}__{rec_names[b]}"
        if key not in parameters.U:
            src = learnt_rows_global[row_types[learnt_rows_global] == a]
            tgt = modelled[ct[modelled] == b]
            # Archived "constant" init: the block's mean teacher weight, perturbed, as
            # an exactly rank-1 log-space matrix (U V = log value everywhere).
            mean = float(weights[np.ix_(src, tgt)].mean()) if src.size else 1e-8
            value = np.log(max(mean * perturbation[a, b], 1e-8))
            scale = np.sqrt(abs(value))
            U = torch.zeros(src.size, low_rank)
            V = torch.zeros(low_rank, tgt.size)
            U[:, 0] = scale
            V[0, :] = np.sign(value) * scale
            parameters.U[key] = nn.Parameter(U)
            parameters.V[key] = nn.Parameter(V)
        return parameters.U[key], parameters.V[key]

    def projections(rows, targets, namespace_offset):
        """(source name, target name) -> Projection for global rows onto targets.

        ``namespace_offset`` is 0 for input rows in the combined namespace and
        ``n_ff_types`` for a layer's own recurrent rows (recurrent namespace).
        """
        out = {}
        types = row_types[rows]
        for a in np.unique(types):
            source_rows = rows[types == a]
            known = row_known[source_rows]
            name = (
                combined_names[a]
                if namespace_offset == 0
                else rec_names[a - n_ff_types]
            )
            for b in range(len(rec_names)):
                target_ids = targets[ct[targets] == b]
                if target_ids.size == 0:
                    continue
                if has_learnt_sources and namespace_offset == 0:
                    U, V = low_rank_factors(a, b) if (~known).any() else (None, None)
                    out[(name, rec_names[b])] = MixedProjection(
                        connectome=weights[np.ix_(source_rows[known], target_ids)]
                        * perturbation[a, b],
                        known_rows=np.flatnonzero(known),
                        learnt_rows=np.flatnonzero(~known),
                        log_sf=log_sf(a, b),
                        U=U,
                        V=V,
                        u_rows=learnt_position[source_rows[~known]],
                        v_cols=type_position[target_ids],
                    )
                elif learnt_recurrence and a >= n_ff_types:
                    out[(name, rec_names[b])] = learnt_recurrence_block(
                        a, b, source_rows - n_ff, target_ids
                    )
                else:
                    projection = ScalingFactorProjection(
                        connectome=(
                            weights[np.ix_(source_rows, target_ids)]
                            * perturbation[a, b]
                        ).astype(np.float32)
                    )
                    projection.log_sf = log_sf(a, b)
                    out[(name, rec_names[b])] = projection
        return out

    ff_rows = np.arange(n_ff)
    observed_rows = n_ff + observed
    unobserved_rows = n_ff + unobserved
    injected_rows = n_ff + unreconstructed

    layer2_rows = np.concatenate(
        [ff_rows, unobserved_rows, observed_rows, injected_rows]
    )
    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=projections(layer2_rows, observed, 0),
        cell_type_indices=ct[observed],
        cell_type_indices_FF=row_types[layer2_rows],
        cell_params=phys["rec_cells"],
        cell_params_FF=phys["combined_cells"],
        synapse_params_FF=phys["combined_synapses"],
        surrgrad_scale=surrgrad_scale,
        batch_size=batch_size,
        track_variables=False,
    )

    if unobserved.size == 0:
        model = ObservedOnlySNN(layer2)
    else:
        layer1_rows = np.concatenate([ff_rows, observed_rows, injected_rows])
        layer1 = ConductanceLIFNetwork(
            dt=dt,
            rec_projections=projections(unobserved_rows, unobserved, n_ff_types),
            ff_projections=projections(layer1_rows, unobserved, 0),
            cell_type_indices=ct[unobserved],
            cell_type_indices_FF=row_types[layer1_rows],
            cell_params=phys["rec_cells"],
            cell_params_FF=phys["combined_cells"],
            synapse_params=phys["rec_synapses"],
            synapse_params_FF=phys["combined_synapses"],
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=False,
        )
        model = TwoLayerSNN(layer1, layer2, n_ff=n_ff, return_hidden_spikes=True)

    model.student_parameters = parameters
    return model, parameters


def count_free_parameters(parameters):
    return int(sum(p.numel() for p in parameters.parameters()))


def scaling_factors_relative_to_target(parameters, structure, params):
    """Learnt scaling factor / correct scaling factor, per pathway (1.0 = recovered)."""
    phys = physiology(params)
    target = structure["target_scaling_factors"]
    out = {}
    for key, log_value in parameters.log_sf.items():
        source, destination = key.split("__")
        a = phys["combined_names"].index(source)
        b = phys["rec_names"].index(destination)
        out[f"{source}_to_{destination}"] = float(
            np.exp(log_value.detach().cpu().item()) / target[a, b]
        )
    return out
