"""What the student is given: which neurons it models and observes, and its weights.

Every figure is one student built from the teacher by a handful of manipulations,
all read from the ``[student]`` table of ``parameters.toml``:

    observed_fraction         fraction of *modelled* neurons whose spikes enter the loss
    neuron_removal_fraction   neurons deleted from the student entirely (Fig 4a)
    synapse_dropout_fraction  recurrent synapses deleted, every neuron kept (Fig 4b)
    weight_noise              archived multiplicative log-normal noise, per cell-type pair (Fig 5)
    recurrent_model           "connectome" | "learnt" | "shuffle_weights" | "shuffle_inputs" |
                              "configuration_model" (Fig 2)
    reconstructed_fraction    fraction of the 6500 pooled units in the reconstructed segment S (Fig 6)
    recorded_pool_fraction    fixed recorded pool over the 5000 recurrent neurons (Fig 6)

The result is saved as ``student_structure.npz`` in the run directory, so that
evaluation rebuilds exactly the network that was trained rather than re-deriving it.

Indices are teacher neuron ids throughout. Source types use the combined namespace
``[feedforward types..., recurrent types...]``: 0 = mitral, 1 = excitatory, 2 = inhibitory.
"""

from pathlib import Path

import numpy as np

#: Recurrent-model options for the Figure 2 controls.
RECURRENT_MODELS = (
    "connectome",
    "learnt",
    "shuffle_weights",
    "shuffle_inputs",
    "configuration_model",
)

# Independent random streams, so changing one manipulation's level never changes
# the draws of another (e.g. the observed split is identical across a noise sweep).
_STREAM_PERTURBATION = 0
_STREAM_REMOVAL = 1
_STREAM_OBSERVED = 2
_STREAM_REWIRE = 3
_STREAM_DROPOUT = 4
_STREAM_NOISE = 5
_STREAM_SEGMENT = 6
_STREAM_POOL = 7


def _rng(seed, stream):
    return np.random.default_rng([int(seed), stream])


def load_teacher(input_dir):
    """Teacher connectome with connectivity masks applied."""
    structure = np.load(Path(input_dir) / "network_structure.npz")
    rec = structure["recurrent_weights"] * structure["recurrent_connectivity"]
    ff = structure["feedforward_weights"] * structure["feedforward_connectivity"]
    return {
        "rec_weights": rec.astype(np.float32),
        "ff_weights": ff.astype(np.float32),
        "cell_type_indices": structure["cell_type_indices"].astype(np.int64),
        "ff_cell_type_indices": structure["feedforward_cell_type_indices"].astype(
            np.int64
        ),
    }


# =====================================================================
# Weight manipulations (archived implementations, copied unchanged)
# =====================================================================


def apply_weight_noise(weights, noise_frac, rng):
    """Archived noisy-weights noise: log-normal multiplier, affine rescale, clip at 0.

    Returns the noisy weights and the number of non-zero weights the affine rescale
    pushed below zero (and which were therefore clipped to zero).
    """
    nonzero_mask = weights != 0
    orig_mean = weights[nonzero_mask].mean()
    orig_std = weights[nonzero_mask].std()

    multiplier = np.exp(
        noise_frac * rng.standard_normal(weights.shape) - noise_frac**2 / 2
    )
    noisy_weights = weights * multiplier

    n_clipped = 0
    if orig_std > 0:
        noisy_nz = noisy_weights[nonzero_mask]
        noisy_std = noisy_nz.std()
        if noisy_std > 0:
            rescaled = orig_mean + (noisy_nz - noisy_nz.mean()) * (orig_std / noisy_std)
            n_clipped = int((rescaled < 0).sum())
            noisy_weights[nonzero_mask] = np.maximum(rescaled, 0)

    return noisy_weights, n_clipped


def shuffle_weights_within_connectome(weights, cell_type_indices, rng):
    """Archived shuffle-weights control: permute non-zero values within each block."""
    weights = weights.copy()
    n_types = int(cell_type_indices.max()) + 1
    for in_type in range(n_types):
        for out_type in range(n_types):
            pair = np.outer(cell_type_indices == in_type, cell_type_indices == out_type)
            pair &= weights != 0
            values = weights[pair]
            weights[pair] = values[rng.permutation(values.size)]
    return weights


def shuffle_weights_within_neuron(weights, cell_type_indices, rng):
    """Permute each neuron's input weights among its own presynaptic partners.

    Within one postsynaptic neuron and one presynaptic cell type, the multiset of incoming
    weights is reassigned at random over that neuron's existing inputs. Topology, Dale's
    law, each neuron's total input from each cell type and its whole distribution of input
    strengths are preserved exactly; only *which* partner supplies which strength changes.
    The weight analogue of the configuration model, and a gentler control than
    :func:`shuffle_weights_within_connectome`, which redraws a neuron's drive entirely.
    """
    weights = weights.copy()
    for cell_type in range(int(cell_type_indices.max()) + 1):
        rows = np.flatnonzero(cell_type_indices == cell_type)
        block = weights[rows]
        present = block != 0
        # Sort each column by a random key with absent synapses last: the first
        # present.sum(axis=0) entries are that neuron's weights in random order.
        keys = np.where(present, rng.random(block.shape), np.inf)
        shuffled = np.take_along_axis(block, np.argsort(keys, axis=0), axis=0)
        # Slots to write them back into, in row order, again with absent synapses last.
        slots = np.argsort(~present, axis=0, kind="stable")
        np.put_along_axis(block, slots, shuffled, axis=0)
        weights[rows] = block
    return weights


def configuration_model_rewire(weights, cell_type_indices, rng, max_iterations=1000):
    """Rewire each cell-type block, preserving every in- and out-degree.

    Within a block the out-stubs (presynaptic ends) stay put and the in-stubs are
    randomly re-paired with them, which preserves both degree sequences exactly.
    Self-connections and duplicate synapses created by the re-pairing are removed
    by swapping their targets with randomly chosen edges until none remain. The
    block's weight values are then randomly reassigned to the new synapses, so the
    within-block weight distribution is preserved too.
    """
    n = weights.shape[0]
    rewired = np.zeros_like(weights)
    n_types = int(cell_type_indices.max()) + 1
    for in_type in range(n_types):
        for out_type in range(n_types):
            sources = np.flatnonzero(cell_type_indices == in_type)
            targets = np.flatnonzero(cell_type_indices == out_type)
            block = weights[np.ix_(sources, targets)]
            rows, cols = np.nonzero(block)
            if rows.size == 0:
                continue
            values = block[rows, cols]
            pre = sources[rows]
            post = targets[cols][rng.permutation(cols.size)]

            for _ in range(max_iterations):
                keys = pre * n + post
                _, first = np.unique(keys, return_index=True)
                bad = np.ones(keys.size, dtype=bool)
                bad[first] = False
                bad |= pre == post
                n_bad = int(bad.sum())
                if n_bad == 0:
                    break
                # Swap each conflicting edge's target with a distinct non-conflicting
                # edge. Disjoint pairs keep this a permutation of the in-stubs.
                bad_idx = np.flatnonzero(bad)
                partners = rng.choice(np.flatnonzero(~bad), size=n_bad, replace=False)
                post[bad_idx], post[partners] = post[partners], post[bad_idx]
            else:
                raise RuntimeError(
                    f"configuration model did not converge for block "
                    f"{in_type}->{out_type} ({n_bad} conflicts left)"
                )
            rewired[pre, post] = values[rng.permutation(values.size)]
    return rewired


# =====================================================================
# Structure
# =====================================================================


def build_student_structure(teacher, student_cfg, seed, perturbation_variance):
    """Draw every random choice the student depends on and apply the manipulations.

    Args:
        teacher: Output of :func:`load_teacher`.
        student_cfg: The ``[student]`` table.
        seed: Run seed; each manipulation draws from its own stream.
        perturbation_variance: Variance of the log-normal scaling-factor perturbation.

    Returns:
        Dict of arrays, saved verbatim as ``student_structure.npz``.
    """
    ct = teacher["cell_type_indices"]
    ff_ct = teacher["ff_cell_type_indices"]
    rec = teacher["rec_weights"].copy()
    ff = teacher["ff_weights"].copy()
    n_rec, n_ff = rec.shape[0], ff.shape[0]
    n_ff_types = int(ff_ct.max()) + 1
    n_rec_types = int(ct.max()) + 1

    recurrent_model = student_cfg.get("recurrent_model", "connectome")
    if recurrent_model not in RECURRENT_MODELS:
        raise ValueError(f"recurrent_model must be one of {RECURRENT_MODELS}")
    removal = float(student_cfg.get("neuron_removal_fraction", 0.0))
    dropout = float(student_cfg.get("synapse_dropout_fraction", 0.0))
    noise = float(student_cfg.get("weight_noise", 0.0))
    reconstructed = student_cfg.get("reconstructed_fraction")

    # --- Scaling-factor perturbation (archived scheme): the student's weights are
    # multiplied by a per-pair factor, so the correct scaling factor is its inverse.
    sigma = np.sqrt(perturbation_variance)
    perturbation = _rng(seed, _STREAM_PERTURBATION).lognormal(
        mean=-(sigma**2) / 2.0,
        sigma=sigma,
        size=(n_ff_types + n_rec_types, n_rec_types),
    )

    # --- Which neurons and inputs are reconstructed.
    known_ff = np.ones(n_ff, dtype=bool)
    modelled = np.ones(n_rec, dtype=bool)
    kappa = np.full(n_rec, np.nan)

    if reconstructed is not None:
        # Figure 6: a random segment S of the pooled units. Everything outside S has
        # known activity but unknown weights, and is injected with learnt weights.
        n_pooled = n_ff + n_rec
        n_segment = round(float(reconstructed) * n_pooled)
        in_segment = np.zeros(n_pooled, dtype=bool)
        in_segment[_rng(seed, _STREAM_SEGMENT).permutation(n_pooled)[:n_segment]] = True
        known_ff = in_segment[:n_ff]
        modelled = in_segment[n_ff:]

        pool_fraction = float(student_cfg["recorded_pool_fraction"])
        recorded_pool = np.zeros(n_rec, dtype=bool)
        n_pool = round(pool_fraction * n_rec)
        recorded_pool[_rng(seed, _STREAM_POOL).permutation(n_rec)[:n_pool]] = True
        observed = modelled & recorded_pool

        total_input = ff.sum(axis=0) + rec.sum(axis=0)
        known_input = ff[known_ff].sum(axis=0) + rec[modelled].sum(axis=0)
        kappa[modelled] = known_input[modelled] / total_input[modelled]
    else:
        if removal > 0:
            n_removed = round(removal * n_rec)
            removed = _rng(seed, _STREAM_REMOVAL).permutation(n_rec)[:n_removed]
            modelled[removed] = False

        modelled_ids = np.flatnonzero(modelled)
        n_observed = int(
            round(float(student_cfg["observed_fraction"]) * modelled_ids.size)
        )
        observed = np.zeros(n_rec, dtype=bool)
        observed[
            _rng(seed, _STREAM_OBSERVED).permutation(modelled_ids)[:n_observed]
        ] = True

    lost_input = np.zeros(n_rec, dtype=np.float64)
    if removal > 0:
        lost_input += rec[~modelled].sum(axis=0)

    # --- Recurrent connectivity manipulations.
    if recurrent_model == "shuffle_weights":
        rec = shuffle_weights_within_connectome(rec, ct, _rng(seed, _STREAM_REWIRE))
    elif recurrent_model == "shuffle_inputs":
        rec = shuffle_weights_within_neuron(rec, ct, _rng(seed, _STREAM_REWIRE))
    elif recurrent_model == "configuration_model":
        rec = configuration_model_rewire(rec, ct, _rng(seed, _STREAM_REWIRE))

    if dropout > 0:
        rows, cols = np.nonzero(rec)
        dropped = _rng(seed, _STREAM_DROPOUT).random(rows.size) < dropout
        np.add.at(lost_input, cols[dropped], rec[rows[dropped], cols[dropped]])
        rec[rows[dropped], cols[dropped]] = 0.0

    if removal > 0 or dropout > 0:
        total_rec_input = teacher["rec_weights"].sum(axis=0)
        kappa[modelled] = lost_input[modelled] / total_rec_input[modelled]

    # --- Weight noise, per (source type, target type) pair over [ff; rec] (archived).
    n_noise_clipped = 0
    n_noise_weights = 0
    if noise > 0:
        noise_rng = _rng(seed, _STREAM_NOISE)
        combined = np.concatenate([ff, rec], axis=0)
        combined_types = np.concatenate([ff_ct, ct + n_ff_types])
        for in_type in range(n_ff_types + n_rec_types):
            for out_type in range(n_rec_types):
                pair = np.outer(combined_types == in_type, ct == out_type)
                values = combined[pair]
                if not (values != 0).any():
                    continue
                noisy, n_clipped = apply_weight_noise(values, noise, noise_rng)
                combined[pair] = noisy
                n_noise_clipped += n_clipped
                n_noise_weights += int((values != 0).sum())
        ff, rec = combined[:n_ff], combined[n_ff:]

    return {
        "cell_type_indices": ct,
        "ff_cell_type_indices": ff_ct,
        "ff_weights": ff.astype(np.float32),
        "rec_weights": rec.astype(np.float32),
        "perturbation": perturbation.astype(np.float64),
        "target_scaling_factors": (1.0 / perturbation).astype(np.float64),
        "modelled": modelled,
        "observed": observed & modelled,
        "known_ff": known_ff,
        "kappa": kappa,
        "recurrent_model": np.array(recurrent_model),
        # Figure 6: units outside the reconstructed segment keep their known activity
        # and are injected through learnt weights. Removed neurons (Fig 4) are not.
        "inject_unreconstructed": np.array(reconstructed is not None),
        "noise_clipped_fraction": np.array(
            n_noise_clipped / n_noise_weights if n_noise_weights else 0.0
        ),
    }


def save_structure(structure, output_dir):
    np.savez_compressed(Path(output_dir) / "student_structure.npz", **structure)


def load_structure(run_dir):
    with np.load(Path(run_dir) / "student_structure.npz") as data:
        structure = {key: data[key] for key in data.files}
    structure["recurrent_model"] = str(structure["recurrent_model"])
    return structure
