"""
PCA of log-transformed gaussian-smoothed feedforward inputs.

Identical to compute-pcas except that the smoothed firing rates are
log-transformed before fitting PCA:

    log_rates = log(rates + log_epsilon)

where log_epsilon avoids log(0).  The resulting PC timeseries and components
are in log-rate space; reconstruction requires inverting:

    rates ≈ exp(pc_timeseries @ components + mean) - log_epsilon

Input data is accessed via symlinks in input_dir (network_structure.npz,
spike_data.zarr). Nothing is copied — results go to output_dir/results/.

Outputs
-------
output_dir/results/log_pca_inputs.zarr
    pc_timeseries      (n_batches, n_timesteps, n_components)  float32
        Low-dimensional representation of each trial in log-rate space.
    components         (n_components, n_feedforward)           float32
        Rows are principal axes in log-rate space.
        Reconstruction: exp(pc_timeseries @ components + mean) - log_epsilon
    mean               (n_feedforward,)                        float32
        Per-neuron mean log-rate subtracted before PCA.
    explained_variance_ratio  (n_components,)                  float32
    Attributes: gaussian_sigma_ms, n_components, log_epsilon, dt
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
import tomllib

try:
    from cuml.decomposition import PCA

    CUML_AVAILABLE = True
except ImportError:
    from sklearn.decomposition import PCA

    CUML_AVAILABLE = False


def gaussian_smooth(spikes, sigma, dt=1.0):
    """Gaussian-smooth spike trains along the time axis on GPU.

    Args:
        spikes: (batch, time, neurons) float32 tensor
        sigma: kernel std in timesteps
        dt: unused, kept for API compatibility

    Returns:
        Smoothed tensor of same shape.
    """
    width = int(6 * sigma)
    t = torch.arange(-width, width + 1, dtype=torch.float32)
    kernel = torch.exp(-(t**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).to(spikes.device)
    x = spikes.permute(0, 2, 1)  # (batch, neurons, time)
    x = x.reshape(-1, 1, x.shape[-1])
    smoothed = F.conv1d(x, kernel, padding=width)
    return smoothed.reshape(spikes.shape[0], spikes.shape[2], spikes.shape[1]).permute(
        0, 2, 1
    )


def main(input_dir, output_dir, params_file, wandb_config=None, resume_from=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    params_file = Path(params_file)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load parameters
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COMPUTE LOG-PCAs — log-rate feedforward input compression")
    print("=" * 60)

    with open(params_file, "rb") as f:
        params = tomllib.load(f)

    gaussian_sigma_ms = float(params["pca"]["gaussian_sigma_ms"])
    n_components = int(params["pca"]["n_components"])
    log_epsilon = float(params["pca"].get("log_epsilon", 0.1))

    backend = "cuML (GPU)" if CUML_AVAILABLE else "scikit-learn (CPU)"
    print(f"\nBackend              : {backend}")
    print("\nParameters")
    print(f"  Gaussian smoothing σ : {gaussian_sigma_ms} ms")
    print(f"  Number of PCs        : {n_components}")
    print(f"  Log epsilon          : {log_epsilon}  (added before log to avoid log(0))")

    # ------------------------------------------------------------------
    # Load spike data
    # ------------------------------------------------------------------
    print(f"\nLoading spike data from {input_dir / 'spike_data.zarr'} ...")
    spike_data = zarr.open_group(input_dir / "spike_data.zarr", mode="r")
    dt = float(spike_data.attrs["dt"])  # ms per timestep

    input_spikes = spike_data["input_spikes"][:]  # (n_batches, n_timesteps, n_ff)

    n_batches, n_timesteps, n_ff = input_spikes.shape
    duration_s = n_timesteps * dt / 1000.0

    print(
        f"  Input spikes shape   : {input_spikes.shape}  (batches × timesteps × neurons)"
    )
    print(f"  dt                   : {dt} ms")
    print(f"  Trial duration       : {duration_s:.1f} s  ({n_timesteps} timesteps)")
    print(f"  Feedforward neurons  : {n_ff}")
    print(f"  Batches              : {n_batches}")

    mean_ff_rate = input_spikes.mean() / (dt / 1000.0)
    print(f"  Mean FF firing rate  : {mean_ff_rate:.2f} Hz")

    # ------------------------------------------------------------------
    # Gaussian-smooth feedforward spikes along time axis
    # ------------------------------------------------------------------
    sigma_steps = gaussian_sigma_ms / dt
    smooth_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nGaussian-smoothing spike trains ...")
    print(f"  σ = {gaussian_sigma_ms} ms  →  {sigma_steps:.1f} timesteps")
    print(f"  Smoothing device     : {smooth_device}")

    spikes_t = torch.from_numpy(input_spikes.astype(np.float32)).to(smooth_device)
    rates = gaussian_smooth(spikes_t, sigma=sigma_steps).cpu().numpy()
    # (n_batches, n_timesteps, n_ff)
    del spikes_t
    if smooth_device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Smoothed rates shape : {rates.shape}")
    print(
        f"  Rate range           : [{rates.min():.4f}, {rates.max():.4f}] spk/timestep"
    )

    # ------------------------------------------------------------------
    # Log-transform
    # ------------------------------------------------------------------
    print(f"\nLog-transforming rates  (log(rates + {log_epsilon})) ...")
    log_rates = np.log(rates + log_epsilon)
    print(f"  Log-rate range       : [{log_rates.min():.4f}, {log_rates.max():.4f}]")

    # ------------------------------------------------------------------
    # Fit PCA across all batches × timesteps (in log-rate space)
    # ------------------------------------------------------------------
    flat = log_rates.reshape(n_batches * n_timesteps, n_ff)  # (N, n_ff)
    N = flat.shape[0]

    print("\nFitting PCA on log-rates ...")
    print(f"  Data matrix shape    : {flat.shape}  ({N} samples × {n_ff} features)")
    print(f"  Fitting {n_components} components ...")

    pca = PCA(n_components=n_components)
    pc_flat = np.asarray(pca.fit_transform(flat))  # (N, n_components)

    pc_timeseries = pc_flat.reshape(n_batches, n_timesteps, n_components)

    cumev = np.cumsum(np.asarray(pca.explained_variance_ratio_))
    total_var = float(cumev[-1])
    print("  Done.")
    print(
        f"  Variance explained   : {total_var:.2%}  ({n_components} PCs, in log-rate space)"
    )
    for thresh in [0.50, 0.75, 0.90, 0.95, 0.99]:
        idx = int(np.searchsorted(cumev, thresh))
        if idx < n_components:
            print(f"    {thresh:.0%} variance     : {idx + 1} PCs")
        else:
            print(f"    {thresh:.0%} variance     : > {n_components} PCs (need more)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = results_dir / "log_pca_inputs.zarr"
    print(f"\nSaving results to {out_path} ...")

    pca_out = zarr.open_group(out_path, mode="w")

    pc_f32 = pc_timeseries.astype(np.float32)
    pca_out.create_dataset(
        "pc_timeseries",
        shape=pc_f32.shape,
        dtype=np.float32,
        data=pc_f32,
        chunks=(1, n_timesteps, n_components),
    )
    print(f"  pc_timeseries        : {pc_f32.shape}")

    # components[i] is the i-th principal axis in log-rate space.
    # Reconstruction: exp(pc_timeseries @ components + mean) - log_epsilon
    comp_f32 = np.asarray(pca.components_).astype(np.float32)
    pca_out.create_dataset(
        "components",
        shape=comp_f32.shape,
        dtype=np.float32,
        data=comp_f32,
    )
    print(
        f"  components           : {comp_f32.shape}  "
        f"(reconstruction: exp(scores @ components + mean) - {log_epsilon})"
    )

    mean_f32 = np.asarray(pca.mean_).astype(np.float32)
    pca_out.create_dataset(
        "mean",
        shape=mean_f32.shape,
        dtype=np.float32,
        data=mean_f32,
    )
    print(f"  mean                 : {mean_f32.shape}  (mean log-rate per neuron)")

    evr_f32 = np.asarray(pca.explained_variance_ratio_).astype(np.float32)
    pca_out.create_dataset(
        "explained_variance_ratio",
        shape=evr_f32.shape,
        dtype=np.float32,
        data=evr_f32,
    )
    print(f"  explained_var_ratio  : {evr_f32.shape}")

    pca_out.attrs["gaussian_sigma_ms"] = gaussian_sigma_ms
    pca_out.attrs["n_components"] = n_components
    pca_out.attrs["log_epsilon"] = log_epsilon
    pca_out.attrs["dt"] = dt

    print(f"\n✓ Done. Results written to {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.params_file)
