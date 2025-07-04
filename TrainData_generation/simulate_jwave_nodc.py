# === Standard Library ===
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor  # For parallel processing (e.g., saving data)

# === Numerical and Scientific Computing ===
import numpy as np
import pandas as pd
import h5py

# === JAX and JAXDF ===
import jax
from jax import jit
import jax.numpy as jnp
from jaxdf import FourierSeries  # Note: Same name as the one in jwave, keep only one based on usage

# === jwave Library ===
from jwave import FourierSeries  # If conflicting with jaxdf, rename as: `as JWFourierSeries`
from jwave.geometry import *     # Imports domain, medium, time axis, etc.
from jwave.geometry import Sources  # Optional, already included above via wildcard

# === Custom Modules ===
from utils import simulate_wave_propagation  # Your custom wave propagation function


def generate_random_sources_and_signals(domain, time_axis):
    """
    Generate tone burst ultrasound sources in 2D or 3D, with randomized delay.

    - In 2D: Sources are linearly aligned along y-axis at fixed x.
    - In 3D: Sources form a 2D grid in the X-Y plane at fixed Z-depth.
    - Signal: Tone burst waveform with randomized delay per source.

    Returns:
        sources (Sources): Configured source object for simulation.
    """

    dim = len(domain.N)
    if dim not in [2, 3]:
        raise ValueError(f"Only 2D and 3D domains are supported. Got dimension {dim}.")

    # === Common Signal Parameters ===
    signal_freq = 40_000  # Hz
    num_cycles = random.randint(2, 4)
    amplitude = 250

    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    sample_freq = 1.0 / time_axis.dt
    max_delay = 1 / signal_freq
    tone_len = num_cycles * int(sample_freq / signal_freq)
    base_wave = amplitude * jnp.sin(2 * jnp.pi * signal_freq * t[:tone_len])

    if dim == 2:
        # === 2D Source Placement ===
        num_sources = 10
        spacing = 10
        fixed_x = 15
        center_y = domain.N[1] // 2
        start_y = center_y - ((num_sources - 1) // 2) * spacing
        y_positions = tuple(start_y + i * spacing for i in range(num_sources))
        x_positions = (fixed_x,) * num_sources
        positions = (x_positions, y_positions)

    else:
        # === 3D Source Placement ===
        num_sources_x = 10
        num_sources_y = 10
        spacing = 10
        fixed_z = 15
        center_x = domain.N[0] // 2
        center_y = domain.N[1] // 2
        start_x = center_x - ((num_sources_x - 1) * spacing) // 2
        start_y = center_y - ((num_sources_y - 1) * spacing) // 2

        x_positions = [start_x + i * spacing for i in range(num_sources_x)] * num_sources_y
        y_positions = []
        for j in range(num_sources_y):
            y_positions.extend([start_y + j * spacing] * num_sources_x)
        z_positions = [fixed_z] * (num_sources_x * num_sources_y)
        positions = (tuple(x_positions), tuple(y_positions), tuple(z_positions))

    # === Generate delayed tone burst signals for each source ===
    num_sources_total = len(positions[0])
    signals = []
    for _ in range(num_sources_total):
        delay = random.uniform(0, max_delay)
        delay_samples = int(delay / time_axis.dt)

        delayed_signal = jnp.zeros_like(t).at[delay_samples:delay_samples + tone_len].set(base_wave)
        signals.append(delayed_signal)

    signals = jnp.stack(signals)

    # === Build and return source object ===
    sources = Sources(
        positions=positions,
        signals=signals,
        dt=time_axis.dt,
        domain=domain,
    )

    return sources

def save_hdf5(hdf5_filename, stacked_pressure, chunk_size=64):
    """
    Save simulation data to an HDF5 file.

    This function handles both 3D (T, H, W) and 4D (T, H, W, L) pressure field data, and
    automatically sets appropriate chunk sizes for efficient storage and access.

    Args:
        hdf5_filename (str): Path to the output HDF5 file.
        stacked_pressure (np.ndarray): Simulated pressure field with shape (T, H, W) or (T, H, W, L).
        density_patch (np.ndarray): Corresponding density values.
        sound_speed_patch (np.ndarray): Corresponding sound speed values.
        chunk_size (int): Target chunk size along each spatial dimension (default: 64).
    """
    shape = stacked_pressure.shape
    ndim = len(shape)

    # Determine chunk shapes based on dimensionality
    if ndim == 3:
        T, H, W = shape
        chunks_p = (1, min(H, chunk_size), min(W, chunk_size))
    elif ndim == 4:
        T, H, W, L = shape
        chunks_p = (1, min(H, chunk_size), min(W, chunk_size), min(L, chunk_size))
    else:
        raise ValueError(f"Unsupported pressure shape: {shape}")

    # Write data to HDF5 file with gzip compression
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset(
            "pressure",
            data=stacked_pressure,
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
            chunks=chunks_p,
        )


if __name__ == "__main__":

    # === Simulation Parameters ===
    N               = 10              # Number of simulations to run
    simulation_time = 0.001         # Duration of each simulation in seconds
    dt              = 1e-8           # Time step interval for j-Wave
    save_step       = 500            # Interval in time steps for recording pressure field
    start_idx       = 10              # Starting index for simulation numbering

    # === Domain Settings ===
    domain_shape    = (276, 276)     # Shape of the simulation grid (2D or 3D)
    grid_spacing    = (1e-3, 1e-3)   # Grid spacing in meters for each axis
    margin          = 0             # Number of grid points to trim at each boundary (e.g., to remove PML)

    # === Patch Extraction Parameters ===
    window_size     = 64             # Size of each spatial patch
    stride          = 32             # Step size when sliding the window over the spatial domain
    threshold       = 0.01           # Threshold to determine if a patch contains meaningful data

    # === Output Settings ===
    skip_steps      = 30             # Number of initial time steps to ignore (before pressure field stabilizes)
    save_patch      = True           # Whether to save patch data
    patch_output_dir = "./training_data_patch_2d_0630_hom"
    hdf5_filename    = "./training_data_2d_0630_hom.h5"

    # === Derived Settings ===
    os.makedirs(patch_output_dir, exist_ok=True)
    save_shape = tuple(dim - 2 * margin for dim in domain_shape)
    num_saved_steps = int(simulation_time / dt / save_step) - skip_steps
    num_patches_x = (save_shape[0] - window_size) // stride + 1
    num_patches_y = (save_shape[1] - window_size) // stride + 1
    num_patches_z = (save_shape[2] - window_size) // stride + 1 if len(domain_shape) == 3 else 1

    if len(domain_shape) != len(grid_spacing):
        raise ValueError("domain_shape and grid_spacing must have the same dimensions")

    devices = jax.devices()

    if any("cuda" in str(device).lower() for device in devices):
        print("Running on GPU, using Cuda")
    else:
        print("Running on CPU")

    with h5py.File(hdf5_filename, "w") as f:
        # Create pre-allocated HDF5 datasets for full-resolution simulation outputs
        dset_pressure = f.create_dataset("pressure", shape=(N, num_saved_steps, *save_shape),
                                         dtype=np.float32, compression="gzip", compression_opts=9,
                                         chunks=(1, 1, *save_shape))

        for i in range(start_idx, start_idx + N):
            print(f"Running simulation {i+1}/{start_idx + N}...")
            start_time = time.time()

            domain = Domain(domain_shape, grid_spacing)

            # Initialize homogeneous field and embed heterogeneous regions
            density = np.ones(domain.N) * 1.38
            sound_speed = np.ones(domain.N) * 343

            # Construct Fourier fields
            density_fs = FourierSeries(np.expand_dims(density, -1), domain)
            sound_speed_fs = FourierSeries(np.expand_dims(sound_speed, -1), domain)
            medium = Medium(domain=domain, sound_speed=sound_speed_fs, density=density_fs)

            time_axis = TimeAxis(dt=float(dt), t_end=float(simulation_time))
            sources = generate_random_sources_and_signals(domain, time_axis)

            @jit
            def compiled_simulator():
                return simulate_wave_propagation(medium, time_axis, sources=sources, save_step=save_step)

            simulation_start_time = time.time()
            pressure = compiled_simulator()
            print(f"jwave simulate time: {time.time()-simulation_start_time}")

            pressure = np.squeeze(pressure, axis=-1)
            if np.any(np.isnan(pressure)):
                print(f"NaN detected in pressure at simulation {i+1}, skipping...")
                continue

            # Crop boundary + skip initial unstable steps
            crop_slices = tuple(slice(margin, dim - margin) for dim in domain_shape)
            full_slices = (slice(skip_steps, None),) + crop_slices
            pressure = pressure[full_slices]

            dset_pressure[i-start_idx] = pressure

            # Patch slicing and saving (multithreaded)
            if save_patch:
                with ThreadPoolExecutor(max_workers=14) as executor:
                    if len(domain.N) == 2:
                        for x in range(num_patches_x):
                            for y in range(num_patches_y):
                                x_start, y_start = x * stride, y * stride

                                stacked_pressure, patch_count = [], 0
                                for t in range(num_saved_steps):
                                    patch = pressure[t, x_start:x_start + window_size, y_start:y_start + window_size]
                                    if np.sum(np.abs(patch) > threshold) < 1000:
                                        if patch_count > 1:
                                            executor.submit(save_hdf5, os.path.join(
                                                patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                np.stack(stacked_pressure), chunk_size=window_size)
                                        stacked_pressure, patch_count = [], 0
                                    else:
                                        stacked_pressure.append(patch)
                                        patch_count += 1
                                if patch_count > 1:
                                    executor.submit(save_hdf5, os.path.join(
                                        patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                        np.stack(stacked_pressure), chunk_size=window_size)

                    else:
                        for x in range(num_patches_x):
                            for y in range(num_patches_y):
                                for z in range(num_patches_z):
                                    x_start, y_start, z_start = x * stride, y * stride, z * stride
                                    
                                    stacked_pressure, patch_count = [], 0
                                    for t in range(num_saved_steps):
                                        patch = pressure[t, x_start:x_start + window_size,
                                                        y_start:y_start + window_size,
                                                        z_start:z_start + window_size]
                                        if np.sum(np.abs(patch) > threshold) < 3000:
                                            if patch_count > 1:
                                                executor.submit(save_hdf5, os.path.join(
                                                    patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                    np.stack(stacked_pressure), chunk_size=window_size)
                                            stacked_pressure, patch_count = [], 0
                                        else:
                                            stacked_pressure.append(patch)
                                            patch_count += 1
                                    if patch_count > 1:
                                        executor.submit(save_hdf5, os.path.join(
                                            patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                            np.stack(stacked_pressure), chunk_size=window_size)

            print(f"Simulation {i+1} saved, total time: {time.time() - start_time:.2f} s")
