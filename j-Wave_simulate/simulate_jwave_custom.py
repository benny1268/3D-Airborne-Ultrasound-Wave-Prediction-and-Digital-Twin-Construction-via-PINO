# === Standard Library ===
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor  # For parallel processing (e.g., saving data)

# === Numerical and Scientific Computing ===
import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import rotate

# === JAX and JAXDF ===
import jax
from jax import jit
import jax.numpy as jnp
from jaxdf import FourierSeries  # Note: Same name as the one in jwave, keep only one based on usage

# === jwave Library ===
from jwave import FourierSeries  # If conflicting with jaxdf, rename as: `as JWFourierSeries`
from jwave.geometry import *     # Imports domain, medium, time axis, etc.
from jwave.geometry import Sources  # Optional, already included above via wildcard
from jwave.signal_processing import tone_burst

# === Custom Modules ===
from utils import simulate_wave_propagation  # Your custom wave propagation function

# === Image Processing ===
from skimage.draw import polygon, disk  # Used to generate masks for geometric shapes

def generate_random_sources_and_signals(domain, time_axis, tone_burst_enable, signal_amplitude, signal_num_cycles):
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
    num_cycles = signal_num_cycles
    amplitude = signal_amplitude

    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    sample_freq = 1.0 / time_axis.dt
    max_delay = 1.0 / signal_freq
    tone_len = num_cycles * int(sample_freq / signal_freq)

    base_wave = amplitude * jnp.sin(2 * jnp.pi * signal_freq * t[:tone_len])
    if tone_burst_enable:
        base_wave = amplitude * tone_burst(sample_freq, signal_freq, num_cycles)[:tone_len]
    
    if dim == 2:
        # === 2D Source Placement ===
        num_sources = 10
        spacing = 7
        fixed_x = 15
        center_y = domain.N[1] // 2
        start_y = center_y - ((num_sources - 1) // 2) * spacing - ((num_sources - 1) % 2) * spacing//2
        y_positions = tuple(start_y + i * spacing for i in range(num_sources))
        x_positions = (fixed_x,) * num_sources
        positions = (x_positions, y_positions)

    else:
        # === 3D Source Placement ===
        num_sources_x = 10
        num_sources_y = 10
        spacing = 7
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
    phase_list = [
        30, 15, 27, 5,  9,  9,  5, 27, 15, 30,
        15,  0, 14, 23, 28, 28, 23, 14,  0, 15,
        27, 14, 28,  7, 12, 12,  7, 28, 14, 27,
        5, 23,  7, 17, 23, 23, 17,  7, 23,  5,
        9, 28, 12, 23, 28, 28, 23, 12, 28,  9,
        9, 28, 12, 23, 28, 28, 23, 12, 28,  9,
        5, 23,  7, 17, 23, 23, 17,  7, 23,  5,
        27, 14, 28,  7, 12, 12,  7, 28, 14, 27,
        15,  0, 14, 23, 28, 28, 23, 14,  0, 15,
        30, 15, 27, 5,  9,  9,  5, 27, 15, 30,
    ]
    signals = []
    for i in range(num_sources_total):
        delay = max_delay * phase_list[i] / 32
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


if __name__ == "__main__":

    # === Simulation Parameters ===
    N               = 1             # Number of simulations to run
    simulation_time = 0.001         # Total duration of each simulation [s]
    dt              = 1e-8          # Time step interval for the j-Wave solver [s]
    material_file   = 'material_properties.csv'  # Material property table (CSV format)

    # === Domain Settings ===
    domain_shape    = (228, 228)    # Size of the simulation domain grid (2D or 3D)
    grid_spacing    = (1.4e-3, 1.4e-3)  # Physical spacing between grid points [m]
    margin          = 14            # Number of grid points to trim at each boundary (e.g., to remove PML)

    # === Signal Settings ===
    tone_burst_enable = True        # Enable tone burst excitation
    signal_amplitude  = 100         # Source signal amplitude
    signal_num_cycles = 3

    # === Output Settings ===
    start_idx       = 0             # Starting index for simulation numbering
    skip_steps      = 50            # Number of initial time steps to discard (transient stabilization)
    save_step       = 500           # Interval (in steps) for recording pressure field snapshots
    hdf5_filename   = "./custom.h5"  # Output HDF5 filename for full simulation data
    threshold       = 2e-2      # Pressure threshold to determine meaningful regions

    save_shape = tuple(dim - 2 * margin for dim in domain_shape)  # Domain shape after margin cropping
    num_saved_steps = int(round(simulation_time / dt / save_step)) - skip_steps  # Number of saved time frames

    # Dimensionality check between domain and grid spacing
    if len(domain_shape) != len(grid_spacing):
        raise ValueError("domain_shape and grid_spacing must have the same dimensions")

    # === Device Selection (GPU / CPU) ===
    devices = jax.devices()
    if any("cuda" in str(device).lower() for device in devices):
        print("Running on GPU (CUDA backend)")
    else:
        print("Running on CPU")

    # === Create HDF5 File and Datasets ===
    with h5py.File(hdf5_filename, "w") as f:
        # Pre-allocate HDF5 datasets for simulation outputs
        hdf5_pressure = f.create_dataset("pressure", shape=(N, num_saved_steps, *save_shape),
                                         dtype=np.float32, compression="gzip", compression_opts=9,
                                         chunks=(1, 1, *save_shape))

        hdf5_density = f.create_dataset("density", shape=(N, *save_shape),
                                        dtype=np.float32, compression="gzip", compression_opts=9,
                                        chunks=(1, *save_shape))

        hdf5_sound_speed = f.create_dataset("sound_speed", shape=(N, *save_shape),
                                            dtype=np.float32, compression="gzip", compression_opts=9,
                                            chunks=(1, *save_shape))

        # Load material property table as a dictionary
        df = pd.read_csv(material_file)
        material_properties = {
            row['Material']: {
                'density': row['Density (kg/m³)'],
                'sound_speed': row['Speed of Sound (m/s)']
            } for _, row in df.iterrows()
        }

        # === Main Simulation Loop ===
        for i in range(start_idx, start_idx + N):
            print(f"Running simulation {i+1}/{start_idx + N}...")
            start_time = time.time()

            # Initialize computational domain
            domain = Domain(domain_shape, grid_spacing)

            # Initialize homogeneous medium and embed heterogeneous regions
            density = np.ones(domain.N) * 1.38     # Background density [kg/m³]
            sound_speed = np.ones(domain.N) * 343  # Background sound speed [m/s]

            # Crop domain margins (remove PML or unused boundaries)
            slices = tuple(slice(margin, dim - margin) for dim in domain_shape)
            density_cropped = density[slices]
            sound_speed_cropped = sound_speed[slices]

            # Save cropped material properties to HDF5
            hdf5_density[i-start_idx] = density_cropped
            hdf5_sound_speed[i-start_idx] = sound_speed_cropped

            # Construct Fourier field representations for j-Wave
            density_fs = FourierSeries(np.expand_dims(density, -1), domain)
            sound_speed_fs = FourierSeries(np.expand_dims(sound_speed, -1), domain)
            medium = Medium(domain=domain, sound_speed=sound_speed_fs, density=density_fs)

            # Generate time axis and source excitation signals
            time_axis = TimeAxis(dt=float(dt), t_end=float(simulation_time))
            sources = generate_random_sources_and_signals(domain, time_axis, tone_burst_enable, signal_amplitude, signal_num_cycles)

            # Define JIT-compiled simulation function for j-Wave
            @jit
            def compiled_simulator():
                return simulate_wave_propagation(medium, time_axis, sources=sources, save_step=save_step)

            # Run j-Wave simulation
            simulation_start_time = time.time()
            pressure = compiled_simulator()
            simulation_end_time = time.time()
            print(f"j-Wave simulation time: {simulation_end_time - simulation_start_time:.2f} s")

            # === Post-processing ===
            # Remove redundant dimension (e.g., channel dimension)
            pressure = np.squeeze(pressure, axis=-1)

            # Check for NaN values to ensure numerical stability
            if np.any(np.isnan(pressure)):
                print(f"NaN detected in pressure at simulation {i+1}, skipping...")
                continue

            pressure = jnp.where(jnp.abs(pressure) < threshold, 1e-9, pressure)

            # Crop boundary and skip initial unstable steps
            crop_slices = tuple(slice(margin, dim - margin) for dim in domain_shape)
            full_slices = (slice(skip_steps, None),) + crop_slices
            pressure = pressure[full_slices]

            # Save final pressure field to HDF5
            hdf5_pressure[i-start_idx] = pressure
                            
            print(f"Simulation {i+1} saved, total time: {time.time() - start_time:.2f} s")
