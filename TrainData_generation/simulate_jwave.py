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

# === Image Processing ===
from skimage.draw import polygon, disk  # Used to generate masks for geometric shapes

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

def generate_shape_mask_2d(domain_size, num_shapes=1):
    """
    Generate binary masks for multiple randomly positioned heterogeneous regions.
    Supports various 2D shapes, with optional hollow interiors.

    Args:
        domain_size (tuple): Size of the 2D simulation domain (H, W).
        num_shapes (int): Number of shapes to generate.

    Returns:
        masks (np.ndarray): Array of shape (num_shapes, H, W), values in {0, 1}.
        densities (List[float]): Corresponding densities for each shape.
        sound_speeds (List[float]): Corresponding sound speeds for each shape.
    """
    H, W = domain_size
    masks = np.zeros((num_shapes, H, W), dtype=int)
    densities, sound_speeds = [], []

    margin = 10                    # Minimum margin from edges
    source_protection = (70, 276)  # Region to zero out for signal integrity

    for i in range(num_shapes):
        # === Random shape configuration ===
        shape_type = random.choice(["circle", "triangle", "rectangle", "polygon", "random"])
        center_x = random.randint(125, H - 35)
        center_y = random.randint(35, W - 35)
        radius = random.randint(15, 100)
        hollow = random.choice([False, True])
        hollow_thickness = random.randint(3, 10) if hollow else 0

        # === Random material selection ===
        material = random.choice(list(material_properties.keys()))
        densities.append(material_properties[material]["density"])
        sound_speeds.append(material_properties[material]["sound_speed"])

        # === Generate shape coordinates ===
        if shape_type == "circle":
            rr_outer, cc_outer = disk((center_x, center_y), radius)
            rr_inner, cc_inner = disk((center_x, center_y), radius - hollow_thickness) if hollow else ([], [])

        elif shape_type == "triangle":
            outer = np.array([
                [center_x, center_y - radius],
                [center_x - radius, center_y + radius],
                [center_x + radius, center_y + radius]
            ])
            rr_outer, cc_outer = polygon(outer[:, 0], outer[:, 1])
            if hollow:
                inner = outer * 0.8 + np.array([center_x, center_y]) * 0.2
                rr_inner, cc_inner = polygon(inner[:, 0], inner[:, 1])
            else:
                rr_inner, cc_inner = [], []

        elif shape_type == "rectangle":
            rr_outer, cc_outer = polygon(
                [center_x - radius, center_x + radius, center_x + radius, center_x - radius],
                [center_y - radius, center_y - radius, center_y + radius, center_y + radius]
            )
            if hollow:
                rr_inner, cc_inner = polygon(
                    [center_x - radius + hollow_thickness, center_x + radius - hollow_thickness,
                     center_x + radius - hollow_thickness, center_x - radius + hollow_thickness],
                    [center_y - radius + hollow_thickness, center_y - radius + hollow_thickness,
                     center_y + radius - hollow_thickness, center_y + radius - hollow_thickness]
                )
            else:
                rr_inner, cc_inner = [], []

        elif shape_type == "polygon":
            outer = np.array([
                [center_x, center_y - radius],
                [center_x - radius, center_y - radius // 2],
                [center_x - radius, center_y + radius // 2],
                [center_x, center_y + radius],
                [center_x + radius, center_y + radius // 2],
                [center_x + radius, center_y - radius // 2]
            ])
            rr_outer, cc_outer = polygon(outer[:, 0], outer[:, 1])
            if hollow:
                inner = outer * 0.8 + np.array([center_x, center_y]) * 0.2
                rr_inner, cc_inner = polygon(inner[:, 0], inner[:, 1])
            else:
                rr_inner, cc_inner = [], []

        else:  # random irregular polygon
            num_vertices = random.randint(3, 5)
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            r = np.random.uniform(15, 30, size=num_vertices)
            x = center_x + (r * np.cos(angles)).astype(int)
            y = center_y + (r * np.sin(angles)).astype(int)
            rr_outer, cc_outer = polygon(x, y)
            if hollow:
                x_inner = center_x + (r * 0.6 * np.cos(angles)).astype(int)
                y_inner = center_y + (r * 0.6 * np.sin(angles)).astype(int)
                rr_inner, cc_inner = polygon(x_inner, y_inner)
            else:
                rr_inner, cc_inner = [], []

        # === Clamp to domain ===
        rr_outer = np.clip(rr_outer, margin, H - margin)
        cc_outer = np.clip(cc_outer, margin, W - margin)
        rr_inner = np.clip(rr_inner, margin, H - margin)
        cc_inner = np.clip(cc_inner, margin, W - margin)

        # === Apply to mask ===
        masks[i, rr_outer, cc_outer] = 1
        if hollow:
            masks[i, rr_inner, cc_inner] = 0

        # === Clear protected signal area ===
        masks[i, :source_protection[0], :source_protection[1]] = 0

    return masks, densities, sound_speeds

def generate_shape_mask_3d(domain_size, num_shapes=1):
    """
    Generate a 3D binary mask with randomly placed heterogeneous shapes.
    Each shape is assigned a random density and sound speed.

    Args:
        domain_size (tuple): Size of the 3D simulation grid (Nx, Ny, Nz).
        num_shapes (int): Number of heterogeneous objects to generate.

    Returns:
        masks (np.ndarray): Shape (num_shapes, Nx, Ny, Nz), binary masks for each object.
        densities (List[float]): Corresponding densities for each shape.
        sound_speeds (List[float]): Corresponding sound speeds for each shape.
    """
    Nx, Ny, Nz = domain_size
    masks = np.zeros((num_shapes, Nx, Ny, Nz), dtype=int)
    densities = []
    sound_speeds = []

    def sphere(center, radius, shape):
        x, y, z = np.indices(shape)
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        return np.where(mask)

    def hollow_sphere(center, radius, thickness, shape):
        x, y, z = np.indices(shape)
        dist_sq = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        outer = dist_sq <= radius**2
        inner = dist_sq <= (radius - thickness)**2
        return np.where(outer & ~inner)

    def cube(center, size, shape):
        half = size // 2
        x_min, x_max = max(center[0] - half, 0), min(center[0] + half, shape[0])
        y_min, y_max = max(center[1] - half, 0), min(center[1] + half, shape[1])
        z_min, z_max = max(center[2] - half, 0), min(center[2] + half, shape[2])
        x, y, z = np.meshgrid(np.arange(x_min, x_max),
                              np.arange(y_min, y_max),
                              np.arange(z_min, z_max),
                              indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

    def hollow_cube(center, size, thickness, shape):
        mask = np.zeros(shape, dtype=bool)
        rr, cc, zz = cube(center, size, shape)
        mask[rr, cc, zz] = 1
        rr_inner, cc_inner, zz_inner = cube(center, size - 2 * thickness, shape)
        mask[rr_inner, cc_inner, zz_inner] = 0
        return np.where(mask)

    def ellipsoid(center, radii, shape):
        x, y, z = np.indices(shape)
        norm = (((x - center[0]) / radii[0]) ** 2 +
                ((y - center[1]) / radii[1]) ** 2 +
                ((z - center[2]) / radii[2]) ** 2)
        return np.where(norm <= 1)

    def pyramid(center, size, shape):
        mask = np.zeros(shape, dtype=bool)
        cx, cy, cz = center
        half = size // 2
        for dz in range(size):
            scale = (size - dz) / size
            x_min = max(int(cx - half * scale), 0)
            x_max = min(int(cx + half * scale), shape[0])
            y_min = max(int(cy - half * scale), 0)
            y_max = min(int(cy + half * scale), shape[1])
            z = cz - dz
            if z < 0 or z >= shape[2]:
                continue
            mask[x_min:x_max, y_min:y_max, z] = 1
        return np.where(mask)

    for i in range(num_shapes):
        shape_type = random.choice([
            'sphere', 'hollow_sphere', 'cube', 'hollow_cube',
            'ellipsoid', 'pyramid'
        ])
        hollow = 'hollow' in shape_type
        size = random.randint(10, 40)
        thickness = random.randint(2, 8) if hollow else 0

        center = (
            random.randint(size, Nx - size),
            random.randint(size, Ny - size),
            random.randint(size, Nz - size)
        )

        material = random.choice(list(material_properties.keys()))
        densities.append(material_properties[material]["density"])
        sound_speeds.append(material_properties[material]["sound_speed"])

        if shape_type == 'sphere':
            rr, cc, zz = sphere(center, size, domain_size)
        elif shape_type == 'hollow_sphere':
            rr, cc, zz = hollow_sphere(center, size, thickness, domain_size)
        elif shape_type == 'cube':
            rr, cc, zz = cube(center, size, domain_size)
        elif shape_type == 'hollow_cube':
            rr, cc, zz = hollow_cube(center, size, thickness, domain_size)
        elif shape_type == 'ellipsoid':
            radii = (size, size // 2, size // 3)
            rr, cc, zz = ellipsoid(center, radii, domain_size)
        elif shape_type == 'pyramid':
            rr, cc, zz = pyramid(center, size, domain_size)

        masks[i, rr, cc, zz] = 1
        masks[i, :, :, :70] = 0

    return masks, densities, sound_speeds

def save_hdf5(hdf5_filename, stacked_pressure, density_patch, sound_speed_patch, chunk_size=64):
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
        chunks_s = (min(H, chunk_size), min(W, chunk_size))
    elif ndim == 4:
        T, H, W, L = shape
        chunks_p = (1, min(H, chunk_size), min(W, chunk_size), min(L, chunk_size))
        chunks_s = (min(H, chunk_size), min(W, chunk_size), min(L, chunk_size))
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
        f.create_dataset(
            "density",
            data=density_patch,
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
            chunks=chunks_s,
        )
        f.create_dataset(
            "sound_speed",
            data=sound_speed_patch,
            dtype=np.float32,
            compression="gzip",
            compression_opts=9,
            chunks=chunks_s,
        )

if __name__ == "__main__":

    # === Simulation Parameters ===
    N               = 10              # Number of simulations to run
    simulation_time = 0.001         # Duration of each simulation in seconds
    dt              = 1e-8           # Time step interval for j-Wave
    save_step       = 500            # Interval in time steps for recording pressure field
    material_file   = 'material_properties.csv'  # Material properties file (CSV)
    start_idx       = 0              # Starting index for simulation numbering

    # === Domain Settings ===
    domain_shape    = (276, 276)     # Shape of the simulation grid (2D or 3D)
    grid_spacing    = (1e-3, 1e-3)   # Grid spacing in meters for each axis
    margin          = 10             # Number of grid points to trim at each boundary (e.g., to remove PML)

    # === Patch Extraction Parameters ===
    window_size     = 64             # Size of each spatial patch
    stride          = 32             # Step size when sliding the window over the spatial domain
    threshold       = 0.01           # Threshold to determine if a patch contains meaningful data

    # === Output Settings ===
    skip_steps      = 30             # Number of initial time steps to ignore (before pressure field stabilizes)
    save_patch      = True           # Whether to save patch data
    patch_output_dir = "./training_data_patch_2d_0703"
    hdf5_filename    = "./training_data_2d_0703.h5"

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

        dset_density = f.create_dataset("density", shape=(N, *save_shape),
                                        dtype=np.float32, compression="gzip", compression_opts=9,
                                        chunks=(1, *save_shape))

        dset_sound_speed = f.create_dataset("sound_speed", shape=(N, *save_shape),
                                            dtype=np.float32, compression="gzip", compression_opts=9,
                                            chunks=(1, *save_shape))

        # Load material property table into dictionary
        df = pd.read_csv(material_file)
        material_properties = {
            row['Material']: {
                'density': row['Density (kg/m³)'],
                'sound_speed': row['Speed of Sound (m/s)']
            } for _, row in df.iterrows()
        }

        for i in range(start_idx, start_idx + N):
            print(f"Running simulation {i+1}/{start_idx + N}...")
            start_time = time.time()

            domain = Domain(domain_shape, grid_spacing)
            num_shapes = 1

            if len(domain.N) == 2:
                shape_mask, densities, sound_speeds = generate_shape_mask_2d(domain.N, num_shapes)
            elif len(domain.N) == 3:
                shape_mask, densities, sound_speeds = generate_shape_mask_3d(domain.N, num_shapes)
            else:
                raise ValueError(f"Unsupported domain dimension: {len(domain.N)}D")

            # Initialize homogeneous field and embed heterogeneous regions
            density = np.ones(domain.N) * 1.38
            sound_speed = np.ones(domain.N) * 343
            for j in range(num_shapes):
                density[shape_mask[j] == 1] = densities[j]
                sound_speed[shape_mask[j] == 1] = sound_speeds[j]

            # Crop margins
            slices = tuple(slice(margin, dim - margin) for dim in domain_shape)
            density_cropped = density[slices]
            sound_speed_cropped = sound_speed[slices]

            dset_density[i-start_idx] = density[slices]
            dset_sound_speed[i-start_idx] = sound_speed[slices]

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

            # Mask heterogeneous zones as zero pressure
            for j in range(num_shapes):
                pressure = pressure.at[:, shape_mask[j] == 1].set(0)

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
                                density_patch = density_cropped[x_start:x_start + window_size, y_start:y_start + window_size]
                                sound_speed_patch = sound_speed_cropped[x_start:x_start + window_size, y_start:y_start + window_size]

                                stacked_pressure, patch_count = [], 0
                                for t in range(num_saved_steps):
                                    patch = pressure[t, x_start:x_start + window_size, y_start:y_start + window_size]
                                    if np.sum(np.abs(patch) > threshold) < 1000:
                                        if patch_count > 1:
                                            executor.submit(save_hdf5, os.path.join(
                                                patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=window_size)
                                        stacked_pressure, patch_count = [], 0
                                    else:
                                        stacked_pressure.append(patch)
                                        patch_count += 1
                                if patch_count > 1:
                                    executor.submit(save_hdf5, os.path.join(
                                        patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                        np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=window_size)

                    else:
                        for x in range(num_patches_x):
                            for y in range(num_patches_y):
                                for z in range(num_patches_z):
                                    x_start, y_start, z_start = x * stride, y * stride, z * stride
                                    density_patch = density_cropped[x_start:x_start + window_size,
                                                                    y_start:y_start + window_size,
                                                                    z_start:z_start + window_size]
                                    sound_speed_patch = sound_speed_cropped[x_start:x_start + window_size,
                                                                            y_start:y_start + window_size,
                                                                            z_start:z_start + window_size]
                                    
                                    stacked_pressure, patch_count = [], 0
                                    for t in range(num_saved_steps):
                                        patch = pressure[t, x_start:x_start + window_size,
                                                        y_start:y_start + window_size,
                                                        z_start:z_start + window_size]
                                        if np.sum(np.abs(patch) > threshold) < 3000:
                                            if patch_count > 1:
                                                executor.submit(save_hdf5, os.path.join(
                                                    patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                    np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=window_size)
                                            stacked_pressure, patch_count = [], 0
                                        else:
                                            stacked_pressure.append(patch)
                                            patch_count += 1
                                    if patch_count > 1:
                                        executor.submit(save_hdf5, os.path.join(
                                            patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                            np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=window_size)

            print(f"Simulation {i+1} saved, total time: {time.time() - start_time:.2f} s")
