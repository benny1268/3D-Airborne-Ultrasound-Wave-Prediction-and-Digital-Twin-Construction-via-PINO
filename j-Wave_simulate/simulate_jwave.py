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

def generate_random_sources_and_signals(domain, time_axis, tone_burst_enable, signal_amplitude):
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
    num_cycles = 3 #random.randint(1, 3)
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

    signals = []
    for i in range(num_sources_total):
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
    Nx, Ny = domain_size
    masks = np.zeros((num_shapes, Nx, Ny), dtype=int)
    densities, sound_speeds = [], []

    margin = 10

    for i in range(num_shapes):
        shape_type = random.choice(["circle", "triangle", "rectangle", "polygon"])
        center_x = random.randint(110, Nx - 40)
        center_y = random.randint(40, Ny - 40)
        radius = random.randint(40, 100)
        hollow = random.choice([False, True])
        hollow_thickness = random.randint(2, 4) if hollow else 0

        material = random.choice(list(material_properties.keys()))
        densities.append(material_properties[material]["density"])
        sound_speeds.append(material_properties[material]["sound_speed"])

        # Generate shape mask on temp canvas
        canvas = np.zeros((Nx, Ny), dtype=np.uint8)

        if shape_type == "circle":
            rr_outer, cc_outer = disk((center_x, center_y), radius//2)
            rr_inner, cc_inner = disk((center_x, center_y), radius//2 - hollow_thickness) if hollow else ([], [])

        elif shape_type == "triangle":
            outer = np.array([
                [center_x, center_y - radius//2],
                [center_x - radius//2, center_y + radius//2],
                [center_x + radius//2, center_y + radius//2]
            ])
            rr_outer, cc_outer = polygon(outer[:, 0], outer[:, 1])
            if hollow:
                inner = outer * 0.8 + np.array([center_x, center_y]) * 0.2
                rr_inner, cc_inner = polygon(inner[:, 0], inner[:, 1])
            else:
                rr_inner, cc_inner = [], []

        elif shape_type == "rectangle":
            rr_outer, cc_outer = polygon(
                [center_x - radius//2, center_x + radius//2, center_x + radius//2, center_x - radius//2],
                [center_y - radius//2, center_y - radius//2, center_y + radius//2, center_y + radius//2]
            )
            if hollow:
                rr_inner, cc_inner = polygon(
                    [center_x - radius//2 + hollow_thickness, center_x + radius//2 - hollow_thickness,
                     center_x + radius//2 - hollow_thickness, center_x - radius//2 + hollow_thickness],
                    [center_y - radius//2 + hollow_thickness, center_y - radius//2 + hollow_thickness,
                     center_y + radius//2 - hollow_thickness, center_y + radius//2 - hollow_thickness]
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

        else:
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

        # Clamp
        rr_outer = np.clip(rr_outer, margin, Nx - margin - 1)
        cc_outer = np.clip(cc_outer, margin, Ny - margin - 1)
        rr_inner = np.clip(rr_inner, margin, Nx - margin - 1)
        cc_inner = np.clip(cc_inner, margin, Ny - margin - 1)

        # Apply to canvas
        canvas[rr_outer, cc_outer] = 1
        if hollow:
            canvas[rr_inner, cc_inner] = 0

        # Rotate shape on canvas
        angle = random.uniform(-90, 90)
        rotated = rotate(canvas, angle=angle, reshape=False, order=1, mode='constant', cval=0)
        masks[i] = (rotated > 0.5).astype(np.uint8)

        # Clear protected signal area
        masks[i, :70, :] = 0

    return masks, densities, sound_speeds

def generate_shape_mask_3d(domain_size, num_shapes=1):
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

    Nx, Ny, Nz = domain_size
    masks = np.zeros((num_shapes, Nx, Ny, Nz), dtype=int)
    densities = []
    sound_speeds = []

    def sphere(center, radius, shape):
        x, y, z = np.indices(shape)
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        return np.where(mask)

    def cube(center, size, shape):
        half_x = size[0] // 2
        half_y = size[1] // 2
        half_z = size[2] // 2
        x_min, x_max = max(center[0] - half_x, 0), min(center[0] + half_x, shape[0])
        y_min, y_max = max(center[1] - half_y, 0), min(center[1] + half_y, shape[1])
        z_min, z_max = max(center[2] - half_z, 0), min(center[2] + half_z, shape[2])
        x, y, z = np.meshgrid(np.arange(x_min, x_max),
                              np.arange(y_min, y_max),
                              np.arange(z_min, z_max),
                              indexing='ij')
        return x.ravel(), y.ravel(), z.ravel()

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

    def cone(center, radius, height, shape):
        mask = np.zeros(shape, dtype=bool)
        cx, cy, cz = center
        for dz in range(height):
            r = radius * (dz / height)
            x_min = max(int(cx - r), 0)
            x_max = min(int(cx + r), shape[0])
            y_min = max(int(cy - r), 0)
            y_max = min(int(cy + r), shape[1])
            z = cz - dz
            if z < 0 or z >= shape[2]:
                continue
            x, y = np.ogrid[x_min:x_max, y_min:y_max]
            mask_layer = (x - cx)**2 + (y - cy)**2 <= r**2
            mask[x_min:x_max, y_min:y_max, z] = mask_layer
        return np.where(mask)

    def rotate_volume(mask3d, angles_deg, center=None, order=1):
        rotated = rotate(mask3d, angle=angles_deg[2], axes=(0, 1), reshape=False, order=order, mode='constant', cval=0)
        rotated = rotate(rotated, angle=angles_deg[1], axes=(0, 2), reshape=False, order=order, mode='constant', cval=0)
        rotated = rotate(rotated, angle=angles_deg[0], axes=(1, 2), reshape=False, order=order, mode='constant', cval=0)
        return (rotated > 0.5).astype(np.uint8)

    for i in range(num_shapes):
        shape_type = random.choice(['sphere', 'cube', 'ellipsoid', 'pyramid', 'cone'])
        hollow = random.random() < 0.5
        size = random.randint(40, 120)
        thickness = random.randint(3, 10) if hollow else 0

        center = (
            random.randint(40, Nx - 40),
            random.randint(40, Ny - 40),
            random.randint(100, Nz - 40)
        )

        material = random.choice(list(material_properties.keys()))
        densities.append(material_properties[material]["density"])
        sound_speeds.append(material_properties[material]["sound_speed"])

        # outer shape
        if shape_type == 'sphere':
            radius = size // 2
            rr, cc, zz = sphere(center, radius, domain_size)
        elif shape_type == 'cube':
            dims = [random.randint(30, size), random.randint(30, size), random.randint(30, size)]
            rr, cc, zz = cube(center, dims, domain_size)
        elif shape_type == 'ellipsoid':
            radii = [random.randint(20, size//2), random.randint(20, size//2), random.randint(20, size//2)]
            rr, cc, zz = ellipsoid(center, radii, domain_size)
        elif shape_type == 'pyramid':
            rr, cc, zz = pyramid(center, size, domain_size)
        elif shape_type == 'cone':
            rr, cc, zz = cone(center, size//2, size, domain_size)

        shape_mask = np.zeros(domain_size, dtype=np.uint8)
        shape_mask[rr, cc, zz] = 1

        # inner shape (if hollow)
        if hollow:
            if shape_type == 'sphere':
                inner_rr, inner_cc, inner_zz = sphere(center, radius - thickness, domain_size)
            elif shape_type == 'cube':
                inner_dims = [max(d - 2*thickness, 1) for d in dims]
                inner_rr, inner_cc, inner_zz = cube(center, inner_dims, domain_size)
            elif shape_type == 'ellipsoid':
                inner_radii = [max(r - thickness, 1) for r in radii]
                inner_rr, inner_cc, inner_zz = ellipsoid(center, inner_radii, domain_size)
            elif shape_type == 'pyramid':
                inner_rr, inner_cc, inner_zz = pyramid(center, size - 2*thickness, domain_size)
            elif shape_type == 'cone':
                inner_rr, inner_cc, inner_zz = cone(center, size//2 - thickness, size - 2*thickness, domain_size)
            shape_mask[inner_rr, inner_cc, inner_zz] = 0

        # apply rotation
        angles = (
            random.uniform(-90, 90),
            random.uniform(-90, 90),
            random.uniform(-90, 90)
        )
        shape_mask_rotated = rotate_volume(shape_mask, angles)

        shape_mask_rotated[:, :, :70] = 0
        masks[i] = shape_mask_rotated

    return masks, densities, sound_speeds

def save_hdf5(hdf5_filename, stacked_pressure, density_patch, sound_speed_patch, chunk_size=[64, 64, 128]):
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
        chunks_p = (1, min(H, chunk_size[0]), min(W, chunk_size[1]))
        chunks_s = (min(H, chunk_size[0]), min(W, chunk_size[1]))
    elif ndim == 4:
        T, H, W, L = shape
        chunks_p = (1, min(H, chunk_size[0]), min(W, chunk_size[1]), min(L, chunk_size[2]))
        chunks_s = (min(H, chunk_size[0]), min(W, chunk_size[1]), min(L, chunk_size[2]))
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
    N               = 10             # Number of simulations to run
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

    # === Output Settings ===
    start_idx       = 0             # Starting index for simulation numbering
    skip_steps      = 50            # Number of initial time steps to discard (transient stabilization)
    save_step       = 500           # Interval (in steps) for recording pressure field snapshots
    hdf5_filename   = "./training_data_2d_1026_hom_test.h5"  # Output HDF5 filename for full simulation data
    threshold           = 2e-2      # Pressure threshold to determine meaningful regions

    # === Patch Extraction Parameters ===
    save_train_data     = True      # Whether to extract and save patch-based data
    patch_output_dir    = "./training_data_patch_2d_1026"    # Directory for active-region patches
    nopatch_output_dir  = "./training_data_nopatch_2d_1026"  # Directory for inactive (low-pressure) patches
    window_size_x       = 100       # Patch window size along the X-axis
    window_size_y       = 100       # Patch window size along the Y-axis
    window_size_z       = 100       # Patch window size along the Z-axis (used for 3D domains)
    stride              = 50        # Stride (overlap) between adjacent patches

    # === Derived / Initialization Settings ===
    if save_train_data:
        os.makedirs(patch_output_dir, exist_ok=True)
        os.makedirs(nopatch_output_dir, exist_ok=True)
    
    save_shape = tuple(dim - 2 * margin for dim in domain_shape)  # Domain shape after margin cropping
    num_saved_steps = int(round(simulation_time / dt / save_step)) - skip_steps  # Number of saved time frames

    # Calculate total patch counts along each dimension
    num_patches_x = (save_shape[0] - window_size_x) // stride + 1
    num_patches_y = (save_shape[1] - window_size_y) // stride + 1
    num_patches_z = (save_shape[2] - window_size_z) // stride + 1 if len(domain_shape) == 3 else 1

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

            # Number of heterogeneous objects (reflectors or scatterers)
            num_shapes = 0  # random.randint(1, 3) can be used for random heterogeneity

            # Generate heterogeneous material regions
            if len(domain.N) == 2:
                shape_mask, densities, sound_speeds = generate_shape_mask_2d(domain.N, num_shapes)
            elif len(domain.N) == 3:
                shape_mask, densities, sound_speeds = generate_shape_mask_3d(domain.N, num_shapes)
            else:
                raise ValueError(f"Unsupported domain dimension: {len(domain.N)}D")

            # Initialize homogeneous medium and embed heterogeneous regions
            density = np.ones(domain.N) * 1.38     # Background density [kg/m³]
            sound_speed = np.ones(domain.N) * 343  # Background sound speed [m/s]
            for j in range(num_shapes):
                density[shape_mask[j] == 1] = densities[j]
                sound_speed[shape_mask[j] == 1] = sound_speeds[j]

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
            sources = generate_random_sources_and_signals(domain, time_axis, tone_burst_enable, signal_amplitude)

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

            # pressure = jnp.where(jnp.abs(pressure) < threshold, 1e-9, pressure)

            # Optionally mask heterogeneous regions as zero pressure (for clean visualization)
            for j in range(num_shapes):
                pressure = pressure.at[:, shape_mask[j] == 1].set(0)

            # Crop boundary and skip initial unstable steps
            crop_slices = tuple(slice(margin, dim - margin) for dim in domain_shape)
            full_slices = (slice(skip_steps, None),) + crop_slices
            pressure = pressure[full_slices]

            # Save final pressure field to HDF5
            hdf5_pressure[i-start_idx] = pressure

            # Patch slicing and saving (multithreaded)
            if save_train_data:
                with ThreadPoolExecutor(max_workers=14) as executor:
                    if len(domain.N) == 2:
                        for x in range(num_patches_x):
                            for y in range(num_patches_y):
                                x_start, y_start = x * stride, y * stride
                                density_patch = density_cropped[x_start:x_start + window_size_x, y_start:y_start + window_size_y]
                                sound_speed_patch = sound_speed_cropped[x_start:x_start + window_size_x, y_start:y_start + window_size_y]

                                stacked_pressure, patch_count = [], 0
                                for t in range(num_saved_steps):
                                    patch = pressure[t, x_start:x_start + window_size_x, y_start:y_start + window_size_y]

                                    if np.sum(np.abs(patch) > threshold) < 100:
                                        if patch_count > 1:
                                            executor.submit(save_hdf5, os.path.join(
                                                patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[window_size_x, window_size_y])
                                        stacked_pressure, patch_count = [], 0
                                    else:
                                        stacked_pressure.append(patch)
                                        patch_count += 1
                                if patch_count > 1:
                                    executor.submit(save_hdf5, os.path.join(
                                        patch_output_dir, f"n_{i}_x_{x}_y_{y}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                        np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[window_size_x, window_size_y])

                        density_patch = density_cropped
                        sound_speed_patch = sound_speed_cropped

                        stacked_pressure, patch_count = [], 0
                        for t in range(num_saved_steps):
                            patch = pressure[t]

                            if np.sum(np.abs(patch) > threshold) < 100:
                                if patch_count > 1:
                                    executor.submit(save_hdf5, os.path.join(
                                        nopatch_output_dir, f"n_{i}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                        np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[patch.shape[0], patch.shape[1]])
                                stacked_pressure, patch_count = [], 0
                            else:
                                stacked_pressure.append(patch)
                                patch_count += 1
                        if patch_count > 1:
                            executor.submit(save_hdf5, os.path.join(
                                nopatch_output_dir, f"n_{i}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[patch.shape[0], patch.shape[1]])

                    else:
                        for x in range(num_patches_x):
                            for y in range(num_patches_y):
                                for z in range(num_patches_z):
                                    x_start, y_start, z_start = x * stride, y * stride, z * stride
                                    density_patch = density_cropped[x_start:x_start + window_size_x,
                                                                    y_start:y_start + window_size_y,
                                                                    z_start:z_start + window_size_z]
                                    sound_speed_patch = sound_speed_cropped[x_start:x_start + window_size_x,
                                                                            y_start:y_start + window_size_y,
                                                                            z_start:z_start + window_size_z]
                                    
                                    stacked_pressure, patch_count = [], 0
                                    for t in range(num_saved_steps):
                                        patch = pressure[t, x_start:x_start + window_size_x,
                                                        y_start:y_start + window_size_y,
                                                        z_start:z_start + window_size_z]
                                        if np.sum(np.abs(patch) > threshold) < 500:
                                            if patch_count > 1:
                                                executor.submit(save_hdf5, os.path.join(
                                                    patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                                    np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[window_size_x, window_size_y, window_size_z])
                                            stacked_pressure, patch_count = [], 0
                                        else:
                                            stacked_pressure.append(patch)
                                            patch_count += 1
                                    if patch_count > 1:
                                        executor.submit(save_hdf5, os.path.join(
                                            patch_output_dir, f"n_{i}_x_{x}_y_{y}_z_{z}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                            np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[window_size_x, window_size_y, window_size_z])
                                        
                        stacked_pressure, patch_count = [], 0
                        for t in range(num_saved_steps):
                            patch = pressure[t]
                            if np.sum(np.abs(patch) > threshold) < 500:
                                if patch_count > 1:
                                    executor.submit(save_hdf5, os.path.join(
                                        nopatch_output_dir, f"n_{i}_t_{t - patch_count}__{patch_count:03d}.h5"),
                                        np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[patch.shape[0], patch.shape[1], patch.shape[2]])
                                stacked_pressure, patch_count = [], 0
                            else:
                                stacked_pressure.append(patch)
                                patch_count += 1
                        if patch_count > 1:
                            executor.submit(save_hdf5, os.path.join(
                                nopatch_output_dir, f"n_{i}_t_{num_saved_steps - patch_count}__{patch_count:03d}.h5"),
                                np.stack(stacked_pressure), density_patch, sound_speed_patch, chunk_size=[patch.shape[0], patch.shape[1], patch.shape[2]])
                            
            print(f"Simulation {i+1} saved, total time: {time.time() - start_time:.2f} s")
