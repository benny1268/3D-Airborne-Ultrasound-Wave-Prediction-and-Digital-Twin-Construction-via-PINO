import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import physicsnemo
import os
import time
from F_FNO import F_FNO

def predict_full_multstep(file_path, model, case_idx=0, mult_input=3, mult_output=10,
                          pressure_std=1.0, threshold=0.02, jwave_as_input=False, device="cuda"):
    """
    Predict the full spatial pressure field over time using a trained model.
    This version supports multi-step input and output (without patching).

    Args:
        file_path (str): Path to the HDF5 file containing simulation data.
        model (nn.Module): Trained model for prediction.
        case_idx (int): Index of the case to predict in the dataset.
        mult_input (int): Number of past time steps used as input.
        mult_output (int): Number of future steps predicted per forward pass.
        pressure_std (float): Standard deviation used for pressure normalization.
        threshold (float): Values below this threshold are set to a very small number.
        jwave_as_input (bool): If True, always use ground-truth (jwave) pressure as input.
                               If False, use the model's previous predictions (autoregressive).
        device (str): Device for inference ("cuda" or "cpu").

    Returns:
        torch.Tensor: Predicted full pressure field of shape (T, H, W) or (T, H, W, D).
    """

    # === Load initial pressure and static fields ===
    def read_input():
        density_min, density_max = 1.38, 5500
        sound_speed_min, sound_speed_max = 150, 5400

        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][case_idx][:].shape[0]
            pressure_in = torch.tensor(f["pressure"][case_idx][:mult_input], dtype=torch.float32)
            
            # Set values with abs < threshold to a very small number
            eps = 1e-9
            pressure_in[torch.abs(pressure_in) < threshold] = eps

            pressure_in = pressure_in / pressure_std

            density = torch.tensor(f["density"][case_idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][case_idx], dtype=torch.float32).unsqueeze(0)

            # Normalize static fields using min-max scaling
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            # Concatenate: pressure sequence + density fields + sound_speed fields
            invar = torch.cat([pressure_in, density, sound_speed], dim=0)

        return invar, total_steps

    # === Initialize inputs and result storage ===
    s0 = time.time()

    input_data, total_steps = read_input()
    input_data = input_data.to(device)
    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype, device=device)
    result_matrix[:mult_input] = input_data[:mult_input]

    s1 = time.time()
    print(f"Read input and initialized result tensor: {s1-s0:.3f} sec")

    # === Set model to evaluation mode ===
    model.eval()

    # === Autoregressive prediction loop ===
    for step in range(0, total_steps - mult_input, mult_output):
        # Prepare model input: choose jwave ground truth or previous predictions
        if jwave_as_input:
            with h5py.File(file_path, "r") as f:
                pressure_t = torch.tensor(f["pressure"][case_idx][step:step+mult_input], dtype=torch.float32, device=device)
                eps = 1e-9
                pressure_t[torch.abs(pressure_t) < threshold] = eps
                input_data[:mult_input] = pressure_t / pressure_std
        else:
            input_data[:mult_input] = result_matrix[step:step+mult_input].to(device)

        X = input_data.unsqueeze(0)  # Shape: (1, C, H, W) or (1, C, H, W, D)

        # Forward prediction
        with torch.no_grad():
            pressure_out = model(X).squeeze(0)  # Shape: (mult_output, H, W) or (mult_output, H, W, D)

        # Ensure predictions do not exceed the remaining time steps
        remaining_steps = total_steps - (step + mult_input)
        actual_pred_time = min(mult_output, remaining_steps)

        result_matrix[step + mult_input:step + mult_input + actual_pred_time] = pressure_out[:actual_pred_time]

    s2 = time.time()
    print(f"Autoregressive model prediction: {s2-s1:.3f} sec")

    # === Denormalize and return ===
    result_matrix = result_matrix * pressure_std

    return result_matrix.to(device)

def predict_patch_multstep(file_path, model, case_idx=0, mult_input=3, mult_output=7, block_size=100, slide_step=50, 
                           batch_size=4, sigma=128, pressure_std=1.0, threshold=0.02, jwave_as_input=False, device="cuda"):
    """
    Predict multi-step pressure fields using a sliding-window patch-based model.

    Args:
        file_path (str): Path to the input HDF5 file.
        model (nn.Module): Trained PhysicsNeMo model.
        case_idx (int): Index of the sample to predict from the HDF5 file.
        mult_input (int): Number of input time steps.
        mult_output (int): Number of steps to predict per forward pass.
        block_size (int): Size of each spatial patch.
        slide_step (int): Sliding stride between patches.
        batch_size (int): Batch size for model inference.
        sigma (float): Standard deviation for the Gaussian blending kernel used in patch recombination.
        pressure_std (float): Standard deviation for pressure normalization.
        threshold (float): Values below this threshold are set to a very small number.
        device (str): Device to run inference on ("cuda" or "cpu").
        jwave_as_input (bool): If True, always use ground-truth (jwave) pressure as input.
                               If False, use the model's previous predictions (autoregressive).

    Returns:
        torch.Tensor: Reconstructed full pressure field of shape (T, H, W) or (T, D, H, W).
    """

    # === Load initial pressure and material fields ===
    def read_input():
        density_min, density_max = 1.38, 5500
        sound_speed_min, sound_speed_max = 150, 5400

        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][case_idx].shape[0]

            # Normalize pressure input
            pressure_t = torch.tensor(f["pressure"][case_idx][:mult_input], dtype=torch.float32, device=device)

            # Set values with abs < threshold to a very small number
            eps = 1e-9
            pressure_t[torch.abs(pressure_t) < threshold] = eps

            pressure_t = pressure_t / pressure_std

            density = torch.tensor(f["density"][case_idx], dtype=torch.float32, device=device).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][case_idx], dtype=torch.float32, device=device).unsqueeze(0)

            # Normalize static fields using min-max scaling
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            # Concatenate: pressure sequence + density fields + sound_speed fields
            invar = torch.cat([pressure_t, density, sound_speed], dim=0)

        return invar, total_steps

    # === Extract overlapping patches using a sliding window ===
    def slide_and_split(input_data):
        ndim = input_data.ndim
        blocks = []

        if ndim == 3:  # (C, H, W)
            _, H, W = input_data.shape
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    blocks.append(input_data[:, i:i+block_size, j:j+block_size])
        elif ndim == 4:  # (C, D, H, W)
            _, H, W, D = input_data.shape
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    for d in range(0, D - block_size + 1, slide_step):
                        blocks.append(input_data[:, i:i+block_size, j:j+block_size, d:d+block_size])
        else:
            raise ValueError("Only 2D or 3D input tensors are supported.")

        return blocks

    # === Build a Gaussian blending kernel ===
    def gaussian_weight_tensor(dim, dtype=torch.float32):
        ax = torch.linspace(-(block_size // 2), block_size // 2, block_size, device=device, dtype=dtype)
        if dim == 2:
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        elif dim == 3:
            xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        else:
            raise ValueError("Only dim=2 or dim=3 is supported.")
        return kernel / torch.max(kernel)

    # === Recombine patches into the full field ===
    def combine_tensorwise(blocks, result_shape):
        dtype = blocks[0].dtype
        if len(result_shape) == 3:
            T, H, W = result_shape
            weight = gaussian_weight_tensor(dim=2, dtype=dtype)
            weight = weight.unsqueeze(0).repeat(mult_output, 1, 1)
            pred_full = torch.zeros((mult_output, H, W), device=device, dtype=dtype)
            count_full = torch.zeros_like(pred_full)
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    block = blocks.pop(0).to(device)
                    pred_full[:, i:i+block_size, j:j+block_size] += block * weight
                    count_full[:, i:i+block_size, j:j+block_size] += weight

        elif len(result_shape) == 4:
            T, H, W, D = result_shape
            weight = gaussian_weight_tensor(dim=3, dtype=dtype)
            weight = weight.unsqueeze(0).repeat(mult_output, 1, 1, 1)
            pred_full = torch.zeros((mult_output, H, W, D), device=device, dtype=dtype)
            count_full = torch.zeros_like(pred_full)
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    for d in range(0, D - block_size + 1, slide_step):
                        block = blocks.pop(0).to(device)
                        pred_full[:, i:i+block_size, j:j+block_size, d:d+block_size] += block * weight
                        count_full[:, i:i+block_size, j:j+block_size, d:d+block_size] += weight

        else:
            raise ValueError("Only 3D or 4D result shapes are supported.")

        return pred_full / torch.clamp(count_full, min=1e-8)

    # === Initialize inputs and result storage ===
    s0 = time.time()

    input_data, total_steps = read_input()
    input_data = input_data.to(device)
    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype, device=device)
    result_matrix[:mult_input] = input_data[:mult_input]

    s1 = time.time()
    print(f"Read input and initialized result tensor: {s1-s0:.3f} sec")

    # === Set model to evaluation mode ===
    model.eval()

    # === Autoregressive prediction loop ===
    for step in range(0, total_steps - mult_input, mult_output):
        preds = []

        # Prepare model input: choose jwave ground truth or previous predictions
        if jwave_as_input:
            with h5py.File(file_path, "r") as f:
                pressure_t = torch.tensor(f["pressure"][case_idx][step:step+mult_input], dtype=torch.float32, device=device)
                eps = 1e-9
                pressure_t[torch.abs(pressure_t) < threshold] = eps
                input_data[:mult_input] = pressure_t / pressure_std
        else:
            input_data[:mult_input] = result_matrix[step:step+mult_input].to(device)

        # Extract patches
        blocks = slide_and_split(input_data)
        batch_blocks = []

        # Forward prediction
        for idx, block in enumerate(blocks):
            # Skip near-zero regions (empty patches)
            if torch.all(torch.abs(block[0]) < (0.02 / pressure_std)):
                preds.append((idx, torch.full((mult_output,) + block.shape[1:], 1e-9, dtype=input_data.dtype)))
            else:
                batch_blocks.append((idx, block.unsqueeze(0)))
                if len(batch_blocks) == batch_size:
                    batch_indices, batch_inputs = zip(*batch_blocks)
                    batch_input = torch.cat(batch_inputs, dim=0).to(device)
                    with torch.no_grad():
                        batch_output = model(batch_input)
                    for out_idx, output in zip(batch_indices, batch_output):
                        preds.append((out_idx, output))
                    batch_blocks = []

        # Process any remaining blocks
        if batch_blocks:
            batch_indices, batch_inputs = zip(*batch_blocks)
            batch_input = torch.cat(batch_inputs, dim=0).to(device)
            with torch.no_grad():
                batch_output = model(batch_input)
            for out_idx, output in zip(batch_indices, batch_output):
                preds.append((out_idx, output))
        batch_blocks = []

        # Recombine patches into full field
        preds = [pred for _, pred in sorted(preds, key=lambda x: x[0])]
        pressure_out = combine_tensorwise(preds, result_matrix.shape)

        # Ensure predictions do not exceed the remaining time steps
        remaining_steps = total_steps - (step + mult_input)
        actual_pred_time = min(mult_output, remaining_steps)
        result_matrix[step + mult_input:step + mult_input + actual_pred_time] = pressure_out[:actual_pred_time]

    s2 = time.time()
    print(f"Autoregressive model prediction: {s2-s1:.3f} sec")

    # === Denormalize and return ===
    result_matrix = result_matrix * pressure_std

    return result_matrix.to(device)

def predict_post_process(predict_res, model, time_size=10, pressure_std=0.8, device="cuda"):
    """
    Post-Processing the full spatial pressure field over time.

    Args:
        file_path (str): Path to HDF5 file containing input fields.
        model (class): PhysicsNeMo model.
        time_size (int): Number of time steps to use.
        pressure_std (float): Std used for pressure normalization.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        np.ndarray: Full predicted pressure tensor of shape (T, H, W).
    """

    total_steps = predict_res.shape[0]
    result_matrix = predict_res / pressure_std   # remains on CPU


    # === Load the trained model ===
    model.eval()

    # === Autoregressive prediction loop ===
    for step in range(0, total_steps - time_size, time_size):
        # Gather mult_input steps as model input
        X = result_matrix[step:step + time_size]

        X = X.unsqueeze(0) 

        with torch.no_grad():
            pressure_out = model(X).squeeze(0)

        # Prevent overflow beyond result_matrix

        result_matrix[step:step + time_size] = pressure_out

    # === Denormalize and return as numpy array ===
    result_matrix = result_matrix * pressure_std

    return result_matrix.to(device)

def save_2D_comparison_gif(
        result_matrix,
        file_path,
        case_idx=0,
        threshold=0.02,
        total_steps=None,
        output_file="Local_PINO_result.gif",
        fps=10
    ):
    """
    Create a GIF animation comparing the predicted pressure field and ground truth over time.

    Args:
        result_matrix (np.ndarray):
            Predicted pressure field with shape (T, H, W).
        file_path (str):
            HDF5 file path containing the ground truth simulation results.
        case_idx (int):
            Case index to load from the HDF5 dataset.
        threshold (float):
            Pressure threshold. Values below this threshold are set to a very small number.
        total_steps (int or None):
            Number of simulation frames used in the animation.
            If None, it will automatically match the predicted sequence length.
        output_gif (str):
            Filename for saving the output GIF animation.
        fps (int):
            Frames per second for GIF playback.

    Returns:
        None – saves a GIF file to `output_gif`.
    """

    # === Load ground truth pressure field ===
    with h5py.File(file_path, "r") as f:
        pressure_field_gt = np.array(f['pressure'][case_idx][:])

    # Auto determine number of steps if not provided
    if total_steps is None:
        total_steps = min(result_matrix.shape[0], pressure_field_gt.shape[0])

    # Suppress values below threshold
    eps = 1e-9
    pressure_field_gt[np.abs(pressure_field_gt) < threshold] = eps

    # Infer spatial shape dynamically
    _, H, W = result_matrix.shape

    # === Setup the plotting layout ===
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title("Ground Truth (Pa)")
    ax[1].set_title("PINO (Pa)")
    ax[2].set_title("Error (%)")

    # === Initialize heatmaps based on real spatial resolution ===
    img_gt = ax[0].imshow(np.zeros((H, W)), cmap="seismic", animated=True)
    img_pred = ax[1].imshow(np.zeros((H, W)), cmap="seismic", animated=True)
    img_rlt_diff = ax[2].imshow(np.zeros((H, W)), cmap="hot", animated=True)

    cbar_gt = plt.colorbar(img_gt, ax=ax[0], shrink=0.5)
    cbar_pred = plt.colorbar(img_pred, ax=ax[1], shrink=0.5)
    cbar_rlt_diff = plt.colorbar(img_rlt_diff, ax=ax[2], shrink=0.5)

    # === Frame update logic ===
    def update_fig(step):
        abs_diff = np.abs(result_matrix[step] - pressure_field_gt[step])

        # Normalize the relative error percentage
        max_norm = max(np.max(np.abs(pressure_field_gt[step])), 0.2)
        rlt_diff = abs_diff / max_norm * 100

        img_gt.set_data(pressure_field_gt[step])
        img_pred.set_data(result_matrix[step])
        img_rlt_diff.set_data(rlt_diff)

        ax[0].set_title(f"Kwave (Pa) ● Step {step}")
        ax[1].set_title(f"PINO (Pa) ● Step {step}")
        ax[2].set_title(f"Relative Error (%) ● Step {step}")

        img_gt.set_clim(np.min(pressure_field_gt[step]), np.max(pressure_field_gt[step]))
        img_pred.set_clim(np.min(result_matrix[step]), np.max(result_matrix[step]))
        img_rlt_diff.set_clim(0, 10)

        # Update colorbars
        cbar_gt.mappable.set_clim(np.min(pressure_field_gt[step]), np.max(pressure_field_gt[step]))
        cbar_pred.mappable.set_clim(np.min(result_matrix[step]), np.max(result_matrix[step]))
        cbar_rlt_diff.mappable.set_clim(0, 10)

        return [img_gt, img_pred, img_rlt_diff]

    # === Create & save GIF ===
    ani = animation.FuncAnimation(fig, update_fig, frames=range(total_steps), interval=1000/fps, blit=True)
    ani.save(output_file, writer='pillow', fps=fps)

    plt.close()

def save_2D_comparison_frames(
        result_matrix,
        file_path,
        output_dir="comparison_frames",
        case_idx=0,
        threshold=0.02,
        total_steps=None,
        dpi=150
    ):
    """
    Save frame-by-frame comparison images (Ground Truth vs PINO vs Relative Error).

    Args:
        result_matrix (np.ndarray):
            Predicted pressure field with shape (T, H, W)
        file_path (str):
            File path to the HDF5 ground truth dataset
        output_dir (str):
            Directory to save the output images
        case_idx (int):
            Case index to load from HDF5 dataset
        threshold (float):
            Pressure threshold. Values below this threshold are set to a very small value.
        total_steps (int or None):
            Number of frames to save. If None, determined by smallest available sequence.
        dpi (int):
            Resolution of saved figures

    Returns:
        None – images will be saved to `output_dir`.
    """

    # Load ground truth
    with h5py.File(file_path, "r") as f:
        pressure_field_gt = np.array(f['pressure'][case_idx][:])

    # Determine number of steps
    if total_steps is None:
        total_steps = min(result_matrix.shape[0], pressure_field_gt.shape[0])

    # Threshold masking
    eps = 1e-9
    pressure_field_gt[np.abs(pressure_field_gt) < threshold] = eps

    # Shapes
    _, H, W = result_matrix.shape
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {total_steps} comparison frames to: {output_dir}")

    # Loop through frames
    for step in range(total_steps):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        abs_diff = np.abs(result_matrix[step] - pressure_field_gt[step])
        max_norm = max(np.max(np.abs(pressure_field_gt[step])), 0.2)
        rlt_diff = abs_diff / max_norm * 100

        # --- Plot GT ---
        im0 = ax[0].imshow(pressure_field_gt[step], cmap="seismic")
        ax[0].set_title(f"Ground Truth (Pa)\nStep {step}")
        plt.colorbar(im0, ax=ax[0], shrink=0.6)

        # --- Plot Prediction ---
        im1 = ax[1].imshow(result_matrix[step], cmap="seismic")
        ax[1].set_title(f"PINO (Pa)\nStep {step}")
        plt.colorbar(im1, ax=ax[1], shrink=0.6)

        # --- Relative Error ---
        im2 = ax[2].imshow(rlt_diff, cmap="hot", vmin=0, vmax=10)
        ax[2].set_title(f"Relative Error (%)\nStep {step}")
        plt.colorbar(im2, ax=ax[2], shrink=0.6)

        # --- Save Figure ---
        save_path = os.path.join(output_dir, f"comparison_step_{step:03d}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print("All frames saved successfully!")

def save_3D_comparison_panels(
    result_matrix,
    file_path,
    case_idx=0,
    total_steps=None,
    selected_steps=4,
    slice_x=None,   # 固定 X -> YZ
    slice_y=None,   # 固定 Y -> XZ
    slice_z=36,     # 固定 Z -> XY
    outdir="output_3d_views",
    cmap_field="seismic",
    cmap_err="magma",
    cmap_rho="Greys",
    cmap_c="plasma",
    threshold=0.02,
    start_step=0,
    end_step=100,
    dt_us=5,
):
    os.makedirs(outdir, exist_ok=True)

    # === 讀 HDF5 ===
    with h5py.File(file_path, "r") as f:
        gt_all = np.array(f["pressure"][case_idx][:])   # (T, X, Y, Z)
        density = np.array(f["density"][case_idx][:])   # (X, Y, Z)
        sound_speed = np.array(f["sound_speed"][case_idx][:])  # (X, Y, Z)

    # 小值抑制
    gt_all = gt_all.copy()
    gt_all[np.abs(gt_all) < threshold] = 1e-9

    # === 對齊步數 ===
    if total_steps is None:
        total_steps = min(len(gt_all), len(result_matrix))
    else:
        total_steps = min(total_steps, len(gt_all), len(result_matrix))

    gt   = gt_all[:total_steps]
    pred = result_matrix[:total_steps]
    assert gt.shape == pred.shape, f"GT {gt.shape} 與 Pred {pred.shape} 必須一致"

    # 軸順序 (T, X, Y, Z)
    _, X, Y, Z = gt.shape

    # 預設切片
    if slice_x is None: slice_x = X // 2
    if slice_y is None: slice_y = Y // 2
    if slice_z is None: slice_z = Z // 2

    # 均勻抽樣時間步
    steps = np.linspace(start_step, min(end_step, total_steps - 1), selected_steps, dtype=int)

    # 誤差分母
    denom_per_t = np.maximum(np.max(np.abs(gt), axis=(1, 2, 3)), 1e-9)

    # === 動態切片 (T, X, Y, Z) ===
    def get_slice_xy(a, t):   # 固定 Z
        return a[t, :, :, slice_z]            # (X, Y)

    def get_slice_xz(a, t):   # 固定 Y
        return np.rot90(a[t, :, slice_y, :], k=1)  # (X, Z) → 顯示時旋轉

    def get_slice_yz(a, t):   # 固定 X
        return np.rot90(a[t, slice_x, :, :], k=1)  # (Y, Z) → 顯示時旋轉

    # === 靜態切片 (X, Y, Z) ===
    def get_slice_xy_static(a):
        return a[:, :, slice_z]            # (X, Y)

    def get_slice_xz_static(a):
        return np.rot90(a[:, slice_y, :], k=1)  # (X, Z)

    def get_slice_yz_static(a):
        return np.rot90(a[slice_x, :, :], k=1)  # (Y, Z)

    views = [
        ("xy", get_slice_xy, get_slice_xy_static, 0.02),
        ("xz", get_slice_xz, get_slice_xz_static, 0.02),
        ("yz", get_slice_yz, get_slice_yz_static, 0.02),
    ]

    # === 原本的 3×N 面板 ===
    for view_name, slicer, slicer_static, wspace in views:
        fig, axes = plt.subplots(
            3, len(steps),
            figsize=(2.6 * len(steps), 7.8),
            constrained_layout=True,
            gridspec_kw={'wspace': wspace}
        )

        pred_slices, gt_slices, relerr_slices = [], [], []
        for t in steps:
            ps = slicer(pred, t)
            gs = slicer(gt, t)
            rel = np.abs(ps - gs) / denom_per_t[t] * 100
            pred_slices.append(ps)
            gt_slices.append(gs)
            relerr_slices.append(rel)

        for j, t in enumerate(steps):
            time_us = int(t * dt_us)

            im0 = axes[0, j].imshow(pred_slices[j], cmap=cmap_field)
            fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.01, aspect=30)
            axes[0, j].set_title(f"t={time_us} μs", fontsize=16, fontweight="bold")
            axes[0, j].axis("off")

            im1 = axes[1, j].imshow(gt_slices[j], cmap=cmap_field)
            fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.01, aspect=30)
            axes[1, j].axis("off")

            im2 = axes[2, j].imshow(relerr_slices[j], cmap=cmap_err)
            fig.colorbar(im2, ax=axes[2, j], fraction=0.046, pad=0.01, aspect=30)
            axes[2, j].axis("off")

        save_path = os.path.join(outdir, f"{view_name}_panels_3x{len(steps)}.png")
        fig.savefig(save_path, dpi=600)
        plt.close(fig)
        print(f"Saved: {save_path}")

    # === 額外輸出 density/sound_speed，各視角各一張 ===
    for view_name, _, slicer_static, _ in views:
        rho_slice = slicer_static(density)
        c_slice   = slicer_static(sound_speed)

        fig2, axes2 = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

        im_rho = axes2[0].imshow(rho_slice, cmap=cmap_rho)
        axes2[0].set_title(f"Density ({view_name})", fontsize=12, fontweight="bold")
        axes2[0].axis("off")
        fig2.colorbar(im_rho, ax=axes2[0], fraction=0.046, pad=0.02, aspect=30)

        im_c = axes2[1].imshow(c_slice, cmap=cmap_c)
        axes2[1].set_title(f"Sound Speed ({view_name})", fontsize=12, fontweight="bold")
        axes2[1].axis("off")
        fig2.colorbar(im_c, ax=axes2[1], fraction=0.046, pad=0.02, aspect=30)

        save_path2 = os.path.join(outdir, f"{view_name}_materials.png")
        fig2.savefig(save_path2, dpi=600)
        plt.close(fig2)
        print(f"Saved: {save_path2}")

def save_3D_steadyfield_comparison_panels(
    result_matrix,
    file_path,
    case_idx=0,
    slice_x=None,
    slice_y=None,
    slice_z=36,
    outdir="output_steady_state_views",
    cmap_field="hot",
    cmap_err="magma",
    pad_cb=0.06,
    dx=1.4,   # X/Y spacing (mm)
    dz=1.4,   # Z spacing (mm)
):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({
        "axes.titlesize": 14,   # 標題字體
        "axes.labelsize": 8,   # X/Y label 字體
        "xtick.labelsize": 7,  # x 軸刻度字體
        "ytick.labelsize": 7   # y 軸刻度字體
    })

    # === 讀 GT ===
    with h5py.File(file_path, "r") as f:
        gt_all = np.array(f["pressure"][case_idx][:,38:-38,38:-38,10:])   # (T, X, Y, Z)


    pred_all = np.array(result_matrix[:,38:-38,38:-38,10:])               # (T, X, Y, Z)
    # === 穩態場 (X, Y, Z) ===
    gt_steady   = np.max(np.abs(gt_all), axis=0)
    pred_steady = np.max(np.abs(pred_all), axis=0)
    start_z = 0


    # === 穩態場 (X, Y, Z) ===
    gt_steady   = np.max(np.abs(gt_all), axis=0)
    pred_steady = np.max(np.abs(pred_all), axis=0)

    save_path = "steady_state.h5"
    with h5py.File(save_path, "w") as f:
        f.create_dataset("gt_steady", data=gt_steady, compression="gzip")
        f.create_dataset("pred_steady", data=pred_steady, compression="gzip")


    # === 相對誤差 (%) ===
    denom = np.max(gt_steady)
    relerr = np.abs(pred_steady - gt_steady) * 100.0 / (denom + 1e-12)
    
    # === 尺度座標 ===
    X, Y, Z = gt_steady.shape
    x_axis = (np.arange(X) - (X - 1) / 2.0) * dx   # mm
    y_axis = (np.arange(Y) - (Y - 1) / 2.0) * dx   # mm
    z_axis = np.arange(Z) * dz + start_z*dz        # mm

    if slice_x is None: slice_x = X // 2
    if slice_y is None: slice_y = Y // 2
    if slice_z is None: slice_z = Z // 2

    # === 切片函式 ===
    def slice_XY(a):   # 固定 Z
        return a[:, :, slice_z]  # (X, Y)
    extent_XY = [y_axis[0], y_axis[-1], x_axis[0], x_axis[-1]]

    def slice_XZ(a):   # 固定 Y
        return a[:, slice_y, :]  # (X, Z)
    extent_XZ = [x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]]

    def slice_YZ(a):   # 固定 X
        return a[slice_x, :, :]  # (Y, Z)
    extent_YZ = [y_axis[0], y_axis[-1], z_axis[0], z_axis[-1]]

    views = [
        ("xy", slice_XY, extent_XY, "X (mm)", "Y (mm)"),
        ("xz", slice_XZ, extent_XZ, "X (mm)", "Z (mm)"),
        ("yz", slice_YZ, extent_YZ, "Y (mm)", "Z (mm)"),
    ]

    for view_name, slicer, extent, xlabel, ylabel in views:
        if view_name in ["xz", "yz"]:
            fig, axes = plt.subplots(
                1, 3, figsize=(9, 3),
                constrained_layout=True,
                gridspec_kw={'wspace': 0.01}
            )
        else:
            fig, axes = plt.subplots(
                1, 3, figsize=(9, 3),
                constrained_layout=True,
                gridspec_kw={'wspace': 0.01}
            )
        # PINO
        im0 = axes[0].imshow(slicer(pred_steady).T, cmap=cmap_field,
                             extent=extent, origin="lower", aspect="equal")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=pad_cb)
        axes[0].set_title("PINO")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        # j-Wave
        im1 = axes[1].imshow(slicer(gt_steady).T, cmap=cmap_field,
                             extent=extent, origin="lower", aspect="equal")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=pad_cb)
        axes[1].set_title("j-Wave")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        # Error
        im2 = axes[2].imshow(slicer(relerr).T, cmap=cmap_err,
                             extent=extent, origin="lower", aspect="equal")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=pad_cb)
        axes[2].set_title("Relative Error (%)")
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)

        save_path = os.path.join(outdir, f"steady_state_{view_name}.png")
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

# === Main Execution ===
if __name__ == "__main__":
    case_idx = 0
    file_path = "/mnt/ssd/guowei_R12921078/pino_project/Inference/Data/3D_hele/training_data_3d_0901.h5"
    model_path = "/mnt/ssd/guowei_R12921078/pino_project/Inference/model/Final/3D_hele/nopatch.pt"
    model_post_path = "./model/post_process/MgNO2D.pt"
    post_process = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Falling back to CPU.")

    model = None

    if model_path.endswith(".pt"):
        model = F_FNO(
            in_channels=5,
            out_channels=7,
            dimension=3,
            latent_channels=32,
            num_fno_layers=8,
            num_fno_modes=[75, 75, 100],
            padding=0,
            decoder_layers=1,
            decoder_layer_size=128,
        ).to(device)
        loadmodel_start_time = time.time()
        model.load_state_dict(torch.load(model_path, map_location=device))
        loadmodel_end_time = time.time()
        print(f"Load .pt model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

    else:
        # === 使用 physicsnemo Module.from_checkpoint 載入 modulus checkpoint ===
        loadmodel_start_time = time.time()
        model = physicsnemo.Module.from_checkpoint(model_path).to(device)
        loadmodel_end_time = time.time()
        print(f"Load .mdlus model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

    if post_process:
        if model_post_path.endswith(".pt"):
            model_post = F_FNO(
                in_channels=5,
                out_channels=10,
                dimension=3,
                latent_channels=32,
                num_fno_layers=8,
                num_fno_modes=[75, 75, 100],
                padding=0,
                decoder_layers=1,
                decoder_layer_size=128,
            ).to(device)
            loadmodel_start_time = time.time()
            model_post.load_state_dict(torch.load(model_post_path, map_location=device))
            loadmodel_end_time = time.time()
            print(f"Load post-processing model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

        else:
            loadmodel_start_time = time.time()
            model_post = physicsnemo.Module.from_checkpoint(model_post_path).to(device)
            loadmodel_end_time = time.time()
            print(f"Load post model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

    # === Build output path based on model name and data name ===
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    data_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("gif_plot", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_gif = os.path.join(output_dir, f"{data_name}_case{case_idx}.gif")
    
    # === Run prediction and create comparison visualization ===
    print(f"Processing Case {case_idx}...")

    start_time = time.time()
    result_matrix = predict_full_multstep(file_path, model, case_idx=case_idx, mult_input=3, mult_output=10,
                                          pressure_std=0.1, threshold=0.02, device=device, jwave_as_input=False)
    end_time = time.time()

    print(f"Compute time : {end_time - start_time} sec")

    if post_process:
        start_time = time.time()
        result_matrix = predict_post_process(predict_res=result_matrix, model=model_post, time_size=10, pressure_std=0.8, device=device)
        end_time = time.time()
        print(f"Post-Processing : {end_time - start_time} sec")

    result_matrix = result_matrix.cpu().numpy()


