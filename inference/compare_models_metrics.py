import os
import time
import math

import h5py
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

import physicsnemo

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

def compute_relative_l2_errors_per_timestep(y_pred, y_true, total_steps, device="cpu", eps=1e-8):
    """
    Compute Relative L2 Error per time step:
        Relative L2 Error = ||y_pred - y_true|| / (||y_true|| + eps)
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)
    
    error_list = []
    for t in range(total_steps):
        numerator = torch.norm(y_pred[t] - y_true[t], p=2)
        denominator = torch.norm(y_true[t], p=2) + eps
        error = (numerator / denominator).item()
        error_list.append(error)

    return error_list

def compute_relative_l2_errors_ROI_per_timestep(y_pred, y_true, total_steps, device="cpu", eps=1e-8):
    """
    Compute Relative L2 Error per time step,
    but only for Region of Interest.
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    max_val = torch.max(torch.abs(y_true[0]))

    error_list = []
    for t in range(total_steps):
        # Create a mask that selects only elements greater than the mean
        mask = torch.abs(y_true[t]) > max_val * 0.1
        
        # Apply the mask to filter both predicted and ground truth tensors
        pred_masked = y_pred[t][mask]
        true_masked = y_true[t][mask]
        
        # If all elements are filtered out, append NaN to indicate no valid points
        if mask.sum() == 0:
            error_list.append(float('nan'))
            continue
        
        # Compute L2 norm of the difference in the masked region
        numerator = torch.norm(pred_masked - true_masked, p=2)
        
        # Compute L2 norm of the ground truth in the masked region (add eps to avoid division by zero)
        denominator = torch.norm(true_masked, p=2) + eps
        
        # Relative L2 Error for the masked region
        error = (numerator / denominator).item()
        error_list.append(error)

    return error_list

def compute_fluctuation_correlation_per_timestep(y_pred, y_true, total_steps, device="cpu"):
    """
    Compute fluctuation correlation (Pearson correlation over fluctuations):
        numerator = Σ((y_pred - mean) * (y_true - mean))
        denominator = sqrt(Σ(y_pred_fluc²) * Σ(y_true_fluc²))
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    corr_list = []
    for t in range(total_steps):
        pred_fluc = y_pred[t] - torch.mean(y_pred[t])
        true_fluc = y_true[t] - torch.mean(y_true[t])

        numerator = torch.sum(pred_fluc * true_fluc)
        denominator = torch.sqrt(torch.sum(pred_fluc ** 2) * torch.sum(true_fluc ** 2))

        if denominator.item() == 0:
            corr = 0.0
        else:
            corr = (numerator / denominator).item()
        corr_list.append(corr)
    return corr_list

def compute_global_relative_l2_error(y_pred, y_true, device="cpu", eps=1e-8):
    """
    Compute global Relative L2 Error:
        ||y_pred - y_true|| / (||y_true|| + eps)
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    numerator = torch.norm(y_pred - y_true, p=2)
    denominator = torch.norm(y_true, p=2) + eps
    return (numerator / denominator).item()

def compute_global_relative_l2_error_ROI(y_pred, y_true, device="cpu", eps=1e-8):
    """
    Compute global Relative L2 Error, but only for Region of Interest
    
    Formula:
        Relative L2 Error = ||y_pred(masked) - y_true(masked)|| / (||y_true(masked)|| + eps)
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # Compute mean of all ground truth values
    max_val = torch.max(torch.abs(y_true))

    # Create mask for positions greater than mean
    mask = torch.abs(y_true) > max_val*0.1

    # Apply mask to filter predicted and ground truth tensors
    pred_masked = y_pred[mask]
    true_masked = y_true[mask]

    # If mask is empty, return NaN
    if mask.sum() == 0:
        return float('nan')

    # Compute L2 norms in the masked region
    numerator = torch.norm(pred_masked - true_masked, p=2)
    denominator = torch.norm(true_masked, p=2) + eps

    return (numerator / denominator).item()

def compute_global_fluctuation_correlation(y_pred, y_true, device="cpu", eps=1e-8):
    """
    Compute global fluctuation correlation (Pearson correlation of centered fields):
        numerator = Σ((y_pred - mean) * (y_true - mean))
        denominator = sqrt(Σ(y_pred_fluc²) * Σ(y_true_fluc²))
    """
    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    pred_fluc = y_pred - torch.mean(y_pred)
    true_fluc = y_true - torch.mean(y_true)

    numerator = torch.sum(pred_fluc * true_fluc)
    denominator = torch.sqrt(torch.sum(pred_fluc ** 2) * torch.sum(true_fluc ** 2)) + eps

    corr = (numerator / denominator).item()
    return corr

def process_dataset(file_path, model, model_post, model_name, case_nums=10, mult_input=3, mult_output=10, sigma=128, 
                    pressure_std=1.0, pred_time=500, threshold=0.02, device="cuda", jwave_as_input=False, post_process=False):
        
    metric_results = {
        "Relative L2 Error": [],
        "Relative L2 Error(Region of Interest)": [],
        "Fluctuation Correlation": [],
    }
    global_l2_list = []
    global_l2_above_mean_list = []
    global_fc_list = []

    for case_idx in range(case_nums):
        #print(f"\nProcessing Case {case_idx}...")
        name = model_name.lower()

        if "no-patch" in name:
            pressure_field_pred = predict_full_multstep(file_path=file_path, model=model, case_idx=case_idx, mult_input=mult_input, mult_output=mult_output,
                                                pressure_std=pressure_std, threshold=threshold, jwave_as_input=jwave_as_input, device=device)
        elif "patch" in name:
            pressure_field_pred = predict_patch_multstep(file_path=file_path, model=model, case_idx=case_idx, mult_input=mult_input, mult_output=mult_output,
                                                sigma=sigma, pressure_std=pressure_std, threshold=threshold, jwave_as_input=jwave_as_input, device=device)
        else:
            pressure_field_pred = predict_full_multstep(file_path=file_path, model=model, case_idx=case_idx, mult_input=mult_input, mult_output=mult_output,
                                                pressure_std=pressure_std, threshold=threshold, jwave_as_input=jwave_as_input, device=device)

        if post_process:
            start_time = time.time()
            pressure_field_pred = predict_post_process(predict_res=pressure_field_pred, model=model_post, time_size=10, pressure_std=pressure_std, device=device)
            end_time = time.time()
            print(f"Post-Processing : {end_time - start_time} sec")

        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][case_idx][:].shape[0]
            pressure_field_gt = torch.tensor(f["pressure"][case_idx][:total_steps], dtype=torch.float32, device=device)
            eps = 1e-9
            pressure_field_gt[torch.abs(pressure_field_gt) < threshold] = eps

        pred_time = min(pred_time, total_steps)
        pressure_field_gt = pressure_field_gt[:pred_time]
        pressure_field_pred = pressure_field_pred[:pred_time]
        metric_results["Relative L2 Error"].append(compute_relative_l2_errors_per_timestep(pressure_field_pred, pressure_field_gt, pred_time, device))
        metric_results["Relative L2 Error(Region of Interest)"].append(compute_relative_l2_errors_ROI_per_timestep(pressure_field_pred, pressure_field_gt, pred_time, device))
        metric_results["Fluctuation Correlation"].append(compute_fluctuation_correlation_per_timestep(pressure_field_pred, pressure_field_gt, pred_time, device))

        global_l2 = compute_global_relative_l2_error(pressure_field_pred, pressure_field_gt, device=device)
        global_l2_above_mean = compute_global_relative_l2_error_ROI(pressure_field_pred, pressure_field_gt, device=device)
        global_fc = compute_global_fluctuation_correlation(pressure_field_pred, pressure_field_gt, device=device)

        # print(f"[Global L2-Rel] : {global_l2:.4f}")
        # print(f"[Global L2-Rel(Region of Interest)] : {global_l2_above_mean:.4f}")
        # print(f"[Global FC] : {global_fc:.4f}")

        global_l2_list.append(global_l2)
        global_l2_above_mean_list.append(global_l2_above_mean)
        global_fc_list.append(global_fc)
        
    print("\n=== Global Metric Summary ===")
    print(f"Global FC         → {np.mean(global_fc_list):.4f} ± {np.std(global_fc_list):.4f} (min: {np.min(global_fc_list):.4f}, max: {np.max(global_fc_list):.4f})")
    print(f"Global L2-Rel     → {np.mean(global_l2_list):.4f} ± {np.std(global_l2_list):.4f} (min: {np.min(global_l2_list):.4f}, max: {np.max(global_l2_list):.4f})")
    print(f"Global L2-Rel(Region of Interest)  → {np.mean(global_l2_above_mean_list):.4f} ± {np.std(global_l2_above_mean_list):.4f} (min: {np.min(global_l2_above_mean_list):.4f}, max: {np.max(global_l2_above_mean_list):.4f})")

    return metric_results

def _set_times_or_fallback():
    tnr_files = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    ]
    picked = None
    for p in tnr_files:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)  # 註冊字型檔
                picked = "Times New Roman" # Matplotlib 使用家族名
                break
            except Exception:
                pass

    if picked is None:
        family = ["DejaVu Serif", "Liberation Serif", "serif"]
    else:
        family = [picked]

    mpl.rcParams.update({
        "font.family": family,
        "mathtext.fontset": "stix",   # 數學字型更接近期刊 Times 風格
    })

def plot_multi_models_error(
    data_all,
    ylabel,
    save_path,
    metrics,
    ylim=None,
    figsize=(3.4, 2.4),
    markevery=10,
):
    _set_times_or_fallback()

    rc = {
        "figure.figsize": figsize,
        "font.family": "DejaVu Sans",   # ✅ 確保有粗體款的字型
        "font.size": 8,
        "font.weight": "bold",          # 全域粗體（多數文字）
        "axes.labelsize": 8,
        "axes.labelweight": "bold",     # 座標軸標籤粗體
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "mathtext.fontset": "stix",     # 數學字元（μ）用 STIX
        "pdf.fonttype": 42,             # ✅ PDF 內嵌 TrueType，粗體更明顯
        "ps.fonttype": 42,              # ✅ 同上（EPS/PS）
        "text.usetex": False,           # ✅ 避免需 LaTeX 才粗體
    }

    with mpl.rc_context(rc):
        fig, ax = plt.subplots()

        colors = plt.get_cmap("tab10").colors
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

        color_map, marker_map = {}, {}
        idx = 0
        for model_name in data_all.keys():
            base_key = model_name.split(" - ")[-1]
            if base_key not in color_map:
                color_map[base_key]  = colors[idx % len(colors)]
                marker_map[base_key] = markers[idx % len(markers)]
                idx += 1

        step_handles, step_labels = [], []
        seen_step = set()

        for model_name, data in data_all.items():
            base_key = model_name.split(" - ")[-1]
            color  = color_map[base_key]
            marker = marker_map[base_key]

            for metric in metrics:
                metric_mean = np.mean(data[metric], axis=0)
                steps_us = np.arange(len(metric_mean[4:])) * 5   # μs
                y = metric_mean[4:]

                ax.plot(
                    steps_us, y,
                    color=color,
                    linestyle='-',
                    marker=marker,
                    markersize=2.5,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.5,
                    markevery=markevery,
                )

            if base_key not in seen_step:
                step_handles.append(
                    Line2D([0], [0], color=color, linestyle='-',
                           marker=marker, markersize=2.5,
                           markerfacecolor=color, markeredgecolor=color,
                           markeredgewidth=0.5, lw=1)
                )
                step_labels.append(base_key)
                seen_step.add(base_key)

        # ---- 座標/外觀（直接指定粗體）----
        ax.set_xlabel(r"Time ($\boldsymbol{\mu}$s)", fontweight="bold")  # ✅ 'Time' 及 μ 都更粗
        ax.set_ylabel(ylabel, fontweight="bold")                          # ✅ y 標籤粗體
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.3)

        # 刻度字串粗體
        ax.tick_params(axis="both", which="major", labelsize=7, width=0.8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        # ---- 圖例（用 prop 指定粗體，rc 不支援 legend.fontweight）----
        max_per_row = 3
        ncol = min(max_per_row, len(step_handles))
        nrow = math.ceil(len(step_handles) / max_per_row)

        fig.legend(step_handles, step_labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.06),
                   ncol=ncol, frameon=False,
                   columnspacing=1.2, handlelength=2.2,
                   prop={"weight": "bold", "family": "DejaVu Sans"})

        fig.subplots_adjust(bottom=0.20 + 0.05 * (nrow - 1))

        # ---- 存圖：同時存 PNG 看肉眼是否更明顯 ----
        root, ext = os.path.splitext(save_path)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
        if ext.lower() != ".pdf":
            fig.savefig(root + ".pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(root + ".png", bbox_inches="tight", pad_inches=0.02, dpi=600)  # ✅ 方便檢查粗體效果
        plt.close(fig)

def plot_l2rel_fc_side_by_side(data_all, save_path, ylim_l2=None, ylim_fc=None):
    _set_times_or_fallback()

    rc = {
        "figure.figsize": (6.8, 2.4),   # 單欄寬度 × 2
        "font.family": "DejaVu Sans",   # ✅ 確保有粗體款
        "font.size": 8,
        "font.weight": "bold",          # 全域粗體
        "axes.labelsize": 8,
        "axes.labelweight": "bold",     # 座標軸標籤粗體
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,             # ✅ TrueType 內嵌，粗體更清楚
        "ps.fonttype": 42,
    }

    with mpl.rc_context(rc):
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)

        colors = plt.get_cmap("tab10").colors
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

        color_map, marker_map = {}, {}
        idx = 0
        for model_name in data_all.keys():
            base_key = model_name.split(" - ")[-1]
            if base_key not in color_map:
                color_map[base_key]  = colors[idx % len(colors)]
                marker_map[base_key] = markers[idx % len(markers)]
                idx += 1

        step_handles, step_labels = [], []
        seen_step = set()

        # === 左圖：L2-Rel ===
        ax = axes[0]
        for model_name, data in data_all.items():
            base_key = model_name.split(" - ")[-1]
            color  = color_map[base_key]
            marker = marker_map[base_key]

            metric_mean = np.mean(data["Relative L2 Error"], axis=0)
            steps_us = np.arange(len(metric_mean[4:])) * 5
            y = metric_mean[4:]

            ax.plot(steps_us, y, color=color, linestyle='-',
                    marker=marker, markersize=2.5,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.5, markevery=10)

            if base_key not in seen_step:
                step_handles.append(
                    Line2D([0], [0], color=color, linestyle='-',
                           marker=marker, markersize=2.5,
                           markerfacecolor=color, markeredgecolor=color,
                           markeredgewidth=0.5, lw=1)
                )
                step_labels.append(base_key)
                seen_step.add(base_key)

        ax.set_xlabel(r"Time ($\boldsymbol{\mu}$s)", fontweight="bold")
        ax.set_ylabel("L2-Rel", fontweight="bold")

        if ylim_l2 is not None:
            ax.set_ylim(ylim_l2)
        ax.grid(True, linestyle="--", alpha=0.3)
        # 刻度粗體
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        # === 右圖：FC ===
        ax = axes[1]
        for model_name, data in data_all.items():
            base_key = model_name.split(" - ")[-1]
            color  = color_map[base_key]
            marker = marker_map[base_key]

            metric_mean = np.mean(data["Fluctuation Correlation"], axis=0)
            steps_us = np.arange(len(metric_mean[4:])) * 5
            y = metric_mean[4:]

            ax.plot(steps_us, y, color=color, linestyle='-',
                    marker=marker, markersize=2.5,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=0.5, markevery=10)

        # 把 xlabel 改成這樣
        ax.set_xlabel(r"Time ($\mu$s)", fontweight="bold")
        ax.set_ylabel("Fluctuation Correlation", fontweight="bold")

        if ylim_fc is not None:
            ax.set_ylim(ylim_fc)
        ax.grid(True, linestyle="--", alpha=0.3)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        # === 共用圖例 ===
        max_per_row = 4
        ncol = min(max_per_row, len(step_handles))
        nrow = math.ceil(len(step_handles) / max_per_row)

        fig.legend(step_handles, step_labels,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.06),
                   ncol=ncol, frameon=False,
                   columnspacing=1.2, handlelength=2.2,
                   prop={"weight": "bold", "family": "DejaVu Sans"})

        fig.subplots_adjust(bottom=0.22 + 0.05 * (nrow - 1), wspace=0.3)

        # 存圖（PDF/PNG 同時）
        root, ext = os.path.splitext(save_path)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
        if ext.lower() != ".pdf":
            fig.savefig(root + ".pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(root + ".png", bbox_inches="tight", pad_inches=0.02, dpi=600)
        plt.close(fig)


if __name__ == "__main__":

    # Dataset 
    dataset_path = "/mnt/ssd/guowei_R12921078/pino_project/Inference/Data/3D_hom/training_data_3d_0819_hom.h5"

    # Directory for saving error comparison results
    output_dir = "/mnt/ssd/guowei_R12921078/pino_project/Inference/error_results/3D_hom/test"

    # Paths to pre-trained models for evaluation
    model_paths = {
        "Test - No-Patch": "/mnt/ssd/guowei_R12921078/pino_project/Inference/model/Final/3D_hom/nopatch.pt",
        "Test - Patch": "/mnt/ssd/guowei_R12921078/pino_project/Inference/model/Final/3D_hom/patch.pt",
    }

    # Paths to optional post-processing models
    model_post_paths = [
        "/mnt/ssd/guowei_R12921078/pino_project/Inference/model/Final/3D_hom/post_process.pt",
        "/mnt/ssd/guowei_R12921078/pino_project/Inference/model/Final/3D_hom/post_process.pt",
    ]

    # Enable/disable post-processing for each test case
    post_process = [False, False]

    # Model parameter variables
    different = [[75,75,100],[50,50,50],[75,75,100],[50,50,50]]

    # === Device Configuration ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Falling back to CPU.")

    # === Initialization of Shared Variables ===
    data_all = {}                 # Store metrics for all evaluated models
    model = None                  # Placeholder for main model
    prev_model_path = None        # Track previously loaded main model to avoid reloading
    model_post = None             # Placeholder for post-processing model
    prev_model_post_path = None   # Track previously loaded post model

    # === Model Inference Loop ===
    for i, (model_name, model_path) in enumerate(model_paths.items()):
        print(f"\n=== Processing Model: {model_name} ===")

        # Load main model only if path differs from previous iteration
        if model_path != prev_model_path:
            if model_path.endswith(".pt"):
                # Load model from PyTorch checkpoint (.pt)
                model = F_FNO(
                    in_channels=5,
                    out_channels=10,
                    dimension=3,
                    latent_channels=32,
                    num_fno_layers=8,
                    num_fno_modes=different[i],
                    padding=0,
                    decoder_layers=1,
                    decoder_layer_size=128,
                ).to(device)
                
                loadmodel_start_time = time.time()
                model.load_state_dict(torch.load(model_path, map_location=device))
                loadmodel_end_time = time.time()
                print(f"Load main model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

            else:
                # Load model from  PhysicsNeMo checkpoint (.mdlus)
                loadmodel_start_time = time.time()
                model = physicsnemo.Module.from_checkpoint(model_path).to(device)
                loadmodel_end_time = time.time()
                print(f"Load main model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

            prev_model_path = model_path
        else:
            print("Main Model path is the same as previous. Skipping reload.")

        # === Load Post-Processing Model (if enabled) ===
        if post_process[i]:
            if model_post_paths[i] != prev_model_post_path:
                if model_post_paths[i].endswith(".pt"):
                    # Load post-processing model from PyTorch checkpoint
                    model_post = F_FNO(
                        in_channels=10,
                        out_channels=10,
                        dimension=3,
                        latent_channels=32,
                        num_fno_layers=8,
                        num_fno_modes=[75,75,100],
                        padding=0,
                        decoder_layers=1,
                        decoder_layer_size=128,
                    ).to(device)
                    loadmodel_start_time = time.time()
                    model_post.load_state_dict(torch.load(model_post_paths[i], map_location=device))
                    loadmodel_end_time = time.time()
                    print(f"Load post-processing model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

                else:
                    # Load PhysicsNeMo checkpoint
                    loadmodel_start_time = time.time()
                    model_post = physicsnemo.Module.from_checkpoint(model_post_paths[i]).to(device)
                    loadmodel_end_time = time.time()
                    print(f"Load post model took {loadmodel_end_time - loadmodel_start_time:.2f} sec")

                prev_model_post_path = model_post_paths[i]
            else:
                print("Post-Processing Model path is the same as previous. Skipping reload.")

        # === Perform Dataset Inference and Metric Computation ===
        # Run the model(s) on the test dataset and store evaluation metrics
        data_all[model_name] = process_dataset(
            dataset_path, 
            model, 
            model_post, 
            model_name, 
            case_nums=1, 
            mult_input=3, 
            mult_output=10, 
            sigma=16, 
            pressure_std=0.1, 
            pred_time=100,
            threshold=0.02, 
            device=device, 
            jwave_as_input=False, 
            post_process=post_process[i]
        )

    # === Visualization and Result Saving ===
    os.makedirs(output_dir, exist_ok=True)

    # Plot comparison of global L2-relative error across models
    plot_multi_models_error(
        data_all=data_all,
        ylabel="L2-Rel",
        save_path=os.path.join(output_dir, "comparison_relative_L2_error.png"),
        metrics=["Relative L2 Error"],
        ylim=[0, 1]
    )

    # Plot comparison of L2-relative error in the region of interest (ROI)
    plot_multi_models_error(
        data_all=data_all,
        ylabel="L2-Rel in the Region of Interest",
        save_path=os.path.join(output_dir, "comparison_relative_L2_error_ROI.png"),
        metrics=["Relative L2 Error(Region of Interest)"],
        ylim=[0, 1]
    )

    # Plot comparison of Fluctuation Correlation (FC) across models
    plot_multi_models_error(
        data_all=data_all,
        ylabel="Fluctuation Correlation",
        save_path=os.path.join(output_dir, "comparison_fluctuation_correlation.png"),
        metrics=["Fluctuation Correlation"],
        ylim=[0.7, 1]
    )

    # Plot combined visualization of L2-Rel and FC side by side
    plot_l2rel_fc_side_by_side(
        data_all=data_all,
        save_path=os.path.join(output_dir, "comparison_L2Rel_FC.png"),
        ylim_l2=[0, 1],
        ylim_fc=[0.7, 1]
    )