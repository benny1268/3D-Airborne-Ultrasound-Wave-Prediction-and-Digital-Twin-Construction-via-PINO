import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import physicsnemo
import pandas as pd
import os
import time

def predict_full_multstep(file_path, model_file_path, mult_input=2, mult_output=7,
                          case_idx=0, pressure_mean=0, pressure_std=1.0, device="cuda"):
    """
    Predict the full spatial pressure field over time using a trained FNO model.
    This version uses multi-step input and output (no patching).

    Args:
        file_path (str): Path to HDF5 file containing input fields.
        model_file_path (str): Path to the trained model (.mdlus).
        mult_input (int): Number of input time steps to use.
        mult_output (int): Number of steps to predict per forward pass.
        case_idx (int): Index of the case to predict.
        pressure_mean (float): Mean used for pressure normalization.
        pressure_std (float): Std used for pressure normalization.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        np.ndarray: Full predicted pressure tensor of shape (T, H, W).
    """

    # === Load initial pressure and static fields ===
    def read_input(file_path, idx=0, mult_input=1, pressure_mean=0, pressure_std=1.0):
        density_min, density_max = 1.38, 5500
        sound_speed_min, sound_speed_max = 150, 5400

        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][idx].shape[0]
            pressure_in = torch.tensor(f["pressure"][idx][:mult_input], dtype=torch.float32)
            pressure_in = (pressure_in - pressure_mean) / pressure_std

            density = torch.tensor(f["density"][idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][idx], dtype=torch.float32).unsqueeze(0)

            # Min-Max normalization
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            # Concatenate mult_input pressure + static fields
            invar = torch.cat([pressure_in, density, sound_speed], dim=0)

        return invar, total_steps

    # === Prepare input and output tensors ===
    input_data, total_steps = read_input(file_path, case_idx, mult_input, pressure_mean, pressure_std)
    input_data = input_data.to(device)

    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype, device=device)
    result_matrix[0:mult_input] = input_data[0:mult_input]  # Initialize with known input steps

    # === Load the trained model ===
    model = physicsnemo.Module.from_checkpoint(model_file_path).to(device)
    model.eval()

    # === Autoregressive prediction loop ===
    for step in range(0, total_steps - mult_input, mult_output):
        # Gather mult_input steps as model input
        input_data[:mult_input] = result_matrix[step:step + mult_input]
        X = input_data.unsqueeze(0)  # Shape: (1, C, H, W)

        with torch.no_grad():
            pressure_out = model(X).squeeze(0)  # Shape: (mult_output, H, W)

        # Prevent overflow beyond result_matrix
        remaining_steps = total_steps - (step + mult_input)
        actual_pred_time = min(mult_output, remaining_steps)

        result_matrix[step + mult_input:step + mult_input + actual_pred_time] = pressure_out[:actual_pred_time]

    # === Denormalize and return as numpy array ===
    result_matrix = (result_matrix * pressure_std) + pressure_mean
    return result_matrix.cpu().numpy()

def predict_patch_multstep(file_path, model_file_path, mult_input=1, mult_output=7, case_idx=0,
                     block_size=64, slide_step=32, batch_size=64,
                     pressure_mean=0, pressure_std=1.0, device="cuda", jwave_as_input=False):
    """
    Predict multi-step pressure fields using a sliding window patch-based FNO model.

    Args:
        file_path (str): Path to input HDF5 file.
        model_file_path (str): Path to saved PhysicsNeMo model checkpoint.
        mult_input (int): Number of input time steps.
        mult_output (int): Number of steps to predict at once.
        case_idx (int): Index of the sample to predict from the HDF5 file.
        block_size (int): Size of the spatial patch.
        slide_step (int): Sliding step between patches.
        batch_size (int): Batch size for model inference.
        pressure_mean (float): Mean for pressure normalization.
        pressure_std (float): Std for pressure normalization.
        device (str): Device to run inference on.
        jwave_as_input (bool): If True, use ground-truth (jwave) for the input sequence.

    Returns:
        np.ndarray: Reconstructed full-field pressure tensor of shape (T, H, W)
    """

    # === Step 1: Load Input Tensor (p0, density, sound_speed) ===
    def read_input(file_path, idx=0, mult_input=1, pressure_mean=0, pressure_std=1.0):
        density_min, density_max = 1.38, 5500
        sound_speed_min, sound_speed_max = 150, 5400

        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][idx].shape[0]

            # Read mult_input steps of pressure
            pressure_t = torch.tensor(f["pressure"][idx][:mult_input], dtype=torch.float32)
            pressure_t = (pressure_t - pressure_mean) / pressure_std

            # Read and normalize material fields
            density = torch.tensor(f["density"][idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][idx], dtype=torch.float32).unsqueeze(0)
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            # Concatenate all input channels
            invar = torch.cat([pressure_t, density, sound_speed], dim=0)

        return invar, total_steps

    # === Step 2: Sliding Window Patch Extraction ===
    def slide_and_split(input_data, block_size=64, slide_step=32):
        ndim = input_data.ndim
        blocks = []

        if ndim == 3:  # (C, H, W)
            _, H, W = input_data.shape
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    blocks.append(input_data[:, i:i+block_size, j:j+block_size])
        elif ndim == 4:  # (C, D, H, W)
            _, D, H, W = input_data.shape
            for d in range(0, D - block_size + 1, slide_step):
                for i in range(0, H - block_size + 1, slide_step):
                    for j in range(0, W - block_size + 1, slide_step):
                        blocks.append(input_data[:, d:d+block_size, i:i+block_size, j:j+block_size])
        else:
            raise ValueError("Only 2D or 3D tensors are supported.")

        return blocks

    # === Step 3: Gaussian Weight Kernel (used for smooth blending) ===
    def gaussian_weight_tensor(block_size=64, sigma=None, dim=2, device="cpu", dtype=torch.float32):
        if sigma is None:
            sigma = block_size / 12
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

    # === Step 4: Combine Patches into Full Field ===
    def combine(blocks, result_shape, pred_time=7, block_size=64, slide_step=32, device="cpu"):
        ndim = len(result_shape)
        dtype = blocks[0].dtype

        if ndim == 3:
            _, H, W = result_shape
            pred_full = torch.zeros((pred_time, H, W), device=device, dtype=dtype)
            count_full = torch.zeros_like(pred_full)
            weight = gaussian_weight_tensor(block_size, dim=2, device=device, dtype=dtype)
            weight = weight.unsqueeze(0).repeat(pred_time, 1, 1)
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    block = blocks.pop(0).to(device)
                    pred_full[:, i:i+block_size, j:j+block_size] += block * weight
                    count_full[:, i:i+block_size, j:j+block_size] += weight

        elif ndim == 4:
            _, D, H, W = result_shape
            pred_full = torch.zeros((pred_time, D, H, W), device=device, dtype=dtype)
            count_full = torch.zeros_like(pred_full)
            weight = gaussian_weight_tensor(block_size, dim=3, device=device, dtype=dtype)
            weight = weight.unsqueeze(0).repeat(pred_time, 1, 1, 1)
            for d in range(0, D - block_size + 1, slide_step):
                for i in range(0, H - block_size + 1, slide_step):
                    for j in range(0, W - block_size + 1, slide_step):
                        block = blocks.pop(0).to(device)
                        pred_full[:, d:d+block_size, i:i+block_size, j:j+block_size] += block * weight
                        count_full[:, d:d+block_size, i:i+block_size, j:j+block_size] += weight

        return pred_full / torch.clamp(count_full, min=1e-8)

    # === Step 5: Initialize Inference ===
    input_data, total_steps = read_input(file_path, case_idx, mult_input, pressure_mean, pressure_std)
    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype)
    result_matrix[0:mult_input] = input_data[0:mult_input]  # Initialize with known input

    model = physicsnemo.Module.from_checkpoint(model_file_path).to(device)
    model.eval()

    # === Step 6: Autoregressive Prediction ===
    for step in range(0, total_steps - mult_input, mult_output):
        preds = []

        # Choose input source: either jwave truth or autoregressive result
        if jwave_as_input:
            with h5py.File(file_path, "r") as f:
                pressure_t = torch.tensor(f["pressure"][case_idx][step:step+mult_input], dtype=torch.float32)
                pressure_t = (pressure_t - pressure_mean) / pressure_std
            input_data[:mult_input] = pressure_t
        else:
            input_data[:mult_input] = result_matrix[step:step+mult_input]

        blocks = slide_and_split(input_data, block_size, slide_step)
        batch_blocks = []

        # === Patchwise Inference ===
        for idx, block in enumerate(blocks):
            if torch.all(torch.abs(block[0]) < 0.001):
                # Skip silent region
                preds.append((idx, torch.zeros((mult_output,) + block.shape[1:], dtype=input_data.dtype)))
            else:
                batch_blocks.append((idx, block.unsqueeze(0)))
                if len(batch_blocks) == batch_size:
                    batch_indices, batch_inputs = zip(*batch_blocks)
                    batch_input = torch.cat(batch_inputs, dim=0).to(device)
                    with torch.no_grad():
                        batch_output = model(batch_input)
                    for out_idx, output in zip(batch_indices, batch_output):
                        preds.append((out_idx, output.cpu()))
                    batch_blocks = []

        # Remaining patches
        if batch_blocks:
            batch_indices, batch_inputs = zip(*batch_blocks)
            batch_input = torch.cat(batch_inputs, dim=0).to(device)
            with torch.no_grad():
                batch_output = model(batch_input)
            for out_idx, output in zip(batch_indices, batch_output):
                preds.append((out_idx, output.cpu()))

        # Sort predictions by index and combine
        preds = [pred for _, pred in sorted(preds, key=lambda x: x[0])]
        remaining_steps = result_matrix.shape[0] - (step + mult_input)
        actual_pred_time = min(mult_output, remaining_steps)

        combined_result = combine(preds, result_matrix.shape, mult_output, block_size, slide_step)
        result_matrix[step+mult_input:step+mult_input+actual_pred_time] = combined_result[:actual_pred_time]

    # === Step 7: Denormalize and Return ===
    result_matrix = (result_matrix * pressure_std) + pressure_mean
    return result_matrix.cpu().numpy()

def create_comparison_gif(result_matrix, file_path, case_idx=0, output_gif="Local_PINO_result.gif", fps=10, plot=True, log_process=False):
    """
    Create a GIF animation comparing the predicted pressure field and ground truth over time,
    and save error statistics.

    Args:
        result_matrix (np.ndarray): Predicted pressure field. Shape = (T, H, W)
        file_path (str): Path to the HDF5 file with ground truth.
        case_idx (int): Index of the case in the HDF5 file.
        output_gif (str): Output filename for the saved GIF.
        fps (int): Frames per second for the animation.
        plot (bool): Whether to generate the visualization or not.
        log_process (bool): Whether inverse log-transform is needed.
    """

    # === Load ground truth pressure field ===
    with h5py.File(file_path, "r") as f:
        total_steps = f["pressure"][case_idx].shape[0]
        pressure_field_gt = np.array(f['pressure'][case_idx, 0:total_steps])

    # === Calculate per-step Relative L2 Error and print ===
    relative_L2_error_list = []
    for i in range(total_steps):
        mean_gt = np.mean(pressure_field_gt[i])
        numerator = np.mean((pressure_field_gt[i] - result_matrix[i]) ** 2)
        denominator = np.mean((pressure_field_gt[i] - mean_gt) ** 2)
        relative_L2_error = numerator / denominator
        print(f"Step {i} : Relative L₂ Error = {relative_L2_error}")
        relative_L2_error_list.append(relative_L2_error)
    total = np.mean(relative_L2_error_list)
    print(f"Average Relative L₂ Error = {total}")

    # === Calculate full-domain Mean Absolute Error (MAE) percentage over all time steps ===
    max_gt_value = np.max(np.abs(pressure_field_gt[0]))
    mae_all_points = np.abs(pressure_field_gt - result_matrix) / max_gt_value * 100
    mae_flat = mae_all_points.flatten()

    # === Group error values into intervals and count percentages ===
    bins = [0, 1, 3, 7, 15, 31, 63, 1000]
    labels = ['0-1%', '1-3%', '3-7%', '7-15%', '15-31%', '31-63%', '>63%']
    categories = pd.cut(mae_flat, bins=bins, labels=labels, right=False)
    counts = categories.value_counts().sort_index()
    total_points = mae_flat.size
    percentages = counts / total_points * 100

    df = pd.DataFrame({
        'MAE Range': counts.index,
        'Number of Points': counts.values,
        'Percentage (%)': percentages.values
    })

    # === Save error distribution as CSV ===
    csv_path = 'mae_allpoints_distribution_percentage.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"MAE distribution over all time and points saved to {csv_path}")
    print(df)

    if plot:
        # === Setup the plotting layout ===
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].set_title("Ground Truth (Pa)")
        ax[1].set_title("PINO (Pa)")
        ax[2].set_title("Error (%)")

        # === Initialize empty heatmaps ===
        img_gt = ax[0].imshow(np.zeros((256, 256)), cmap="RdBu", animated=True)
        img_pred = ax[1].imshow(np.zeros((256, 256)), cmap="RdBu", animated=True)
        img_rlt_diff = ax[2].imshow(np.zeros((256, 256)), cmap="RdBu", animated=True)

        cbar_gt = plt.colorbar(img_gt, ax=ax[0], shrink=0.5)
        cbar_pred = plt.colorbar(img_pred, ax=ax[1], shrink=0.5)
        cbar_rlt_diff = plt.colorbar(img_rlt_diff, ax=ax[2], shrink=0.5)

        # === Function to update frame for animation ===
        def update_fig(step):
            abs_diff_data = np.abs(result_matrix[step] - pressure_field_gt[step])
            rlt_diff_data = abs_diff_data / max_gt_value * 100

            img_gt.set_data(pressure_field_gt[step])
            img_pred.set_data(result_matrix[step])
            img_rlt_diff.set_data(rlt_diff_data)

            ax[0].set_title(f"Kwave (Pa) at Step {step}")
            ax[1].set_title(f"PINO (Pa) at Step {step}")
            ax[2].set_title(f"Relative Error (%) at Step {step}")

            img_gt.set_clim(np.min(pressure_field_gt[step]), np.max(pressure_field_gt[step]))
            img_pred.set_clim(np.min(result_matrix[step]), np.max(result_matrix[step]))
            img_rlt_diff.set_clim(np.min(rlt_diff_data), np.max(rlt_diff_data))

            cbar_gt.mappable.set_clim(np.min(pressure_field_gt[step]), np.max(pressure_field_gt[step]))
            cbar_pred.mappable.set_clim(np.min(result_matrix[step]), np.max(result_matrix[step]))
            cbar_rlt_diff.mappable.set_clim(np.min(rlt_diff_data), np.max(rlt_diff_data))

            return [img_gt, img_pred, img_rlt_diff]

        # === Create and save the GIF ===
        ani = animation.FuncAnimation(fig, update_fig, frames=range(0, total_steps), interval=1000/fps, blit=True)
        ani.save(output_gif, writer='pillow', fps=fps)

# === Main Execution ===
if __name__ == "__main__":
    case_idx = 3
    file_path = "./Data/training_data_2d_0703_v4.h5"
    model_path = "./model/hele_2d_patch/0703/v1.mdlus"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Falling back to CPU.")

    # === Build output path based on model name and data name ===
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    data_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("gif_plot", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_gif = os.path.join(output_dir, f"{data_name}_case{case_idx}.gif")
    
    # === Run prediction and create comparison visualization ===
    print(f"Processing Case {case_idx}...")
    start_time = time.time()
    result_matrix = predict_patch_multstep(file_path=file_path, model_file_path=model_path, mult_input=2, mult_output=7, 
                                            case_idx=case_idx, device=device, pressure_std=0.065)
    print(f"Compute time : {time.time() - start_time}")
    create_comparison_gif(result_matrix, file_path, case_idx, output_gif=output_gif, fps=10)

