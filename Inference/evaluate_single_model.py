import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import physicsnemo
import time
import os

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

def compute_relative_l2_errors(y_pred, y_true, total_steps):
    """
    Compute Relative L2 Error:
        ||y_pred - y_true||² / ||y_true||²
    """
    error_list = []
    for t in range(total_steps):
        denom = np.sum(y_true[t] ** 2)
        if denom == 0:
            error = np.nan
        else:
            error = np.sum((y_pred[t] - y_true[t]) ** 2) / denom
        error_list.append(error)
    return error_list

def compute_nash_sutcliffe_efficiency(y_pred, y_true, total_steps):
    """
    Compute Nash–Sutcliffe Efficiency (NSE):
        1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)
    """
    nse_list = []
    for t in range(total_steps):
        mean_true = np.mean(y_true[t])
        denom = np.sum((y_true[t] - mean_true) ** 2)
        if denom == 0:
            nse = np.nan
        else:
            nse = 1 - np.sum((y_pred[t] - y_true[t]) ** 2) / denom
        nse_list.append(nse)
    return nse_list

def compute_rmse(y_pred, y_true, total_steps):
    """
    Compute Root Mean Squared Error (RMSE):
        sqrt(mean((y_pred - y_true)^2))
    """
    rmse_list = []
    for t in range(total_steps):
        diff = y_pred[t] - y_true[t]
        rmse = np.sqrt(np.mean(diff ** 2))
        rmse_list.append(rmse)
    return rmse_list

def compute_rmse_over_mean_threshold(y_pred, y_true, total_steps):
    """
    Compute RMSE over region where y_true > mean(y_true)
    """
    rmse_list = []
    for t in range(total_steps):
        threshold = np.mean(y_true[t])
        mask = y_true[t] > threshold
        if np.any(mask):
            diff = y_pred[t][mask] - y_true[t][mask]
            rmse = np.sqrt(np.mean(diff ** 2))
        else:
            rmse = 0.0
        rmse_list.append(rmse)
    return rmse_list

def compute_fluctuation_correlation(y_pred, y_true, total_steps):
    """
    Compute fluctuation correlation (Pearson correlation over fluctuations):
        numerator = Σ((y_pred - mean) * (y_true - mean))
        denominator = sqrt(Σ(y_pred_fluc²) * Σ(y_true_fluc²))
    """
    corr_list = []
    for t in range(total_steps):
        true_mean = np.mean(y_true[t])
        pred_mean = np.mean(y_pred[t])

        true_fluc = y_true[t] - true_mean
        pred_fluc = y_pred[t] - pred_mean

        denom = np.sqrt(np.sum(true_fluc ** 2) * np.sum(pred_fluc ** 2))
        if denom == 0:
            corr = 0.0
        else:
            corr = np.sum(true_fluc * pred_fluc) / denom
        corr_list.append(corr)
    return corr_list

def compute_mean_over_double_threshold(y_true, total_steps):
    """
    Compute the mean of values in each time step that are above a 
    second threshold, where the threshold is the mean of values 
    above the first mean.

    For each time step:
        1. Compute first mean (mean_1) of the whole field.
        2. Select values > mean_1 → get mean_2.
        3. Select values > mean_2 → compute final mean.

    Args:
        y_pred (ndarray): Predicted field with shape (T, ...)
        total_steps (int): Total number of time steps (T)

    Returns:
        List[float]: List of final means per time step
    """
    filtered_mean_list = []
    for t in range(total_steps):
        data = y_true[t]
        mask1 = data > np.mean(data)
        if np.any(mask1):
            mean_2 = np.mean(data[mask1])
            mask2 = data > mean_2
            if np.any(mask2):
                filtered_mean = np.mean(data[mask2])
                filtered_mean_list.append(filtered_mean)
                continue
        # Fallback if no valid mask
        filtered_mean_list.append(0.0)
    return filtered_mean_list

def process_dataset(file_path, model_path, case_nums=10, mult_input=1, mult_output=7, device="cuda", pressure_mean=0, pressure_std=1.0, jwave_as_input=False):

        
    metric_results = {
        "Relative L2 Error": [],
        "RMSE": [],
        "Nash–Sutcliffe Efficiency": [],
        "RMSE (Value> Mean)": [],
        "Fluctuation Correlation": [],
        "GT_mean_over_mean": [],
    }

    for case_idx in range(case_nums):
        print(f"Processing Case {case_idx}...")
        start_time = time.time()
        result_matrix = predict_patch_multstep(file_path=file_path, model_file_path=model_path, mult_input=mult_input, mult_output=mult_output, 
                                                case_idx=case_idx, device=device, pressure_mean=pressure_mean, pressure_std=pressure_std, jwave_as_input=jwave_as_input)        
        print(f"Compute time : {time.time() - start_time}")

        
        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][case_idx].shape[0]
            pressure_field_gt = np.array(f["pressure"][case_idx], dtype=np.float32)

        metric_results["Relative L2 Error"].append(compute_relative_l2_errors(result_matrix, pressure_field_gt, total_steps))
        metric_results["RMSE"].append(compute_rmse(result_matrix, pressure_field_gt, total_steps))
        metric_results["Nash–Sutcliffe Efficiency"].append(compute_nash_sutcliffe_efficiency(result_matrix, pressure_field_gt, total_steps))
        metric_results["RMSE (Value> Mean)"].append(compute_rmse_over_mean_threshold(result_matrix, pressure_field_gt, total_steps))
        metric_results["Fluctuation Correlation"].append(compute_fluctuation_correlation(result_matrix, pressure_field_gt, total_steps))
        metric_results["GT_mean_over_mean"].append(compute_mean_over_double_threshold(pressure_field_gt, total_steps))

    return metric_results

def plot_with_shaded_error(train_data, test_data, ylabel, title, save_path, metrics, unify_color=True):
    plt.figure(figsize=(10, 5))
    colors = plt.get_cmap("tab10")

    for idx, label in enumerate(metrics):
        color = colors(idx) if unify_color else None
        train_mean = np.mean(train_data[label], axis=0)
        train_std = np.std(train_data[label], axis=0)
        test_mean = np.mean(test_data[label], axis=0)
        test_std = np.std(test_data[label], axis=0)

        plt.plot(train_mean, label=f"Train Datasets-{label}", color=color, linestyle="-", linewidth=2)
        plt.fill_between(np.arange(len(train_mean)), train_mean - train_std, train_mean + train_std, alpha=0.2, color=color)

        plt.plot(test_mean, label=f"Test Datasets-{label}", color=color, linestyle="--", linewidth=2)
        plt.fill_between(np.arange(len(test_mean)), test_mean - test_std, test_mean + test_std, alpha=0.2, color=color)

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    train_path = "./Data/training_data_2d_0630.h5"
    test_path = "./Data/training_data_2d_0628_v1.h5"
    model_path = "./model/hele_2d_patch/0630/v1.mdlus"

    # Extract model name from model_path (without extension)
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]

    # Create output directory with model name
    output_dir = f"./error_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Processing training set...")
    train_metrics = process_dataset(train_path, model_path, case_nums=10, device=device)

    print("\nProcessing test set...")
    test_metrics = process_dataset(test_path, model_path, case_nums=10, device=device)

    # figure 1: Relative L2 Error
    plot_with_shaded_error(
        train_data=train_metrics,
        test_data=test_metrics,
        ylabel="Relative L2 Error",
        title="Relative L2 Error on Training and Testing Datasets",
        save_path=os.path.join(output_dir, "relative_L2_error_train_vs_test.png"),
        metrics=["Relative L2 Error"],
        unify_color=True
    )

    # figure 2: RMSE, Dynamic RMSE, GT Max Over Mean
    plot_with_shaded_error(
        train_data=train_metrics,
        test_data=test_metrics,
        ylabel="RMSE (Pa)",
        title="RMSE / Dynamic RMSE on Training and Testing Datasets",
        save_path=os.path.join(output_dir, "rmse_dyn_train_vs_test.png"),
        metrics=["RMSE", "RMSE (Value> Mean)"],
        unify_color=True
    )

    # figure 3: Fluctuation Correlation
    plot_with_shaded_error(
        train_data=train_metrics,
        test_data=test_metrics,
        ylabel="Fluctuation Correlation",
        title="Fluctuation Correlation on Training and Testing Datasets",
        save_path=os.path.join(output_dir, "fluctuation_correlation_train_vs_test.png"),
        metrics=["Fluctuation Correlation"],
        unify_color=True
    )

    # figure 4: Nash–Sutcliffe Efficiency
    plot_with_shaded_error(
        train_data=train_metrics,
        test_data=test_metrics,
        ylabel="Nash–Sutcliffe Efficiency",
        title="Nash–Sutcliffe Efficiency on Training and Testing Datasets",
        save_path=os.path.join(output_dir, "Nash_Sutcliffe_Efficiency_train_vs_test.png"),
        metrics=["Nash–Sutcliffe Efficiency"],
        unify_color=True
    )



