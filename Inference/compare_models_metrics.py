import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import physicsnemo
import time
import os
def predict_full_domain(file_path, model_file_path, pred_time=7, case_idx=0, pressure_mean=0, pressure_std=1.0, device="cuda"):
    """
    Predict the full spatial domain using a trained model (no patching).

    Args:
        file_path (str): Path to the HDF5 dataset.
        model_file_path (str): Path to the trained model (.mdlus).
        pred_time (int): Number of time steps to predict in each forward pass.
        case_idx (int): Index of the case in the HDF5 file.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        np.ndarray: Predicted pressure field over time. Shape: (T, ...)
    """

    # === Load initial input: p0, density, sound_speed ===
    def read_input(file_path, idx=0, pressure_mean=0, pressure_std=1.0):
        density_min = 1.38
        density_max = 5500
        sound_speed_min = 150
        sound_speed_max = 5400
        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][idx].shape[0]
            pressure_t = torch.tensor(f["pressure"][idx][0], dtype=torch.float32).unsqueeze(0)
            pressure_t = (pressure_t - pressure_mean)  / pressure_std
            density = torch.tensor(f["density"][idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][idx], dtype=torch.float32).unsqueeze(0)
            # Min-Max Normalization to [0, 1]
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            invar = torch.cat([pressure_t, density, sound_speed], dim=0)
        return invar, total_steps

    # === Read input and initialize result tensor ===
    input_data, total_steps = read_input(file_path, case_idx, pressure_mean, pressure_std)
    input_data = input_data.to(device)
    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype, device=device)
    result_matrix[0] = input_data[0]  # Set initial pressure p0

    # === Load the trained model ===
    model = physicsnemo.Module.from_checkpoint(model_file_path).to(device)
    model.eval()

    # === Recursive prediction loop ===
    for step in range(0, total_steps, pred_time):
        input_data[0] = result_matrix[step]  # Update p0 for next input
        with torch.no_grad():
            output = model(input_data.unsqueeze(0))  # Shape: (1, pred_time, ...)
            output = output.squeeze(0)               # Shape: (pred_time, ...)
        remaining_steps = result_matrix.shape[0] - (step + 1)
        actual_pred_time = min(pred_time, remaining_steps)
        result_matrix[step+1:step+1+actual_pred_time] = output[:actual_pred_time]

    result_matrix = (result_matrix * pressure_std ) + pressure_mean

    return result_matrix.cpu().numpy()

def predict_multstep(file_path, model_file_path, pred_time=7, case_idx=0, block_size=64, slide_step=32, batch_size=64, pressure_mean=0, pressure_std=1.0, device="cuda"):
    # === Load the initial input (p0, density, sound_speed) ===
    def read_input(file_path, idx=0, pressure_mean=0, pressure_std=1.0):
        density_min = 1.38
        density_max = 5500
        sound_speed_min = 150
        sound_speed_max = 5400
        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][idx].shape[0]
            pressure_t = torch.tensor(f["pressure"][idx][0], dtype=torch.float32).unsqueeze(0)
            pressure_t = (pressure_t - pressure_mean)  / pressure_std
            density = torch.tensor(f["density"][idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][idx], dtype=torch.float32).unsqueeze(0)
            # Min-Max Normalization to [0, 1]
            density = (density - density_min) / (density_max - density_min)
            sound_speed = (sound_speed - sound_speed_min) / (sound_speed_max - sound_speed_min)

            invar = torch.cat([pressure_t, density, sound_speed], dim=0)
        return invar, total_steps

    # === Split the input into sliding patches (2D or 3D supported) ===
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

    # === Compute Gaussian weight kernel (2D or 3D) ===
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

    # === Reconstruct full tensor from overlapping patches ===
    def combine(blocks, result_shape, pred_time=5, block_size=64, slide_step=32, device="cpu"):
        ndim = len(result_shape)
        dtype = blocks[0].dtype

        if ndim == 3:
            _, H, W = result_shape
            pred_full = torch.zeros((pred_time, H, W), device=device, dtype=dtype)
            count_full = torch.zeros((pred_time, H, W), device=device, dtype=dtype)
            weight = gaussian_weight_tensor(block_size, dim=2, device=device, dtype=dtype)
            stacked = weight.unsqueeze(0).repeat(pred_time, 1, 1)
            for i in range(0, H - block_size + 1, slide_step):
                for j in range(0, W - block_size + 1, slide_step):
                    block = blocks.pop(0).to(device)
                    pred_full[:, i:i+block_size, j:j+block_size] += block * stacked
                    count_full[:, i:i+block_size, j:j+block_size] += stacked

        elif ndim == 4:
            _, D, H, W = result_shape
            pred_full = torch.zeros((pred_time, D, H, W), device=device, dtype=dtype)
            count_full = torch.zeros((pred_time, D, H, W), device=device, dtype=dtype)
            weight = gaussian_weight_tensor(block_size, dim=3, device=device, dtype=dtype)
            stacked = weight.unsqueeze(0).repeat(pred_time, 1, 1, 1)
            for d in range(0, D - block_size + 1, slide_step):
                for i in range(0, H - block_size + 1, slide_step):
                    for j in range(0, W - block_size + 1, slide_step):
                        block = blocks.pop(0).to(device)
                        pred_full[:, d:d+block_size, i:i+block_size, j:j+block_size] += block * stacked
                        count_full[:, d:d+block_size, i:i+block_size, j:j+block_size] += stacked

        return pred_full / torch.clamp(count_full, min=1e-8)

    # === Load input and initialize result tensor ===
    input_data, total_steps = read_input(file_path, case_idx, pressure_mean, pressure_std)
    result_shape = (total_steps,) + input_data.shape[1:]
    result_matrix = torch.zeros(result_shape, dtype=input_data.dtype)
    result_matrix[0] = input_data[0]  # set p0

    # === Load model ===
    model = physicsnemo.Module.from_checkpoint(model_file_path).to(device)
    model.eval()

    # === Main prediction loop ===
    for step in range(0, total_steps, pred_time):
        preds = []
        input_data[0] = result_matrix[step]  # update p0
        blocks = slide_and_split(input_data, block_size, slide_step)
        batch_blocks = []

        for idx, block in enumerate(blocks):
            if torch.all(torch.abs(block[0]) < 0.001):
                zero_shape = (pred_time,) + block.shape[1:]
                preds.append((idx, torch.zeros(zero_shape, dtype=input_data.dtype)))
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

        if batch_blocks:
            batch_indices, batch_inputs = zip(*batch_blocks)
            batch_input = torch.cat(batch_inputs, dim=0).to(device)
            with torch.no_grad():
                batch_output = model(batch_input)
            for out_idx, output in zip(batch_indices, batch_output):
                preds.append((out_idx, output.cpu()))

        preds = [pred for idx, pred in sorted(preds, key=lambda x: x[0])]
        remaining_steps = result_matrix.shape[0] - (step + 1)
        actual_pred_time = min(pred_time, remaining_steps)

        combined_result = combine(preds, result_matrix.shape, pred_time, block_size, slide_step)
        result_matrix[step+1:step+1+actual_pred_time] = combined_result[:actual_pred_time]

    result_matrix = (result_matrix * pressure_std ) + pressure_mean

    return result_matrix.cpu().numpy()

def compute_relative_l2_errors(y_pred, y_true, total_steps):
    """
    Compute Relative L2 Error (standard version):
        ||y_pred - y_true|| / ||y_true||
    """
    error_list = []
    for t in range(total_steps):
        denom = np.sum(y_true[t] ** 2)
        if denom == 0:
            error = np.nan
        else:
            error = np.sqrt(np.sum((y_pred[t] - y_true[t]) ** 2)) / np.sqrt(denom)
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

def process_dataset(file_path, model_path, case_nums=10, pred_time=7, device="cuda", log_process=False, pressure_mean=0, pressure_std=1.0):

    def inverse_transform(pressure_transformed):
        abs_val = np.abs(pressure_transformed)
        sign = np.sign(pressure_transformed)
        original = sign * (np.power(10.0, abs_val) - 1.0)
        return original

        
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
        result_matrix = predict_multstep(file_path=file_path, model_file_path=model_path, pred_time=pred_time, case_idx=case_idx, device=device, pressure_mean=pressure_mean, pressure_std=pressure_std)
        print(f"Compute time : {time.time() - start_time}")
        with h5py.File(file_path, "r") as f:
            total_steps = f["pressure"][case_idx].shape[0]
            pressure_field_gt = np.array(f["pressure"][case_idx], dtype=np.float32)
        if log_process :
            pressure_field_gt = inverse_transform(pressure_field_gt)
            result_matrix = inverse_transform(result_matrix)
        metric_results["Relative L2 Error"].append(compute_relative_l2_errors(result_matrix, pressure_field_gt, total_steps))
        metric_results["RMSE"].append(compute_rmse(result_matrix, pressure_field_gt, total_steps))
        metric_results["Nash–Sutcliffe Efficiency"].append(compute_nash_sutcliffe_efficiency(result_matrix, pressure_field_gt, total_steps))
        metric_results["RMSE (Value> Mean)"].append(compute_rmse_over_mean_threshold(result_matrix, pressure_field_gt, total_steps))
        metric_results["Fluctuation Correlation"].append(compute_fluctuation_correlation(result_matrix, pressure_field_gt, total_steps))
        metric_results["GT_mean_over_mean"].append(compute_mean_over_double_threshold(pressure_field_gt, total_steps))

    return metric_results

def plot_multi_models_with_shaded_error(data_all, ylabel, title, save_path, metrics):
    plt.figure(figsize=(10, 5))
    colors = plt.get_cmap("tab10")

    for model_idx, model_name in enumerate(data_all.keys()):
        color = colors(model_idx)
        data = data_all[model_name]

        for metric in metrics:
            train_mean = np.mean(data[metric], axis=0)
            train_std = np.std(data[metric], axis=0)

            plt.plot(train_mean, label=f"{model_name}", color=color, linestyle="-", linewidth=2)
            plt.fill_between(np.arange(len(train_mean)), train_mean - train_std, train_mean + train_std, alpha=0.2, color=color)

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    data_path = "./Data/training_data_2d_0628_v2.h5"
    output_dir = "./error_results/model_comparison_diff_layers_nums"

    model_paths = {
        "Fourier Layer = 4": "./model/hele_2d_patch/0703/diff_layers/l4.mdlus",
        "Fourier Layer = 6": "./model/hele_2d_patch/0703/diff_layers/l6.mdlus",
        "Fourier Layer = 8": "./model/hele_2d_patch/0703/diff_layers/l8.mdlus",
        "Fourier Layer = 10": "./model/hele_2d_patch/0703/diff_layers/l10.mdlus",       
    }
    data_all = {}

    for i, (model_name, model_path) in enumerate(model_paths.items()):
        print(f"\n=== Processing Model: {model_name} ===")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Processing Dataset...")
        data_all[model_name] = process_dataset(data_path, model_path, case_nums=10, pred_time=7, device=device, log_process=False)

    os.makedirs(output_dir, exist_ok=True)
    plot_multi_models_with_shaded_error(
        data_all=data_all,
        ylabel="Relative L2 Error",
        title="Comparison of Models: Relative L2 Error",
        save_path=os.path.join(output_dir, "comparison_relative_L2_error.png"),
        metrics=["Relative L2 Error"]
    )

    plot_multi_models_with_shaded_error(
        data_all=data_all,
        ylabel="RMSE (Pa)",
        title="Comparison of Models: RMSE and Dynamic RMSE",
        save_path=os.path.join(output_dir, "comparison_rmse_dynamic.png"),
        metrics=["RMSE"]
    )

    plot_multi_models_with_shaded_error(
        data_all=data_all,
        ylabel="Fluctuation Correlation",
        title="Comparison of Models: Fluctuation Correlation",
        save_path=os.path.join(output_dir, "comparison_fluctuation_correlation.png"),
        metrics=["Fluctuation Correlation"],
    )

    plot_multi_models_with_shaded_error(
        data_all=data_all,
        ylabel="Nash–Sutcliffe Efficiency",
        title="Comparison of Models: Nash–Sutcliffe Efficiency",
        save_path=os.path.join(output_dir, "comparison_nse.png"),
        metrics=["Nash–Sutcliffe Efficiency"]
    )



