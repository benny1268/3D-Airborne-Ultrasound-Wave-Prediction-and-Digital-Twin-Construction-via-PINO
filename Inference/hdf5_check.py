import h5py
import numpy as np

# 讀取 HDF5 檔案
file_path = "./Data/training_data_2d_0628.h5"  # 請根據實際路徑修改
with h5py.File(file_path, 'r') as f:
    # 列出所有資料集名稱與形狀
    print("資料集名稱及其形狀：")
    for name in f:
        dataset = f[name]
        print(f"{name} 的形狀: {dataset.shape}")

    # 取得 pressure 資料並計算 mean 和 std
    pressure = f["pressure"][:]  # shape = (10, 170, 256, 256)
    mean_val = np.mean(pressure)
    std_val = np.std(pressure)

    print(f"\nPressure 的 Mean: {mean_val:.6f}")
    print(f"Pressure 的 Std: {std_val:.6f}")
