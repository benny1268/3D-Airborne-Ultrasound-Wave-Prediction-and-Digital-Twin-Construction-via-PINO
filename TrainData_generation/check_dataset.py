import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 讀取 HDF5 檔案
file_path = "./training_data_2d_0627.h5"  # 請修改為實際的檔案路徑
output_dir = "./output_images"  # 儲存圖片的目錄
output_gif_dir = "./output_gifs"  # 儲存GIF的目錄

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

with h5py.File(file_path, 'r') as f:
    # 取得資料集
    density = f['density'][:]
    pressure = f['pressure'][:]
    sound_speed = f['sound_speed'][:]
    # 顯示並儲存 sound_speed 圖片
    for i in range(pressure.shape[0]):
        plt.imshow(pressure[i][0], cmap='RdBu', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Pressure Distribution')
        plt.savefig(f"{output_dir}/pressure{i+1}.png")
        plt.close()

    # 顯示並儲存 density 圖片
    for i in range(density.shape[0]):
        plt.imshow(density[i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'density {i+1}')
        plt.savefig(f"{output_dir}/density_{i+1}.png")
        plt.close()

    # 顯示並儲存 sound_speed 圖片
    for i in range(sound_speed.shape[0]):
        plt.imshow(sound_speed[i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'sound_speed {i+1}')
        plt.savefig(f"{output_dir}/sound_speed_{i+1}.png")
        plt.close()

# 確保輸出目錄存在
os.makedirs(output_gif_dir, exist_ok=True)

with h5py.File(file_path, 'r') as f:
    # 取得資料集
    pressure = f['pressure'][:]

    # 處理每個 case 的 pressure
    for i in range(pressure.shape[0]):  # 50 種 case
        pressure_frames = []

        # 針對每個 case 的 250 幀
        for j in range(pressure.shape[1]):  # 250 frames
            # 將每一張壓力圖轉為灰階圖並縮放為適合的大小
            fig, ax = plt.subplots()
            ax.imshow(pressure[i, j], cmap='hot', interpolation='nearest')
            ax.axis('off')  # 關閉坐標軸
            plt.close(fig)

            # 儲存為圖片
            img_path = f"{output_gif_dir}/pressure_case_{i+1}_frame_{j+1}.png"
            fig.savefig(img_path)
            pressure_frames.append(img_path)

        # 創建 GIF
        images = [Image.open(frame) for frame in pressure_frames]
        gif_path = f"{output_gif_dir}/pressure_case_{i+1}.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=100)  # 每張圖片的顯示時間為 100ms

        # 刪除單張圖片以保持整潔
        for frame in pressure_frames:
            os.remove(frame)

print(f"所有圖片和 GIF 已儲存在")
