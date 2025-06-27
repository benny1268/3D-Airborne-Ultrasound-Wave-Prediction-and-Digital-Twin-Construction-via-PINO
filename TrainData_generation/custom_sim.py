import numpy as np
import h5py
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.signal_processing import tone_burst
from utils import simulate_wave_propagation
import pandas as pd

def generate_random_sources_and_signals(domain, time_axis):
    """
    生成10*10的線性陣列聲源
    
    - 聲源數量: 隨機在 [min_sources, max_sources] 之間
    - 聲源位置: 固定 z=15，以 y 軸為中心等間距排列
    - 間距: 隨機在 [min_spacing, max_spacing] 之間
    - 信號: num_cycles 介於 [1, 5]
    - 延遲時間: 0~2π（即不超過一個週期）

    Returns:
    - `sources`: Sources 物件
    - `signals`: JAX array, shape=(num_sources, len(time_axis))
    - `positions`: (tuple, tuple) 格式的聲源位置
    """
    
    # 聲源數量與間距
    num_sources_x = 10
    num_sources_y = 10
    num_total_sources = num_sources_x * num_sources_y
    spacing = 10
    
    # 計算聲源位置 (固定在 z=15，x,y 軸以 domain 中心為對稱點)
    domain_center_x = domain.N[0] // 2 
    start_x = domain_center_x - ((num_sources_x - 1) // 2) * spacing -  ((num_sources_x - 1) % 2) * (spacing//2)# 中心對齊排列
    x_positions = tuple(start_x + i * spacing for i in range(num_sources_x))
    x_positions = x_positions * num_sources_y

    domain_center_y = domain.N[1] // 2
    start_y = domain_center_y - ((num_sources_y - 1) // 2) * spacing -  ((num_sources_y - 1) % 2) * (spacing//2)# 中心對齊排列
    y_positions = ()
    for i in range(num_sources_y):
        y_positions += (start_y + i * spacing,) * num_sources_x

    z_positions = (15,) * num_total_sources

    positions = (x_positions, y_positions, z_positions)

    # 產生時間軸
    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    max_delay = 0.000025

    # 定義波形參數
    signal_freq = 40e3  # 頻率 (Hz)
    max_delay = 1/signal_freq
    sample_freq = 1 / time_axis.dt  # 取樣頻率
    signals = []
    num_cycles = 5  # 隨機 cycles 數量
    amplitude = 150
    phase_list = [
        13, 23, 0, 5, 8, 0, 29, 24, 15, 5,
        23, 3, 11, 16, 19, 11, 8, 3, 27, 15,
        0, 11, 19, 25, 27, 19, 17, 11, 3, 24,
        5, 16, 25, 30, 2, 26, 22, 17, 8, 29,
        8, 19, 27, 2, 5, 29, 26, 19, 11, 0,
        16, 27, 3, 10, 13, 21, 18, 11, 3, 24,
        13, 24, 1, 6, 10, 18, 14, 9, 0, 21,
        8, 19, 27, 1, 3, 11, 9, 3, 27, 16,
        31, 11, 19, 24, 27, 3, 0, 27, 19, 7,
        21, 31, 8, 13, 16, 24, 21, 16, 7, 29
    ]
    for i in range(num_total_sources):
        delay = max_delay*phase_list[i]/32  # 可依需求設定延遲
    
        # 產生連續波
        s = amplitude * jnp.sin(2 * jnp.pi * signal_freq * t)
    
        # 延遲位移（sample數）
        delay_samples = int(delay / time_axis.dt)
    
        # 建立空訊號陣列
        delayed_signal = jnp.zeros_like(t)
    
        # 把完整連續波貼到延遲後的位置，若延遲超過0則前面補零
        if delay_samples < len(t):
            delayed_signal = delayed_signal.at[delay_samples:].set(s[:len(t) - delay_samples])
        # 若 delay_samples >= len(t) 就是整個訊號被delay超過長度，訊號全為零
    
        signals.append(delayed_signal)

    signals = jnp.stack(signals)  # 堆疊所有聲源的訊號

    # 建立 Sources 物件
    sources = Sources(
        positions=positions,
        signals=signals,
        dt=time_axis.dt,
        domain=domain,
    )

    return sources

def generate_shape_mask(domain_size=(148, 148, 276), num_shapes=1):
    masks = np.zeros((num_shapes, domain_size[0], domain_size[1], domain_size[2]), dtype=int)  # 3D 矩陣，每層對應一個異質區
    densities = []  # 記錄密度
    sound_speeds = []  # 記錄聲速
    fixed_center = (74, 74, 170)
    
    for i in range(num_shapes):
        shape_type = random.choice(['sphere', 'cube', 'cylinder'])
        center = fixed_center  # 使用固定的中心
        
        if shape_type == 'sphere':
            size = random.randint(5, 50)
            rr, cc, zz = sphere(center, size, domain_size)
        elif shape_type == 'cube':
            size = random.randint(10, 100)
            rr, cc, zz = cube(center, size, domain_size)
        elif shape_type == 'cylinder':
            size = random.randint(10, 50)
            rr, cc, zz = cylinder(center, size, domain_size)
        
        masks[i, rr, cc, zz] = 1
    
        # 隨機選擇材質
        material = random.choice(list(material_properties.keys()))
        density = material_properties[material]["density"]
        sound_speed = material_properties[material]["sound_speed"]
        
        # 儲存異質區資訊
        densities.append(density)
        sound_speeds.append(sound_speed)

    return masks, densities, sound_speeds

def sphere(center, radius, domain_size):
    x_center, y_center, z_center = center
    x, y, z = np.indices(domain_size)
    sphere_mask = (x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2 <= radius**2
    return np.where(sphere_mask)

def cube(center, size, domain_size):
    x_center, y_center, z_center = center
    half_size = size // 2
    x_min, x_max = max(x_center - half_size, 0), min(x_center + half_size, domain_size[0])
    y_min, y_max = max(y_center - half_size, 0), min(y_center + half_size, domain_size[1])
    z_min, z_max = max(z_center - half_size, 0), min(z_center + half_size, domain_size[2])
    
    x, y, z = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), np.arange(z_min, z_max), indexing='ij')
    return x.flatten(), y.flatten(), z.flatten()

def cylinder(center, radius, domain_size):
    x_center, y_center, z_center = center
    x, y, z = np.indices(domain_size)
    cylinder_mask = ((x - x_center)**2 + (y - y_center)**2 <= radius**2) & (z >= z_center - radius) & (z <= z_center + radius)
    return np.where(cylinder_mask)

def save_hdf5(hdf5_filename, stacked_pressure, density_patch, sound_speed_patch):
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("pressure", data=stacked_pressure, dtype=np.float16, compression="gzip", compression_opts=9, chunks=(1, 64, 64, 64))
        f.create_dataset("density", data=density_patch, dtype=np.float16, compression="gzip", compression_opts=9, chunks=(64, 64, 64))
        f.create_dataset("sound_speed", data=sound_speed_patch, dtype=np.float16, compression="gzip", compression_opts=9, chunks=(64, 64, 64))
    print(f"Saved {hdf5_filename} with shape {stacked_pressure.shape}")

N = 1
window_size = 64
stride = 32
num_patches_xy = (128- window_size) // stride + 1
num_patches_z = (256 - window_size) // stride + 1
threshold = 1e-3

devices = jax.devices()
print(devices)
hdf5_filename = "/home/jovyan/shared/Kuowei/trainingdata/votrex_3d_0515.h5"
with h5py.File(hdf5_filename, "w") as f1:
    dset_pressure = f1.create_dataset("pressure", shape=(N, 200, 128, 128, 256), dtype=np.float16, compression="gzip", compression_opts=9, chunks=(1, 1, 128, 128, 256))
    dset_density = f1.create_dataset("density", shape=(N, 128, 128, 256), dtype=np.float16, compression="gzip", compression_opts=9, chunks=(1, 128, 128, 256))
    dset_sound_speed = f1.create_dataset("sound_speed", shape=(N, 128, 128, 256), dtype=np.float16, compression="gzip", compression_opts=9, chunks=(1, 128, 128, 256))

    # 讀取CSV檔案
    csv_file = 'material_properties.csv'  # 假設您的CSV文件名為 'material_properties.csv'
    df = pd.read_csv(csv_file)

    # 檢查數據結構
    print(df.head())

    # 建立材質屬性字典
    material_properties = {}

    # 迭代每一行，並將資料填入字典
    for index, row in df.iterrows():
        material_properties[row['Material']] = {
            'density': row['Density (kg/m³)'],
            'sound_speed': row['Speed of Sound (m/s)']
        }
        
    for i in range(N):
        print(f"Running simulation {i+1}/{N}...")
        start_time = time.time()
        domain = Domain((148, 148, 276), (1e-3, 1e-3, 1e-3))

        # # 隨機產生 1 到 5 個異質區
        # num_shapes = 1 #random.randint(1, 2)
        # shape_mask, densities, sound_speeds = generate_shape_mask(domain.N, num_shapes)

        # 定義非均質場
        density = np.ones(domain.N) * 1.38
        sound_speed = np.ones(domain.N) * 343
        # for j in range(num_shapes):
        #     density[shape_mask[j] == 1] = densities[j]
        #     sound_speed[shape_mask[j] == 1] = sound_speeds[j]
        # density[0:148, 0:148, 0:100] = 1.38
        # sound_speed[0:148, 0:148, 0:100] = 343
        
        # **對 density 和 sound_speed 取對數**
        density_ = np.log10(density*10)
        sound_speed_ = np.log10(sound_speed)
        
        density_ = density_[10:-10, 10:-10, 10:-10].astype(np.float16)
        sound_speed_ = sound_speed_[10:-10, 10:-10, 10:-10].astype(np.float16)

        dset_density[i] = density_
        dset_sound_speed[i] = sound_speed_

        density = FourierSeries(np.expand_dims(density, -1), domain)
        sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)

        medium = Medium(domain=domain, sound_speed=sound_speed, density=density)
        dt = 1e-8
        t_end = 0.001
        time_axis = TimeAxis(dt=float(dt), t_end=float(t_end))
        total_step = int(time_axis.t_end / time_axis.dt / 500)
        
        # 定義 Sources
        sources = generate_random_sources_and_signals(domain, time_axis)

        @jit
        def compiled_simulator():
            return simulate_wave_propagation(medium, time_axis, sources=sources)
        pressure = compiled_simulator()

        # # 設定異質區的聲壓為 0
        # for j in range(num_shapes):
        #     pressure = pressure.at[:, shape_mask[j] == 1].set(0)  # 設定所有時間步的該區域聲壓為 0
        pressure = np.squeeze(pressure, axis=-1)[:, 10:-10, 10:-10, 10:-10]
        pressure = (pressure.at[:].set(jnp.sign(pressure) * jnp.log10(1 + jnp.abs(pressure)))).astype(np.float16)

        if np.any(np.isnan(pressure)):
            print(f"NaN detected in pressure at simulation {i+1}, skipping.")
            continue

        dset_pressure[i] = pressure.astype(np.float16)
        
        execution_time = time.time() - start_time
        print(f"Simulation {i+1} saved, time: {execution_time:.2f} seconds")
