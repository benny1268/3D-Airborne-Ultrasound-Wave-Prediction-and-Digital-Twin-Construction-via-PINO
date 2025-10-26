import os
import h5py
import torch
from typing import Union
from torch.utils.data import Dataset
import re
import random

class HDF5MapDataset_patch_multstep(Dataset):
    """
    PyTorch Dataset for multi-step ultrasound wave prediction using HDF5 files.
    Supports:
      - Sliding window extraction of sequences
      - Multi-step input/output (e.g., use N steps to predict M steps)
      - Optional noise augmentation (multiplicative + additive)
    """

    def __init__(
        self,
        folder_path: str,
        mult_input: int = 2,
        mult_output: int = 7,
        stride: int = 1,
        noise_enable: bool = True,
        mul_noise_level: float = 0.02,
        add_noise_level: float = 1e-6,
        pressure_std: float = 1.0,
        density_soundspeed_enable: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize the dataset and index all available samples.

        Args:
            folder_path (str): Directory containing HDF5 simulation files.
            mult_input (int): Number of input time steps.
            mult_output (int): Number of output (prediction) time steps.
            stride (int): Step size for sliding window.
            noise_enable (bool): Whether to apply noise to the inputs.
            mul_noise_level (float): Multiplicative noise strength.
            add_noise_level (float): Additive noise strength.
            pressure_std (float): Standard deviation of pressure values for normalization.
            density_soundspeed_enable (bool):  Whether to add density and sound speed to the inputs.
            device (str or torch.device): Device on which tensors should be created.
        """
        self.folder_path = folder_path
        self.mult_input = mult_input
        self.mult_output = mult_output
        self.stride = stride
        self.noise_enable = noise_enable
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_std = pressure_std
        self.density_soundspeed_enable = density_soundspeed_enable
        self.device = torch.device(device) if isinstance(device, str) else device

        # Normalization constants for material properties
        self.density_min = 1.38
        self.density_max = 5500
        self.sound_speed_min = 150
        self.sound_speed_max = 5400

        # Load all valid .h5 files
        self.h5_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")])

        # Precompute the sample indices for all files
        self.sample_offsets = []
        self.total_samples = 0

        for file_path in self.h5_files:
            T = self._extract_time_steps(file_path)
            num_samples = (T - self.mult_output - self.mult_input) // self.stride + 1  # 修改這一行
            if num_samples > 0:
                self.sample_offsets.append((self.total_samples, file_path))
                self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files.")
        print(f"Total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:

        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            return int(parts[-1].split(".")[0])
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):

        return self.total_samples

    def __getitem__(self, idx):

        for i in range(len(self.sample_offsets) - 1, -1, -1):
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx
                break

        with h5py.File(file_path, "r") as f:
            base = local_idx * self.stride

            pressure_in = torch.tensor(f["pressure"][base : base + self.mult_input], dtype=torch.float32)
            pressure_out = torch.tensor(f["pressure"][base + self.mult_input : base + self.mult_input + self.mult_output], dtype=torch.float32)

            # === Set values with abs < threshold to a very small number ===
            threshold = 0.02
            eps = 1e-9
            pressure_in[torch.abs(pressure_in) < threshold] = eps
            pressure_out[torch.abs(pressure_out) < threshold] = eps

            # === Add Gaussian noise if enabled ===
            if self.noise_enable:
                mul_noise = torch.normal(mean=1.0, std=self.mul_noise_level, size=pressure_in.shape, device=pressure_in.device)
                add_noise = torch.normal(mean=0.0, std=self.add_noise_level, size=pressure_in.shape, device=pressure_in.device)
                pressure_in *= mul_noise
                pressure_in += add_noise

            pressure_in /= self.pressure_std
            pressure_out /= self.pressure_std

            # === Load density and sound speed if enabled ===
            if self.density_soundspeed_enable:
                density = torch.tensor(f["density"][:], dtype=torch.float32).unsqueeze(0)
                sound_speed = torch.tensor(f["sound_speed"][:], dtype=torch.float32).unsqueeze(0)
                density = (density - self.density_min) / (self.density_max - self.density_min)
                sound_speed = (sound_speed - self.sound_speed_min) / (self.sound_speed_max - self.sound_speed_min)
                X = torch.cat([pressure_in, density, sound_speed], dim=0)
                Y = pressure_out
            else:
                X = pressure_in
                Y = pressure_out
                
            # # ===== Random rotation angle (0°, 90°, 180°, 270°) =====
            # k = random.randint(0, 3)  # k=0: 0°, k=1: 90°, ...
            # if k > 0:
            #     X = torch.rot90(X, k=k, dims=[1, 2])
            #     Y = torch.rot90(Y, k=k, dims=[1, 2])

        return X, Y

class HDF5MapDataset_post_process(Dataset):
    """
    PyTorch Dataset for multi-step ultrasound wave prediction using HDF5 files.
    Supports:
      - Sliding window extraction of sequences
      - Multi-step input/output (e.g., use N steps to predict M steps)
      - Optional noise augmentation (multiplicative + additive)
    """

    def __init__(
        self,
        folder_path: str,
        time_size: int = 2,
        stride: int = 1,
        noise_enable: bool = True,
        mul_noise_level: float = 0.02,
        add_noise_level: float = 1e-6,
        pressure_std: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize the dataset and index all available samples.
        """
        self.folder_path = folder_path
        self.time_size = time_size
        self.stride = stride
        self.noise_enable = noise_enable
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_std = pressure_std
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load all valid .h5 files
        self.h5_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")])

        # Precompute the sample indices for all files
        self.sample_offsets = []
        self.total_samples = 0

        for file_path in self.h5_files:
            T = self._extract_time_steps(file_path)
            num_samples = (T - self.time_size) // self.stride + 1  # 修改這一行
            if num_samples > 0:
                self.sample_offsets.append((self.total_samples, file_path))
                self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files.")
        print(f"Total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:

        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            return int(parts[-1].split(".")[0])
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):

        return self.total_samples

    def __getitem__(self, idx):

        for i in range(len(self.sample_offsets) - 1, -1, -1):
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx
                break

        with h5py.File(file_path, "r") as f:
            base = local_idx * self.stride

            pressure_in = torch.tensor(f["pressure_pred"][base : base + self.time_size], dtype=torch.float32)
            pressure_out = torch.tensor(f["pressure_gt"][base : base + self.time_size], dtype=torch.float32)

            # # === Set values with abs < threshold to a very small number ===
            # threshold = 0.02
            # eps = 1e-9
            # pressure_in[torch.abs(pressure_in) < threshold] = eps
            # pressure_out[torch.abs(pressure_out) < threshold] = eps
            
            # === Add Gaussian noise if enabled ===
            if self.noise_enable:
                mul_noise = torch.normal(mean=1.0, std=self.mul_noise_level, size=pressure_in.shape, device=pressure_in.device)
                add_noise = torch.normal(mean=0.0, std=self.add_noise_level, size=pressure_in.shape, device=pressure_in.device)
                pressure_in *= mul_noise
                pressure_in += add_noise

            pressure_in /= self.pressure_std
            pressure_out /= self.pressure_std


        return pressure_in, pressure_out