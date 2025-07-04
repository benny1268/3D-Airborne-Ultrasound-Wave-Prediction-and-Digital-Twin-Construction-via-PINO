import os
import h5py
import torch
from typing import Union
from torch.utils.data import Dataset

class HDF5MapDataset_full_multstep(Dataset):
    """
    HDF5 dataset with support for multi-step prediction on full-field (entire spatial domain) data.
    Supports multi-step input/output and optional noise augmentation.
    """

    def __init__(
        self,
        file_path: str,
        mult_input: int = 2,
        mult_output: int = 7,
        move_size: int = 1,
        noise_enable: bool = True,
        mul_noise_level: float = 0.2,
        add_noise_level: float = 0.2,
        pressure_mean: float = 0,
        pressure_std: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize the dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            mult_input (int): Number of input time steps.
            mult_output (int): Number of output time steps.
            move_size (int): Step size for sliding window.
            noise_enable (bool): Whether to add noise to pressure input.
            mul_noise_level (float): Multiplicative noise level.
            add_noise_level (float): Additive noise level.
            pressure_mean (float): Mean of pressure values (not used).
            pressure_std (float): Std of pressure values for normalization.
            device (str or torch.device): Computation device.
        """
        self.file_path = file_path
        self.mult_input = mult_input
        self.mult_output = mult_output
        self.move_size = move_size
        self.noise_enable = noise_enable
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.device = torch.device(device) if isinstance(device, str) else device

        self.density_min = 1.38
        self.density_max = 5500
        self.sound_speed_min = 150
        self.sound_speed_max = 5400

        self.num_segments = None
        self.total_samples = None

        print(f"Dataset loaded from {file_path}.")

    def __len__(self):
        """
        Return total number of sliding window samples.
        """
        if self.num_segments is None:
            self._load_h5_file()
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve one full-field input/output pair with optional noise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (X, Y)
            X shape: (mult_input + 2, H, W)  # input pressure + density + sound speed
            Y shape: (mult_output, H, W)     # future pressure
        """
        with h5py.File(self.file_path, "r") as f:
            segment_idx = idx % self.num_segments
            sample_idx = idx // self.num_segments
            input_idx = segment_idx * self.move_size

            pressure_all = torch.tensor(f["pressure"][sample_idx], dtype=torch.float32)  # (T, H, W)
            pressure_in = pressure_all[input_idx : input_idx + self.mult_input]  # (mult_input, H, W)
            pressure_out = pressure_all[input_idx + self.mult_input : input_idx + self.mult_input + self.mult_output]  # (mult_output, H, W)

            # Normalize pressure
            pressure_in = pressure_in / self.pressure_std
            pressure_out = pressure_out / self.pressure_std

            # Load and normalize static fields
            density = torch.tensor(f["density"][sample_idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][sample_idx], dtype=torch.float32).unsqueeze(0)

            density = (density - self.density_min) / (self.density_max - self.density_min)
            sound_speed = (sound_speed - self.sound_speed_min) / (self.sound_speed_max - self.sound_speed_min)

            # Combine channels: mult_input pressure + 1 density + 1 sound speed
            X = torch.cat([pressure_in, density, sound_speed], dim=0)
            Y = pressure_out

            # Apply optional noise only to pressure input
            if self.noise_enable:
                mul_noise = 1.0 + ((torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.mul_noise_level)
                X[:self.mult_input] *= mul_noise

                add_noise = (torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.add_noise_level
                X[:self.mult_input] += add_noise

            return X.to(torch.float32).to(self.device), Y.to(torch.float32).to(self.device)

    def _load_h5_file(self):
        """
        Precompute number of valid sliding windows and total samples.
        """
        with h5py.File(self.file_path, "r") as f:
            pressure_shape = f["pressure"].shape  # (N, T, H, W)
            T = pressure_shape[1]
            self.num_segments = T - self.mult_input - self.mult_output + 1
            self.total_samples = pressure_shape[0] * self.num_segments
            print(f"num_segments: {self.num_segments}, total_samples: {self.total_samples}")

    def __del__(self):
        """
        Destructor: No persistent resource cleanup needed.
        """
        pass


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
        noise_enable: bool = True,
        mul_noise_level: float = 0.5,
        add_noise_level: float = 0.5,
        pressure_mean: float = 0,
        pressure_std: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize the dataset and index all available samples.

        Args:
            folder_path (str): Directory containing HDF5 simulation files.
            mult_input (int): Number of input time steps.
            mult_output (int): Number of output (prediction) time steps.
            noise_enable (bool): Whether to apply noise to the inputs.
            mul_noise_level (float): Multiplicative noise strength (e.g., 0.5 means ±50%).
            add_noise_level (float): Additive noise strength (e.g., 0.5 means ±0.5).
            pressure_mean (float): Mean of pressure values (not used here).
            pressure_std (float): Standard deviation of pressure values for normalization.
            device (str or torch.device): Device on which tensors should be created.
        """
        self.folder_path = folder_path
        self.mult_input = mult_input
        self.mult_output = mult_output
        self.noise_enable = noise_enable
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.device = torch.device(device) if isinstance(device, str) else device

        # Normalization constants for material properties
        self.density_min = 1.38
        self.density_max = 5500
        self.sound_speed_min = 150
        self.sound_speed_max = 5400

        # Load all valid .h5 files
        self.h5_files = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")
        ])

        # Precompute the sample indices for all files
        self.sample_offsets = []
        self.total_samples = 0

        for file_path in self.h5_files:
            T = self._extract_time_steps(file_path)
            # A sample is valid if it has enough steps for both input and output
            num_samples = T - self.mult_output - self.mult_input + 1
            if num_samples > 0:
                self.sample_offsets.append((self.total_samples, file_path))
                self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files, total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:
        """
        Extract the number of time steps from the HDF5 filename.

        Assumes filename format: <something>_<time>.h5 (e.g., sim_170.h5 -> T=170)

        Returns:
            int: Number of time steps.
        """
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            return int(parts[-1].split(".")[0])
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):
        """
        Returns the total number of sliding window samples available.
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        Returns one sample (X, Y) given a global sample index.

        X shape: (mult_input + 2, H, W)  # mult_input pressure steps + density + sound_speed
        Y shape: (mult_output, H, W)
        """
        # Find which file the idx belongs to
        for i in range(len(self.sample_offsets) - 1, -1, -1):
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx
                break

        with h5py.File(file_path, "r") as f:
            # Load pressure data: input sequence and target sequence
            pressure_in = torch.tensor(f["pressure"][local_idx : local_idx + self.mult_input], dtype=torch.float32)
            pressure_out = torch.tensor(f["pressure"][local_idx + self.mult_input : local_idx + self.mult_input + self.mult_output], dtype=torch.float32)

            # Normalize pressure
            pressure_in /= self.pressure_std
            pressure_out /= self.pressure_std

            # Load and normalize density and sound speed maps
            density = torch.tensor(f["density"][:], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][:], dtype=torch.float32).unsqueeze(0)
            density = (density - self.density_min) / (self.density_max - self.density_min)
            sound_speed = (sound_speed - self.sound_speed_min) / (self.sound_speed_max - self.sound_speed_min)

            # Concatenate channels: [mult_input pressure steps, density, sound_speed]
            X = torch.cat([pressure_in, density, sound_speed], dim=0)
            Y = pressure_out

            if self.noise_enable:
                # Apply multiplicative noise only to pressure inputs
                mul_noise = 1.0 + ((torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.mul_noise_level)
                X[0] *= mul_noise
                # Apply additive noise
                add_noise = (torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.add_noise_level
                X[0] += add_noise

        return X, Y

class HDF5MapDataset_patch_multstep_no_dc(Dataset):
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
        noise_enable: bool = True,
        mul_noise_level: float = 0.5,
        add_noise_level: float = 0.5,
        pressure_mean: float = 0,
        pressure_std: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Initialize the dataset and index all available samples.

        Args:
            folder_path (str): Directory containing HDF5 simulation files.
            mult_input (int): Number of input time steps.
            mult_output (int): Number of output (prediction) time steps.
            noise_enable (bool): Whether to apply noise to the inputs.
            mul_noise_level (float): Multiplicative noise strength (e.g., 0.5 means ±50%).
            add_noise_level (float): Additive noise strength (e.g., 0.5 means ±0.5).
            pressure_mean (float): Mean of pressure values (not used here).
            pressure_std (float): Standard deviation of pressure values for normalization.
            device (str or torch.device): Device on which tensors should be created.
        """
        self.folder_path = folder_path
        self.mult_input = mult_input
        self.mult_output = mult_output
        self.noise_enable = noise_enable
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load all valid .h5 files
        self.h5_files = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")
        ])

        # Precompute the sample indices for all files
        self.sample_offsets = []
        self.total_samples = 0

        for file_path in self.h5_files:
            T = self._extract_time_steps(file_path)
            # A sample is valid if it has enough steps for both input and output
            num_samples = T - self.mult_output - self.mult_input + 1
            if num_samples > 0:
                self.sample_offsets.append((self.total_samples, file_path))
                self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files, total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:
        """
        Extract the number of time steps from the HDF5 filename.

        Assumes filename format: <something>_<time>.h5 (e.g., sim_170.h5 -> T=170)

        Returns:
            int: Number of time steps.
        """
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            return int(parts[-1].split(".")[0])
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):
        """
        Returns the total number of sliding window samples available.
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        Returns one sample (X, Y) given a global sample index.

        X shape: (mult_input + 2, H, W)  # mult_input pressure steps + density + sound_speed
        Y shape: (mult_output, H, W)
        """
        # Find which file the idx belongs to
        for i in range(len(self.sample_offsets) - 1, -1, -1):
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx
                break

        with h5py.File(file_path, "r") as f:
            # Load pressure data: input sequence and target sequence
            pressure_in = torch.tensor(f["pressure"][local_idx : local_idx + self.mult_input], dtype=torch.float32)
            pressure_out = torch.tensor(f["pressure"][local_idx + self.mult_input : local_idx + self.mult_input + self.mult_output], dtype=torch.float32)

            # Normalize pressure
            pressure_in /= self.pressure_std
            pressure_out /= self.pressure_std

            # Concatenate channels: [mult_input pressure steps, density, sound_speed]
            X = pressure_in
            Y = pressure_out

            if self.noise_enable:
                # Apply multiplicative noise only to pressure inputs
                mul_noise = 1.0 + ((torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.mul_noise_level)
                X[0] *= mul_noise
                # Apply additive noise
                add_noise = (torch.rand_like(X[:self.mult_input]) - 0.5) * 2.0 * self.add_noise_level
                X[0] += add_noise

        return X, Y

