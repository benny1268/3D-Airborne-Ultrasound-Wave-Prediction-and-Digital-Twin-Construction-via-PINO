import os
import h5py
import torch
from typing import Union
from torch.utils.data import Dataset

class HDF5MapDataset_full_multstep(Dataset):
    """
    Sliding window dataset for HDF5 data, with support for single-sample access and noise injection.
    """

    def __init__(
        self,
        file_path: str,
        window_size: int = 2,
        move_size: int = 1,
        mult_step: int = 7,
        device: Union[str, torch.device] = "cuda",
        add_noise: bool = True,
        mul_noise_level: float = 0.2,
        add_noise_level: float = 0.2,
        pressure_mean: float = 0,
        pressure_std: float = 1.0,  
    ):
        """
        Initialize dataset with file path, sliding window settings, prediction steps, and noise levels.

        Args:
            file_path (str): Path to the HDF5 file.
            window_size (int): Sliding window size (unused in current logic).
            move_size (int): Step size for sliding window.
            pred_time (int): Number of time steps to predict.
            device (str or torch.device): Device to load data on.
            add_noise (bool): Whether to apply noise to input.
            mul_noise_level (float): Multiplicative noise level.
            add_noise_level (float): Additive noise level.
        """
        self.file_path = file_path
        self.window_size = window_size
        self.move_size = move_size
        self.mult_step = mult_step
        self.device = torch.device(device) if isinstance(device, str) else device
        self.add_noise = add_noise
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.density_min = 1.38
        self.density_max = 5500
        self.sound_speed_min = 150
        self.sound_speed_max = 5400
        self.num_segments = None
        self.total_samples = None

        print(f"Dataset loaded from {file_path}.")

    def __len__(self):
        """
        Compute total number of sliding window samples across the dataset.

        Returns:
            int: Total number of samples.
        """
        if self.num_segments is None:
            self._load_h5_file()
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve a sample from HDF5 by index, applying window slicing and noise if enabled.

        Args:
            idx (int): Global sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor and output tensor.
        """
        with h5py.File(self.file_path, "r") as f:
            segment_idx = idx % self.num_segments
            sample_idx = idx // self.num_segments
            input_idx = segment_idx * self.move_size

            pressure_in = torch.tensor(f["pressure"][sample_idx, input_idx], dtype=torch.float32).unsqueeze(0)
            pressure_out = torch.tensor(f["pressure"][sample_idx, input_idx+1:input_idx+1+self.mult_ste], dtype=torch.float32)
            density = torch.tensor(f["density"][sample_idx], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][sample_idx], dtype=torch.float32).unsqueeze(0)

            pressure_in = pressure_in  / self.pressure_std
            pressure_out = pressure_out  / self.pressure_std

            # Min-Max Normalization to [0, 1]
            density = (density - self.density_min) / (self.density_max - self.density_min)
            sound_speed = (sound_speed - self.sound_speed_min) / (self.sound_speed_max - self.sound_speed_min)

            # Concatenate inputs along channel dimension
            X = torch.cat([pressure_in, density, sound_speed], dim=0)
            Y = pressure_out

            # Add noise to pressure input
            if self.add_noise:
                # Apply multiplicative noise
                mul_noise = 1.0 + ((torch.rand_like(X[0]) - 0.5) * 2.0 * self.mul_noise_level)
                X[0] *= mul_noise
                # Apply additive noise
                add_noise = (torch.rand_like(X[0]) - 0.5) * 2.0 * self.add_noise_level
                X[0] += add_noise
                
            # Move to device if CUDA
            if self.device.type == "cuda":
                invar = invar.cuda()
                outvar = outvar.cuda()

            return X.to(torch.float32), Y.to(torch.float32)

    def _load_h5_file(self):
        """
        Internal helper to calculate sample and segment counts.
        """
        with h5py.File(self.file_path, "r") as file:
            pressure_shape = file["pressure"].shape
            self.num_time_steps = pressure_shape[1]
            self.num_segments = self.num_time_steps - self.mult_step
            self.total_samples = len(file["pressure"]) * self.num_segments
            print(f"num_segments: {self.num_segments}, total_samples: {self.total_samples}")

    def __del__(self):
        """
        Clean up if needed (currently no persistent resources).
        """
        pass

class HDF5MapDataset_patch_multstep(Dataset):
    """
    HDF5 dataset with support for multi-step prediction and optional noise augmentation.
    Loads multiple HDF5 files and performs sliding window extraction.
    """

    def __init__(
        self,
        folder_path: str,
        mult_step: int = 5,
        device: Union[str, torch.device] = "cuda",
        add_noise: bool = True,
        mul_noise_level: float = 0.5,
        add_noise_level: float = 0.5,
        pressure_mean: float = 0,
        pressure_std: float = 1.0,        
    ):
        """
        Initialize the dataset and precompute sample indices.

        Args:
            folder_path (str): Path to folder containing HDF5 files.
            mult_step (int): Number of steps to predict (multi-step output).
            device (str or torch.device): Device to load data on ('cuda' or 'cpu').
            add_noise (bool): Whether to add noise to the input.
            noise_level (float): Level of multiplicative noise to add.
        """
        self.folder_path = folder_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mult_step = mult_step
        self.add_noise = add_noise
        self.mul_noise_level = mul_noise_level
        self.add_noise_level = add_noise_level
        self.pressure_mean = pressure_mean
        self.pressure_std = pressure_std
        self.density_min = 1.38
        self.density_max = 5500
        self.sound_speed_min = 150
        self.sound_speed_max = 5400

        self.h5_files = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")
        ])

        self.sample_offsets = []  # Stores (start index, file path) for each file
        self.total_samples = 0

        for file_path in self.h5_files:
            T = self._extract_time_steps(file_path)
            num_samples = T - self.mult_step
            if num_samples > 0:
                self.sample_offsets.append((self.total_samples, file_path))
                self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files, total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:
        """
        Extract the total number of time steps from the filename.

        Args:
            file_path (str): Full path to the HDF5 file.

        Returns:
            int: Number of time steps inferred from the filename.
        """
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            return int(parts[-1].split(".")[0])  # Extract last numeric segment
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):
        """
        Return total number of valid sliding window samples across all files.

        Returns:
            int: Total sample count.
        """
        return self.total_samples

    def __getitem__(self, idx):
        """
        Retrieve a single input/output sample pair based on global index.

        Args:
            idx (int): Global index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor X and output tensor Y.
        """
        # Efficient reverse lookup for which file the sample belongs to
        for i in range(len(self.sample_offsets) - 1, -1, -1):
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx
                break

        # Load data from file
        with h5py.File(file_path, "r") as f:
            pressure_in = torch.tensor(f["pressure"][local_idx], dtype=torch.float32).unsqueeze(0) 
            pressure_out = torch.tensor(f["pressure"][local_idx+1:local_idx+1+self.mult_step], dtype=torch.float32)

            pressure_in = pressure_in  / self.pressure_std
            pressure_out = pressure_out  / self.pressure_std

            density = torch.tensor(f["density"][:], dtype=torch.float32).unsqueeze(0)
            sound_speed = torch.tensor(f["sound_speed"][:], dtype=torch.float32).unsqueeze(0)

            # Min-Max Normalization to [0, 1]
            density = (density - self.density_min) / (self.density_max - self.density_min)
            sound_speed = (sound_speed - self.sound_speed_min) / (self.sound_speed_max - self.sound_speed_min)

            # Concatenate inputs along channel dimension
            X = torch.cat([pressure_in, density, sound_speed], dim=0)
            Y = pressure_out

            if self.add_noise:
                # Apply multiplicative noise
                mul_noise = 1.0 + ((torch.rand_like(X[0]) - 0.5) * 2.0 * self.mul_noise_level)
                X[0] *= mul_noise
                # Apply additive noise
                add_noise = (torch.rand_like(X[0]) - 0.5) * 2.0 * self.add_noise_level
                X[0] += add_noise

        return X, Y
