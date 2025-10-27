from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
import physicsnemo.models.layers as layers

from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.module import Module

# ===================================================================
# ===================================================================
# 1D FNO
# ===================================================================
# ===================================================================


class FNO1DEncoder(nn.Module):
    """1D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.activation_fn = activation_fn

        # Add relative coordinate feature
        self.coord_features = coord_features
        if self.coord_features:
            self.in_channels = self.in_channels + 1

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], value.size(-1))
        return torch.permute(output, (0, 2, 1))


# ===================================================================
# ===================================================================
# 2D F-FNO
# ===================================================================
# ===================================================================

class FNO2DEncoder(nn.Module):
    """2D Spectral encoder for F-FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 3,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]

        # build lift
        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        """construct network for lifting variables to latent space."""
        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions
        """
        self.spconv_layers_fno = nn.ModuleList()
        self.mlp_layers_fno = nn.ModuleList()

        for _ in range(self.num_fno_layers):
            self.spconv_layers_fno.append(
                SpectralConv2d_FFNO(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )

            self.mlp_layers_fno.append(
                nn.Sequential(
                    layers.Conv2dFCLayer(self.fno_width, self.fno_width*2),
                    self.activation_fn,
                    layers.Conv2dFCLayer(self.fno_width*2, self.fno_width),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)

        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)

        # Spectral layers
        for _, (conv, mlp)  in enumerate(zip(self.spconv_layers_fno, self.mlp_layers_fno)):
            x = mlp(conv(x)) + x

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        return torch.permute(output, (0, 3, 1, 2))

class SpectralConv2d_FFNO(nn.Module):
    """Factorized 2D Fourier layer (F-FNO style)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weight_x = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, 2))
        self.weight_y = nn.Parameter(torch.empty(in_channels, out_channels, self.modes2, 2))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_x.data = self.scale * torch.rand_like(self.weight_x)
        self.weight_y.data = self.scale * torch.rand_like(self.weight_y)

    def forward(self, x):
        # x: (B, C, X, Y)
        B, C, X, Y = x.shape
        out = 0

        # === Y-Direction ===
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (B, C, X, Y//2+1)
        out_ft = torch.zeros(B, self.out_channels, X, Y//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes2] = torch.einsum("bixy,ioy->boxy", x_ft[..., :self.modes2], torch.view_as_complex(self.weight_y))
        out += torch.fft.irfft(out_ft, n=Y, dim=-1, norm='ortho') # (B, C, X, Y)

        # === X-Direction ===
        x_ft = torch.fft.rfft(x, dim=-2, norm='ortho') # (B, C, X//2+1, Y)
        out_ft = torch.zeros(B, self.out_channels, X//2 + 1, Y, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :] = torch.einsum("bixy,iox->boxy", x_ft[:, :, :self.modes1, :], torch.view_as_complex(self.weight_x))
        out += torch.fft.irfft(out_ft, n=X, dim=-2, norm='ortho') # (B, C, X, Y)

        return out


# ===================================================================
# ===================================================================
# 3D F-FNO
# ===================================================================
# ===================================================================


class FNO3DEncoder(nn.Module):
    """3D Spectral encoder for F-FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 6,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 32,
        padding: Union[int, List[int]] = 0,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.activation_fn = activation_fn

        if self.coord_features:
            self.in_channels = self.in_channels + 3

        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]

        self.build_lift_network()
        self.build_fno(num_fno_modes)

    def build_lift_network(self) -> None:
        self.lift_network = nn.Sequential(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2)),
            self.activation_fn,
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width),
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions
        """
        self.spconv_layers_fno = nn.ModuleList()
        self.mlp_layers_fno = nn.ModuleList()

        for _ in range(self.num_fno_layers):
            self.spconv_layers_fno.append(
                SpectralConv3d_FFNO(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1], num_fno_modes[2]
                )
            )

            self.mlp_layers_fno.append(
                nn.Sequential(
                    layers.Conv3dFCLayer(self.fno_width, self.fno_width*2),
                    self.activation_fn,
                    layers.Conv3dFCLayer(self.fno_width*2, self.fno_width),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                "Only 5D tensors [batch, in_channels, grid_x, grid_y, grid_z] accepted for 3D FNO"
            )
        
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)

        # Spectral layers
        for _, (conv, mlp)  in enumerate(zip(self.spconv_layers_fno, self.mlp_layers_fno)):
            x = mlp(conv(x)) + x

        x = x[..., :self.ipad[0], :self.ipad[1], :self.ipad[2]]
        return x


    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)

    def grid_to_points(self, value: Tensor) -> Tuple[Tensor, List[int]]:
        """converting from grid based (image) to point based representation

        Parameters
        ----------
        value : Meshgrid tensor

        Returns
        -------
        Tuple
            Tensor, meshgrid shape
        """
        y_shape = list(value.size())
        output = torch.permute(value, (0, 2, 3, 4, 1))
        return output.reshape(-1, output.size(-1)), y_shape

    def points_to_grid(self, value: Tensor, shape: List[int]) -> Tensor:
        """converting from point based to grid based (image) representation

        Parameters
        ----------
        value : Tensor
            Tensor
        shape : List[int]
            meshgrid shape

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        return torch.permute(output, (0, 4, 1, 2, 3))


class SpectralConv3d_FFNO(nn.Module):
    """Factorized 3D Fourier layer (F-FNO style)."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weight_x = nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, 2))
        self.weight_y = nn.Parameter(torch.empty(in_channels, out_channels, self.modes2, 2))
        self.weight_z = nn.Parameter(torch.empty(in_channels, out_channels, self.modes3, 2))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_x.data = self.scale * torch.rand_like(self.weight_x)
        self.weight_y.data = self.scale * torch.rand_like(self.weight_y)
        self.weight_z.data = self.scale * torch.rand_like(self.weight_z)


    def forward(self, x):
        # x: (B, C_in, X, Y, Z)
        B, C, X, Y, Z = x.shape
        out = 0

        # === Z-Direction ===
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (B, C, X, Y, Z//2+1)
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes3] = torch.einsum("bixyz,ioz->boxyz", x_ft[..., :self.modes3], torch.view_as_complex(self.weight_z))
        out += torch.fft.irfft(out_ft, n=Z, dim=-1, norm='ortho')

        # === Y-Direction ===
        x_ft = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = torch.zeros(B, self.out_channels, X, Y//2 + 1, Z, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes2, :] = torch.einsum("bixyz,ioy->boxyz", x_ft[:, :, :, :self.modes2, :], torch.view_as_complex(self.weight_y))
        out += torch.fft.irfft(out_ft, n=Y, dim=-2, norm='ortho')

        # === X-Direction ===
        x_ft = torch.fft.rfft(x, dim=-3, norm='ortho')
        out_ft = torch.zeros(B, self.out_channels, X//2 + 1, Y, Z, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :, :] = torch.einsum("bixyz,iox->boxyz", x_ft[:, :, :self.modes1, :, :], torch.view_as_complex(self.weight_x))
        out += torch.fft.irfft(out_ft, n=X, dim=-3, norm='ortho')

        return out


# ===================================================================
# ===================================================================
# F-FNO Model
# ===================================================================
# ===================================================================

@dataclass
class MetaData(ModelMetaData):
    name: str = "Factorized_FourierNeuralOperator"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class F_FNO(Module):
    """
    Factorized Fourier Neural Operator (F-FNO) model.

    Description
    -----------
    F-FNO is an extension of the original Fourier Neural Operator (FNO) architecture
    that introduces factorization within the spectral convolution layers to improve
    computational efficiency and generalization. By decomposing the Fourier modes
    along separate axes or channels, F-FNO reduces memory usage and computational
    overhead while maintaining the ability to capture long-range dependencies.

    This model is particularly suited for learning complex physical dynamics in
    high-dimensional PDEs, especially where fine spatial resolution and large
    computational domains make traditional FNO models costly or memory-intensive.

    F-FNO retains the core strengths of FNO (global receptive field, spectral learning)
    while improving scalability for real-world applications such as fluid dynamics,
    wave propagation, and climate modeling.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    decoder_layers : int, optional
        Number of layers in the decoder MLP, by default 1.
    decoder_layer_size : int, optional
        Number of hidden units in each decoder layer, by default 32.
    decoder_activation_fn : str, optional
        Activation function used in the decoder, by default "silu".
    dimension : int
        Dimensionality of the spatial domain (1D, 2D, or 3D).
    latent_channels : int, optional
        Number of latent feature channels used throughout spectral layers, by default 32.
    num_fno_layers : int, optional
        Number of spectral layers, by default 3.
    num_ufno_layers : int, optional
        Reserved for compatibility with U-FNO; unused in F-FNO, by default 3.
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes retained in spectral convolutions, by default 32.
    padding : int, optional
        Amount of zero padding applied to the input domain, by default 0.
    padding_type : str, optional
        Type of padding to use, e.g., "constant", "reflect", by default "constant".
    activation_fn : str, optional
        Activation function applied between layers, by default "gelu".
    coord_features : bool, optional
        Whether to append normalized coordinate grids to input channels, by default True.

    Example
    -------
    >>> model = F_FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=1,
    ...     decoder_layer_size=128,
    ...     dimension=2,
    ...     latent_channels=64,
    ...     num_fno_layers=4,
    ...     num_fno_modes=[16, 16],
    ...     padding=4,
    ... )
    >>> input = torch.randn(16, 4, 64, 64)
    >>> output = model(input)
    >>> output.shape
    torch.Size([16, 3, 64, 64])

    References
    ----------
    [1] Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations."
        arXiv:2010.08895 (2020).

    [2] Tran, Dat, et al. "Factorized Fourier Neural Operators."
        arXiv:2302.07944 (2023).
    """


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 3,
        num_fno_modes: Union[int, List[int]] = 32,
        padding: int = 0,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
    ) -> None:
        super().__init__(meta=MetaData())

        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features
        self.dimension = dimension

        # Fully connected decoder applied pointwise
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )

        # Select encoder class by dimension
        FNOModel = self.getFNOEncoder()

        # Spectral encoder: will switch from FNO to UFNO after num_fno_layers
        self.spec_encoder = FNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
        )

    def getFNOEncoder(self):
        if self.dimension == 1:
            return FNO1DEncoder
        elif self.dimension == 2:
            return FNO2DEncoder
        elif self.dimension == 3:
            return FNO3DEncoder
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D and 3D U-FNO implemented."
            )

    def forward(self, x: Tensor) -> Tensor:
        # Spectral encoder
        y_latent = self.spec_encoder(x)

        # Convert from grid to pointwise if using FC decoder
        y_shape = y_latent.shape
        y_latent, y_shape = self.spec_encoder.grid_to_points(y_latent)

        # Pointwise decoder
        y = self.decoder_net(y_latent)

        # Convert back to grid format
        y = self.spec_encoder.points_to_grid(y, y_shape)

        return y
