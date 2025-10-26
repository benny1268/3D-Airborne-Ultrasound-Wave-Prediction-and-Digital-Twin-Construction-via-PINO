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
# 2D U-FNO
# ===================================================================
# ===================================================================

class FNO2DEncoder(nn.Module):
    """2D Spectral encoder for U-FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_ufno_layers : int, optional
        Number of UFNO layers with U-shaped enhancements, by default 2
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
        num_ufno_layers: int = 3,
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
        self.num_ufno_layers = num_ufno_layers
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
        self.build_ufno(num_fno_modes)

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
        # Build Neural Fourier Operators
        self.spconv_layers_fno = nn.ModuleList()
        self.conv_layers_fno = nn.ModuleList()
        self.mlp_layers_fno = nn.ModuleList()

        for _ in range(self.num_fno_layers):
            self.spconv_layers_fno.append(
                layers.SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers_fno.append(nn.Conv2d(self.fno_width, self.fno_width, 1))
            self.mlp_layers_fno.append(
                nn.Sequential(
                    layers.Conv2dFCLayer(self.fno_width, self.fno_width*2),
                    self.activation_fn,
                    layers.Conv2dFCLayer(self.fno_width*2, self.fno_width),
                )
            )

    def build_ufno(self, num_fno_modes: List[int]) -> None:
        self.spconv_layers_ufno = nn.ModuleList()
        self.conv_layers_ufno = nn.ModuleList()
        self.ufno_blocks = nn.ModuleList()
        self.mlp_layers_ufno = nn.ModuleList()

        for _ in range(self.num_ufno_layers):
            self.spconv_layers_fno.append(
                layers.SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers_fno.append(nn.Conv2d(self.fno_width, self.fno_width, 1))
            self.ufno_blocks.append(
                UNet2D(
                    self.fno_width,
                    self.fno_width,
                )
            )
            self.mlp_layers_ufno.append(
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
            coord_feat = self.meshgrid(list(x.shape), x.device)  # shape: [B, 2, H, W]
            x = torch.cat((x, coord_feat), dim=1)  # [B, 4, H, W]

        # === Lift network  ===
        x = self.lift_network(x)  # [B, fno_width/2, H, W]

        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, (conv, w, mlp)  in enumerate(zip(self.conv_layers_fno, self.spconv_layers_fno, self.mlp_layers_fno)):
            x = mlp(conv(x)) + w(x)

        for k, (conv, w, u_block, mlp) in enumerate(zip(self.conv_layers_ufno, self.spconv_layers_ufno, self.ufno_blocks, self.mlp_layers_ufno)):
            x = mlp(conv(x)) + w(x) + u_block(x)

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


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256]):
        super(UNet2D, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder pathway
        for feature in features:
            self.encoder_blocks.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder pathway
        for feature in reversed(features):
            self.upsample_blocks.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(self.double_conv(feature * 2, feature))

        # Final output layer
        self.output_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder forward
        for i, (upsample, decode_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            x = upsample(x)
            skip_conn = skip_connections[i]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])

            x = torch.cat((skip_conn, x), dim=1)
            x = decode_block(x)

        return self.output_layer(x)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )



# ===================================================================
# ===================================================================
# 3D U-FNO
# ===================================================================
# ===================================================================


class FNO3DEncoder(nn.Module):
    """3D Spectral encoder for U-FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_ufno_layers : int, optional
        Number of UFNO layers with U-shaped enhancements, by default 0
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
        num_ufno_layers: int = 3,
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
        self.num_ufno_layers = num_ufno_layers
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
        self.build_ufno(num_fno_modes)

    def build_lift_network(self) -> None:
        self.lift_network = nn.Sequential(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2)),
            self.activation_fn,
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width),
        )

    def build_fno(self, num_fno_modes: List[int]) -> None:
        self.spconv_layers_fno = nn.ModuleList()
        self.conv_layers_fno = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers_fno.append(
                layers.SpectralConv3d(
                    self.fno_width, self.fno_width,
                    num_fno_modes[0], num_fno_modes[1], num_fno_modes[2],
                )
            )
            self.conv_layers_fno.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

    def build_ufno(self, num_fno_modes: List[int]) -> None:
        self.spconv_layers_ufno = nn.ModuleList()
        self.conv_layers_ufno = nn.ModuleList()
        self.ufno_blocks = nn.ModuleList()

        for _ in range(self.num_ufno_layers):
            self.spconv_layers_ufno.append(
                layers.SpectralConv3d(
                    self.fno_width, self.fno_width,
                    num_fno_modes[0], num_fno_modes[1], num_fno_modes[2],
                )
            )
            self.conv_layers_ufno.append(nn.Conv3d(self.fno_width, self.fno_width, 1))
            self.ufno_blocks.append(
                UNet3D(
                    self.fno_width,
                    self.fno_width,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        x = F.pad(x, (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)

        for k, (conv, w) in enumerate(zip(self.conv_layers_fno, self.spconv_layers_fno)):
            x = self.activation_fn(conv(x) + w(x))

        for k, (conv, w, u_block) in enumerate(zip(self.conv_layers_ufno, self.spconv_layers_ufno, self.ufno_blocks)):
            if k < len(self.conv_layers_ufno) - 1:
                x = self.activation_fn(conv(x) + w(x) + u_block(x))
            else:
                x = conv(x) + w(x) + u_block(x)

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

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128]):
        super(UNet3D, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Encoder pathway
        for feature in features:
            self.encoder_blocks.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder pathway
        for feature in reversed(features):
            self.upsample_blocks.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                self.double_conv(feature * 2, feature)
            )

        # Final output layer
        self.output_layer = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder forward
        for i, (upsample, decode_block) in enumerate(zip(self.upsample_blocks, self.decoder_blocks)):
            x = upsample(x)
            skip_conn = skip_connections[i]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat((skip_conn, x), dim=1)
            x = decode_block(x)

        return self.output_layer(x)

    @staticmethod
    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

# ===================================================================
# ===================================================================
# U-FNO Model
# ===================================================================
# ===================================================================

@dataclass
class MetaData(ModelMetaData):
    name: str = "U_shape_FourierNeuralOperator"
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


class U_FNO(Module):
    """U-Fourier Neural Operator (U-FNO) model.

    Description
    -----------
    The U-FNO architecture extends the original Fourier Neural Operator (FNO) by
    incorporating U-Net-style skip connections or low-resolution pathways in spectral
    convolutional layers. This enhancement enables the model to better capture both
    global and local features, improving performance in complex physical systems.

    U-FNO is particularly effective for problems involving heterogeneity, anisotropy,
    and nonlinear dynamics, such as multiphase flow in porous media or high-contrast
    acoustic fields.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Dimensionality of input/output fields (supports 1, 2, 3D)
    latent_channels : int, optional
        Latent feature size in spectral layers, by default 32
    num_fno_layers : int, optional
        Number of standard Fourier layers before switching to UFNO layers, by default 3
    num_ufno_layers : int, optional
        Number of UFNO layers with U-shaped enhancements, by default 3
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes retained in spectral convolutions, by default 16
    padding : int, optional
        Padding size for spectral convolution, by default 8
    padding_type : str, optional
        Padding type, such as "constant" or "reflect", by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Whether to append coordinate grid features to the input, by default True

    Example
    -------
    >>> model = U_FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=64,
    ...     dimension=2,
    ...     latent_channels=48,
    ...     num_fno_layers=2,
    ...     num_ufno_layers=2,
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

    [2] Wen, Gege, et al. "U-FNO -- An enhanced Fourier neural operator-based deep-learning model for multiphase flow."
        arXiv:2109.01001v3 (2022).
    """


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 128,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 3,
        num_ufno_layers: int = 3,
        num_fno_modes: Union[int, List[int]] = 32,
        padding: int = 0,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
    ) -> None:
        super().__init__(meta=MetaData())

        self.num_fno_layers = num_fno_layers
        self.num_ufno_layers = num_ufno_layers
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
            num_ufno_layers=self.num_ufno_layers,
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
