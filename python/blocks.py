"""Reusable building blocks for Sharp MLX.

Contains ResidualBlock, FeatureFusionBlock2d, normalization layers, and upsampling utilities.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Literal, Optional, List


NormLayerName = Literal["noop", "batch_norm", "group_norm", "instance_norm"]
UpsamplingMode = Literal["transposed_conv", "nearest", "bilinear"]


class PyTorchGroupNorm(nn.Module):
    """Wrapper for MLX GroupNorm with pytorch_compatible flag for NHWC format."""
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.dims = num_channels
        self._affine = affine
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine, pytorch_compatible=True)
    
    @property
    def weight(self):
        return self.norm.weight if self._affine else None
    
    @weight.setter
    def weight(self, value):
        if self._affine:
            self.norm.weight = value
    
    @property
    def bias(self):
        return self.norm.bias if self._affine else None
    
    @bias.setter
    def bias(self, value):
        if self._affine:
            self.norm.bias = value
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x)


def norm_layer_2d(num_features: int, norm_type: NormLayerName, num_groups: int = 8) -> nn.Module:
    """Create normalization layer for 2D features (NHWC format)."""
    if norm_type == "noop":
        return nn.Identity()
    elif norm_type == "batch_norm":
        return nn.BatchNorm(num_features)
    elif norm_type == "group_norm":
        return PyTorchGroupNorm(num_groups, num_features)
    elif norm_type == "instance_norm":
        # InstanceNorm is GroupNorm with groups=num_features
        return PyTorchGroupNorm(num_features, num_features)
    else:
        raise ValueError(f"Invalid normalization layer type: {norm_type}")


class ConvTranspose2d(nn.Module):
    """ConvTranspose2d for MLX with NHWC format."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Weight shape: (out_channels, kernel_h, kernel_w, in_channels)
        scale = 1.0 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.weight = mx.random.normal((out_channels, kernel_size[0], kernel_size[1], in_channels)) * scale
        
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - NHWC format
        y = mx.conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y


def upsampling_layer(upsampling_mode: UpsamplingMode, scale_factor: int, dim_in: int) -> nn.Module:
    """Create upsampling layer."""
    if upsampling_mode == "transposed_conv":
        return ConvTranspose2d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=0,
            bias=False,
        )
    elif upsampling_mode in ("nearest", "bilinear"):
        return Upsample(scale_factor=scale_factor, mode=upsampling_mode)
    else:
        raise ValueError(f"Invalid upsampling mode {upsampling_mode}.")


class Upsample(nn.Module):
    """Upsampling using interpolation."""
    
    def __init__(self, scale_factor: int, mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - NHWC format
        B, H, W, C = x.shape
        new_H = H * self.scale_factor
        new_W = W * self.scale_factor
        
        if self.mode == "nearest":
            # Nearest neighbor upsampling
            x = mx.repeat(x, self.scale_factor, axis=1)
            x = mx.repeat(x, self.scale_factor, axis=2)
            return x
        elif self.mode == "bilinear":
            # Bilinear interpolation for integer scale factors
            # For 2x upscale: interpolate between adjacent pixels
            sf = self.scale_factor
            
            # Pad for edge handling
            x_pad = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)], mode='edge')
            
            # Create output
            out = mx.zeros((B, new_H, new_W, C), dtype=x.dtype)
            
            # For each output position, compute bilinear weights
            for dy in range(sf):
                for dx in range(sf):
                    # Weights for bilinear interpolation
                    wy = dy / sf
                    wx = dx / sf
                    
                    # Four corner contributions
                    top_left = x_pad[:, :H, :W, :] * (1-wy) * (1-wx)
                    top_right = x_pad[:, :H, 1:W+1, :] * (1-wy) * wx
                    bot_left = x_pad[:, 1:H+1, :W, :] * wy * (1-wx)
                    bot_right = x_pad[:, 1:H+1, 1:W+1, :] * wy * wx
                    
                    interpolated = top_left + top_right + bot_left + bot_right
                    
                    # Place at correct output positions
                    out = out.at[:, dy::sf, dx::sf, :].set(interpolated)
            
            return out
        else:
            raise ValueError(f"Unknown interpolation mode: {self.mode}")


class ResidualBlock(nn.Module):
    """Generic implementation of residual blocks.
    
    This implements a generic residual block from
    He et al. - Identity Mappings in Deep Residual Networks (2016),
    https://arxiv.org/abs/1603.05027
    """
    
    def __init__(self, residual: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut
    
    def __call__(self, x: mx.array) -> mx.array:
        delta_x = self.residual(x)
        
        if self.shortcut is not None:
            x = self.shortcut(x)
        
        return x + delta_x


class Sequential(nn.Module):
    """Sequential container for modules."""
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
    
    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class ReLU(nn.Module):
    """ReLU activation."""
    
    def __call__(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0)


def residual_block_2d(
    dim_in: int,
    dim_out: int,
    dim_hidden: Optional[int] = None,
    norm_type: NormLayerName = "noop",
    norm_num_groups: int = 8,
    dilation: int = 1,
    kernel_size: int = 3,
) -> ResidualBlock:
    """Create a simple 2D residual block."""
    if dim_hidden is None:
        dim_hidden = dim_out // 2
    
    # Padding to maintain output size
    padding = (dilation * (kernel_size - 1)) // 2
    
    def create_block(d_in: int, d_out: int) -> List[nn.Module]:
        layers = [
            norm_layer_2d(d_in, norm_type, num_groups=norm_num_groups),
            ReLU(),
            nn.Conv2d(d_in, d_out, kernel_size=kernel_size, stride=1, padding=padding),
        ]
        return layers
    
    residual = Sequential(
        *create_block(dim_in, dim_hidden),
        *create_block(dim_hidden, dim_out),
    )
    
    shortcut = None
    if dim_in != dim_out:
        shortcut = nn.Conv2d(dim_in, dim_out, kernel_size=1)
    
    return ResidualBlock(residual, shortcut)


class FeatureFusionBlock2d(nn.Module):
    """Feature fusion block for DPT-style decoders.
    
    Fuses features at different resolutions with optional upsampling.
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        upsampling_mode: Optional[UpsamplingMode] = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        
        self.resnet1 = self._residual_block(dim_in, batch_norm)
        self.resnet2 = self._residual_block(dim_in, batch_norm)
        
        if upsampling_mode is not None:
            self.deconv = upsampling_layer(upsampling_mode, scale_factor=2, dim_in=dim_in)
        else:
            self.deconv = nn.Identity()
        
        self.out_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
    
    def __call__(self, x0: mx.array, x1: Optional[mx.array] = None) -> mx.array:
        """Process and fuse input features."""
        x = x0
        
        if x1 is not None:
            res = self.resnet1(x1)
            x = x + res
        
        x = self.resnet2(x)
        x = self.deconv(x)
        x = self.out_conv(x)
        
        return x
    
    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool) -> ResidualBlock:
        """Create a residual block."""
        layers = [
            ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm(num_features))
        
        layers.extend([
            ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        ])
        if batch_norm:
            layers.append(nn.BatchNorm(num_features))
        
        residual = Sequential(*layers)
        return ResidualBlock(residual)
