"""Multi-resolution convolutional decoder for Sharp MLX.

Implements the DPT-style decoder that fuses multi-resolution encoder features.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional, Iterable

try:
    from .blocks import FeatureFusionBlock2d, UpsamplingMode, Sequential
except ImportError:
    from blocks import FeatureFusionBlock2d, UpsamplingMode, Sequential


class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings.
    
    Progressively fuses features from low resolution to high resolution.
    """
    
    def __init__(
        self,
        dims_encoder: Iterable[int],
        dims_decoder: Iterable[int] | int,
        upsampling_mode: UpsamplingMode = "transposed_conv",
    ):
        """Initialize multiresolution convolutional decoder.
        
        Args:
            dims_encoder: Expected dims at each level from the encoder.
            dims_decoder: Dim of decoder features.
            upsampling_mode: What method to use for upsampling.
        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        
        if isinstance(dims_decoder, int):
            self.dims_decoder = [dims_decoder] * len(self.dims_encoder)
        else:
            self.dims_decoder = list(dims_decoder)
        
        if len(self.dims_decoder) != len(self.dims_encoder):
            raise ValueError("Received dims_encoder and dims_decoder of different sizes.")
        
        self.dim_out = self.dims_decoder[0]
        num_encoders = len(self.dims_encoder)
        
        # Projection convolutions (bias=False to match PyTorch checkpoint)
        convs = []
        for i in range(num_encoders):
            if i == 0:
                # At highest resolution, use 1x1 conv if dimensions differ
                if self.dims_encoder[i] != self.dims_decoder[i]:
                    conv = nn.Conv2d(self.dims_encoder[i], self.dims_decoder[i], kernel_size=1, bias=False)
                else:
                    conv = nn.Identity()
            else:
                conv = nn.Conv2d(self.dims_encoder[i], self.dims_decoder[i], kernel_size=3, stride=1, padding=1, bias=False)
            convs.append(conv)
        self.convs = convs
        
        # Fusion blocks
        fusions = []
        for i in range(num_encoders):
            dim_out = self.dims_decoder[i - 1] if i != 0 else self.dim_out
            fusions.append(
                FeatureFusionBlock2d(
                    dim_in=self.dims_decoder[i],
                    dim_out=dim_out,
                    upsampling_mode=upsampling_mode if i != 0 else None,
                    batch_norm=False,
                )
            )
        self.fusions = fusions
    
    def __call__(self, encodings: List[mx.array]) -> mx.array:
        """Decode the multi-resolution encodings.
        
        Args:
            encodings: List of features at different resolutions, ordered from
                highest resolution to lowest resolution.
                
        Returns:
            Fused features at the highest resolution.
        """
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)
        
        if num_levels != num_encoders:
            raise ValueError(
                f"Encoder output levels={num_levels} mismatch with expected levels={num_encoders}."
            )
        
        # Project features and fuse from lowest resolution to highest
        features = self.convs[-1](encodings[-1])
        features = self.fusions[-1](features)
        
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        
        return features


class BaseDecoder(nn.Module):
    """Base class for decoders."""
    
    def __init__(self):
        super().__init__()
        self.dim_out = 0
