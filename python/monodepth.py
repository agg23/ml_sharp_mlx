"""Monodepth Dense Prediction Transformer for Sharp MLX.

Implements depth estimation using:
- SlidingPyramidNetwork encoder
- MultiresConvDecoder decoder
- Disparity prediction head
"""

import mlx.core as mx
import mlx.nn as nn
from typing import NamedTuple, List, Optional, Tuple

try:
    from .spn_encoder import SlidingPyramidNetwork, create_spn_encoder
    from .decoder import MultiresConvDecoder
    from .blocks import Sequential, ReLU, ConvTranspose2d
except ImportError:
    from spn_encoder import SlidingPyramidNetwork, create_spn_encoder
    from decoder import MultiresConvDecoder
    from blocks import Sequential, ReLU, ConvTranspose2d


class AffineRangeNormalizer(nn.Module):
    """Normalize input from one range to another."""
    
    def __init__(
        self,
        input_range: Tuple[float, float] = (0, 1),
        output_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        self.input_min, self.input_max = input_range
        self.output_min, self.output_max = output_range
        
        # Calculate scale and shift
        input_span = self.input_max - self.input_min
        output_span = self.output_max - self.output_min
        self.scale = output_span / input_span
        self.shift = self.output_min - self.input_min * self.scale
    
    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale + self.shift


class MonodepthOutput(NamedTuple):
    """Output of the monodepth model."""
    
    disparity: mx.array
    encoder_features: List[mx.array]
    decoder_features: mx.array
    output_features: List[mx.array]
    intermediate_features: List[mx.array] = []


class MonodepthDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for monodepth.
    
    Combines SPN encoder + MultiresConvDecoder + disparity head.
    """
    
    def __init__(
        self,
        encoder: SlidingPyramidNetwork,
        decoder: MultiresConvDecoder,
        last_dims: Tuple[int, int] = (32, 1),
    ):
        """Initialize MonodepthDPT.
        
        Args:
            encoder: The SlidingPyramidNetwork backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: Dimensions for the last conv layers.
        """
        super().__init__()
        
        self.normalizer = AffineRangeNormalizer(
            input_range=(0, 1), output_range=(-1, 1)
        )
        self.encoder = encoder
        self.decoder = decoder
        
        dim_decoder = decoder.dim_out
        
        # Disparity prediction head
        self.head = [
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            ReLU(),
        ]
    
    def __call__(self, image: mx.array) -> mx.array:
        """Process image and return disparity.
        
        Args:
            image: Input image (B, H, W, C) in NHWC format, values in [0, 1]
            
        Returns:
            Disparity map (B, H_out, W_out, 1)
        """
        encodings = self.encoder(self.normalizer(image))
        num_encoder_features = len(self.encoder.dims_encoder)
        features = self.decoder(encodings[:num_encoder_features])
        
        # Apply head
        disparity = features
        for layer in self.head:
            disparity = layer(disparity)
        
        return disparity
    
    def internal_resolution(self) -> int:
        """Return the internal image size."""
        return self.encoder.internal_resolution()


class MonodepthWithEncodingAdaptor(nn.Module):
    """Monodepth model wrapper that returns features along with disparity."""
    
    def __init__(
        self,
        monodepth_predictor: MonodepthDensePredictionTransformer,
        return_encoder_features: bool = True,
        return_decoder_features: bool = True,
        num_monodepth_layers: int = 1,
        sorting_monodepth: bool = False,
    ):
        super().__init__()
        self.monodepth_predictor = monodepth_predictor
        self.return_encoder_features = return_encoder_features
        self.return_decoder_features = return_decoder_features
        self.num_monodepth_layers = num_monodepth_layers
        self.sorting_monodepth = sorting_monodepth
    
    def __call__(self, image: mx.array) -> MonodepthOutput:
        """Process image and return disparity + features."""
        inputs = self.monodepth_predictor.normalizer(image)
        encoder_output = self.monodepth_predictor.encoder(inputs)
        
        num_encoder_features = len(self.monodepth_predictor.encoder.dims_encoder)
        encoder_features = encoder_output[:num_encoder_features]
        intermediate_features = encoder_output[num_encoder_features:]
        
        decoder_features = self.monodepth_predictor.decoder(encoder_features)
        
        # Apply head
        disparity = decoder_features
        for layer in self.monodepth_predictor.head:
            disparity = layer(disparity)
        
        # Sort disparity layers if needed
        if self.num_monodepth_layers == 2 and self.sorting_monodepth:
            first_layer = mx.max(disparity, axis=-1, keepdims=True)
            second_layer = mx.min(disparity, axis=-1, keepdims=True)
            disparity = mx.concatenate([first_layer, second_layer], axis=-1)
        
        output_features = []
        if self.return_encoder_features:
            output_features.extend(encoder_features)
        if self.return_decoder_features:
            output_features.append(decoder_features)
        
        return MonodepthOutput(
            disparity=disparity,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            output_features=output_features,
            intermediate_features=intermediate_features,
        )
    
    def get_feature_dims(self) -> List[int]:
        """Return dimensions of output feature maps."""
        dims = []
        if self.return_encoder_features:
            dims.extend(self.monodepth_predictor.encoder.dims_encoder)
        if self.return_decoder_features:
            dims.append(self.monodepth_predictor.decoder.dim_out)
        return dims
    
    def internal_resolution(self) -> int:
        """Return the internal image size."""
        return self.monodepth_predictor.internal_resolution()


def create_monodepth_dpt(
    dims_encoder: List[int] = [256, 256, 512, 1024, 1024],
    dims_decoder: List[int] = [256, 256, 256, 256, 256],
    num_disparity_channels: int = 2,
    use_patch_overlap: bool = True,
) -> MonodepthDensePredictionTransformer:
    """Create MonodepthDensePredictionTransformer.
    
    Args:
        dims_encoder: Encoder output dimensions
        dims_decoder: Decoder output dimensions
        use_patch_overlap: Whether to use overlapping patches in SPN
        
    Returns:
        MonodepthDensePredictionTransformer model
    """
    encoder = create_spn_encoder(
        dims_encoder=dims_encoder,
        use_patch_overlap=use_patch_overlap,
    )
    
    decoder = MultiresConvDecoder(
        dims_encoder=dims_encoder,
        dims_decoder=dims_decoder,
        upsampling_mode="transposed_conv",
    )
    
    return MonodepthDensePredictionTransformer(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, num_disparity_channels),
    )


def create_monodepth_adaptor(
    monodepth_predictor: MonodepthDensePredictionTransformer,
    return_encoder_features: bool = True,
    return_decoder_features: bool = True,
    num_monodepth_layers: int = 1,
    sorting_monodepth: bool = False,
) -> MonodepthWithEncodingAdaptor:
    """Create MonodepthWithEncodingAdaptor."""
    return MonodepthWithEncodingAdaptor(
        monodepth_predictor=monodepth_predictor,
        return_encoder_features=return_encoder_features,
        return_decoder_features=return_decoder_features,
        num_monodepth_layers=num_monodepth_layers,
        sorting_monodepth=sorting_monodepth,
    )
