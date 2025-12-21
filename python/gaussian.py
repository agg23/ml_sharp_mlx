"""Gaussian prediction modules for Sharp MLX.

Contains:
- GaussianBaseValues: Base values for Gaussian initialization
- MultiLayerInitializer: Creates base Gaussians from RGBD
- GaussianComposer: Combines base + delta values with activations
- GaussianDensePredictionTransformer: Predicts delta values
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import NamedTuple, Optional, List, Tuple

try:
    from .decoder import MultiresConvDecoder
    from .blocks import (
        FeatureFusionBlock2d,
        residual_block_2d,
        ConvTranspose2d,
        Sequential,
        ReLU,
        NormLayerName,
    )
except ImportError:
    from decoder import MultiresConvDecoder
    from blocks import (
        FeatureFusionBlock2d,
        residual_block_2d,
        ConvTranspose2d,
        Sequential,
        ReLU,
        NormLayerName,
    )


class Gaussians3D(NamedTuple):
    """3D Gaussian splat parameters."""
    
    means: mx.array  # (B, N, 3) or (B, H, W, 3)
    scales: mx.array  # (B, N, 3) or (B, H, W, 3)
    quaternions: mx.array  # (B, N, 4) or (B, H, W, 4)
    colors: mx.array  # (B, N, 3) or (B, H, W, 3)
    opacities: mx.array  # (B, N, 1) or (B, H, W, 1)


class GaussianBaseValues(NamedTuple):
    """Base values for Gaussian predictor.
    
    We predict x and y in normalized device coordinates (NDC) where (-1, -1) is
    the top left corner and (1, 1) the bottom right corner.
    """
    
    mean_x_ndc: mx.array
    mean_y_ndc: mx.array
    mean_inverse_z_ndc: mx.array
    scales: mx.array
    quaternions: mx.array
    colors: mx.array
    opacities: mx.array


class InitializerOutput(NamedTuple):
    """Output of initializer."""
    
    gaussian_base_values: GaussianBaseValues
    feature_input: mx.array
    global_scale: Optional[mx.array] = None


class ImageFeatures(NamedTuple):
    """Image features extracted from decoder."""
    
    texture_features: mx.array
    geometry_features: mx.array


class MultiLayerInitializer(nn.Module):
    """Initialize Gaussians with multilayer representation.
    
    Creates base values for 3D Gaussians from RGB image and depth map.
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        stride: int = 2,
        base_depth: float = 10.0,  # PyTorch uses 10.0
        scale_factor: float = 1.0,
        disparity_factor: float = 1.0,
        normalize_depth: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.normalize_depth = normalize_depth
    
    def __call__(self, image: mx.array, depth: mx.array) -> InitializerOutput:
        """Construct Gaussian base values.
        
        Args:
            image: RGB image (B, H, W, 3) in NHWC format
            depth: Depth map (B, H, W, 1 or 2)
            
        Returns:
            InitializerOutput with base Gaussian values
        """
        B, H, W, C = depth.shape
        base_height = H // self.stride
        base_width = W // self.stride
        
        global_scale = None
        if self.normalize_depth:
            depth, depth_factor = self._rescale_depth(depth)
            global_scale = 1.0 / depth_factor
        
        # Create disparity layers (matching PyTorch surface_min logic)
        # PyTorch uses first_layer_depth_option="surface_min" and rest_layer_depth_option="surface_min"
        # This means: layer 0 comes from depth[:, :, :, 0:1], layer 1 from depth[:, :, :, 1:] (or 0:1 if single channel)
        disparity = 1.0 / mx.maximum(depth, 1e-4)
        
        # Use max_pool on disparity (equivalent to surface_min on depth)
        disparity_pooled = self._max_pool2d(disparity, self.stride)
        
        # Build disparity_layers: [B, H_base, W_base, num_layers]
        if self.num_layers == 1:
            disparity_layers = disparity_pooled[:, :, :, 0:1]
        else:
            # First layer from channel 0, rest from channel 1 (or channel 0 if single-channel depth)
            first_layer = disparity_pooled[:, :, :, 0:1]  # [B, H, W, 1]
            if depth.shape[-1] > 1:
                # Use channel 1 for following layers
                following_layer = disparity_pooled[:, :, :, 1:2]  # [B, H, W, 1]
            else:
                # Single-channel depth: use same channel for all layers
                following_layer = first_layer
            # Concatenate to form [B, H, W, 2]
            disparity_layers = mx.concatenate([first_layer, following_layer], axis=-1)
        
        # Create base x, y coordinates
        base_x_ndc, base_y_ndc = self._create_base_xy(H, W, B)
        
        # Create base scales
        disparity_scale_factor = 2 * self.scale_factor * self.stride / float(W)
        base_scales = (1.0 / disparity_layers) * disparity_scale_factor
        base_scales = mx.broadcast_to(
            base_scales[:, :, :, :, None],
            (B, base_height, base_width, self.num_layers, 3)
        )
        
        # Base quaternions (identity rotation)
        base_quaternions = mx.array([1.0, 0.0, 0.0, 0.0])
        base_quaternions = mx.broadcast_to(
            base_quaternions[None, None, None, None, :],
            (B, base_height, base_width, self.num_layers, 4)
        )
        
        # Base opacities
        base_opacities = mx.array([min(1.0 / self.num_layers, 0.5)])
        base_opacities = mx.broadcast_to(
            base_opacities[None, None, None, None, :],
            (B, base_height, base_width, self.num_layers, 1)
        )
        
        # Base colors from pooled image
        image_pooled = self._avg_pool2d(image, self.stride)
        base_colors = mx.broadcast_to(
            image_pooled[:, :, :, None, :],
            (B, base_height, base_width, self.num_layers, 3)
        )
        
        # Prepare feature input
        normalized_disparity = self.disparity_factor / mx.maximum(depth, 1e-4)
        features_in = mx.concatenate([image, normalized_disparity], axis=-1)
        features_in = 2.0 * features_in - 1.0
        
        base_gaussian_values = GaussianBaseValues(
            mean_x_ndc=base_x_ndc,
            mean_y_ndc=base_y_ndc,
            mean_inverse_z_ndc=disparity_layers,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )
        
        return InitializerOutput(
            gaussian_base_values=base_gaussian_values,
            feature_input=features_in,
            global_scale=global_scale,
        )
    
    def _create_base_xy(self, H: int, W: int, B: int) -> Tuple[mx.array, mx.array]:
        """Create base x, y coordinates in NDC."""
        base_H = H // self.stride
        base_W = W // self.stride
        
        xx = mx.arange(0.5 * self.stride, W, self.stride)
        yy = mx.arange(0.5 * self.stride, H, self.stride)
        xx = 2 * xx / W - 1.0
        yy = 2 * yy / H - 1.0
        
        # Create meshgrid
        xx_grid = mx.broadcast_to(xx[None, None, :], (B, base_H, base_W))
        yy_grid = mx.broadcast_to(yy[None, :, None], (B, base_H, base_W))
        
        # Expand for num_layers: [B, H, W] -> [B, H, W, num_layers]
        base_x_ndc = mx.broadcast_to(
            xx_grid[:, :, :, None],
            (B, base_H, base_W, self.num_layers)
        )
        base_y_ndc = mx.broadcast_to(
            yy_grid[:, :, :, None],
            (B, base_H, base_W, self.num_layers)
        )
        
        return base_x_ndc, base_y_ndc
    
    def _rescale_depth(
        self, depth: mx.array, depth_min: float = 1.0, depth_max: float = 100.0
    ) -> Tuple[mx.array, mx.array]:
        """Rescale depth to stable range."""
        current_depth_min = mx.min(depth.reshape(depth.shape[0], -1), axis=-1)
        depth_factor = depth_min / (current_depth_min + 1e-6)
        depth = mx.clip(depth * depth_factor[:, None, None, None], a_min=0.0, a_max=depth_max)
        return depth, depth_factor
    
    def _max_pool2d(self, x: mx.array, pool_size: int) -> mx.array:
        """Max pooling for NHWC format."""
        B, H, W, C = x.shape
        new_H = H // pool_size
        new_W = W // pool_size
        x = x.reshape(B, new_H, pool_size, new_W, pool_size, C)
        return mx.max(x, axis=(2, 4))
    
    def _avg_pool2d(self, x: mx.array, pool_size: int) -> mx.array:
        """Average pooling for NHWC format."""
        B, H, W, C = x.shape
        new_H = H // pool_size
        new_W = W // pool_size
        x = x.reshape(B, new_H, pool_size, new_W, pool_size, C)
        return mx.mean(x, axis=(2, 4))


class SkipConvBackbone(nn.Module):
    """A simple conv layer wrapper for feature extraction."""
    
    def __init__(self, dim_in: int, dim_out: int, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        # For strided downsampling, kernel_size = stride with no padding
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=stride, stride=stride, padding=0)
    
    def __call__(self, x: mx.array) -> ImageFeatures:
        output = self.conv(x)
        return ImageFeatures(texture_features=output, geometry_features=output)


class GaussianDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for Gaussian parameters.
    
    Reuses monodepth features to predict delta values for Gaussians.
    """
    
    def __init__(
        self,
        decoder: MultiresConvDecoder,
        dim_in: int,
        dim_out: int,
        stride_out: int = 2,
        norm_type: NormLayerName = "group_norm",
        norm_num_groups: int = 8,
        use_depth_input: bool = True,
    ):
        super().__init__()
        
        self.decoder = decoder
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride_out = stride_out
        self.use_depth_input = use_depth_input
        
        # Image encoder to lift to decoder dimension
        actual_dim_in = dim_in if use_depth_input else dim_in - 1
        kernel_size = 3 if stride_out != 1 else 1
        self.image_encoder = SkipConvBackbone(
            actual_dim_in, decoder.dim_out, kernel_size=kernel_size, stride=stride_out
        )
        
        self.fusion = FeatureFusionBlock2d(decoder.dim_out)
        
        if stride_out == 1:
            self.upsample = ConvTranspose2d(
                decoder.dim_out, decoder.dim_out, kernel_size=2, stride=2, padding=0, bias=False
            )
        else:
            self.upsample = nn.Identity()
        
        # Prediction heads
        self.texture_head = self._create_head(decoder.dim_out, dim_out, norm_type, norm_num_groups)
        self.geometry_head = self._create_head(decoder.dim_out, dim_out, norm_type, norm_num_groups)
    
    def _create_head(
        self, dim_decoder: int, dim_out: int, norm_type: NormLayerName, norm_num_groups: int
    ) -> nn.Module:
        """Create prediction head."""
        return Sequential(
            residual_block_2d(dim_decoder, dim_decoder, dim_decoder // 2, norm_type, norm_num_groups),
            residual_block_2d(dim_decoder, dim_decoder, dim_decoder // 2, norm_type, norm_num_groups),
            ReLU(),
            nn.Conv2d(dim_decoder, dim_out, kernel_size=1, stride=1),
            ReLU(),
        )
    
    def __call__(self, input_features: mx.array, encodings: List[mx.array]) -> ImageFeatures:
        """Predict delta values for Gaussians.
        
        Args:
            input_features: Feature input from initializer
            encodings: Multi-resolution features from monodepth (may include decoder features)
            
        Returns:
            texture_features and geometry_features
        """
        # Only use the first N encodings that match decoder input dims
        num_decoder_levels = len(self.decoder.convs)
        encoder_features = encodings[:num_decoder_levels]
        
        features = self.decoder(encoder_features)
        features = self.upsample(features)
        
        if self.use_depth_input:
            skip_features = self.image_encoder(input_features).texture_features
        else:
            skip_features = self.image_encoder(input_features[:, :, :, :3]).texture_features
        
        features = self.fusion(features, skip_features)
        
        texture_features = self.texture_head(features)
        geometry_features = self.geometry_head(features)
        
        return ImageFeatures(texture_features, geometry_features)


class GaussianComposer(nn.Module):
    """Converts base values and deltas into final Gaussians.
    
    Applies activations and combines base + delta values.
    """
    
    def __init__(
        self,
        delta_factor_xy: float = 0.001,
        delta_factor_z: float = 0.001,
        delta_factor_scale: float = 1.0,
        delta_factor_quaternion: float = 1.0,
        delta_factor_color: float = 0.1,
        delta_factor_opacity: float = 1.0,
        min_scale: float = 0.0,  # PyTorch uses 0.0
        max_scale: float = 10.0,
        scale_factor: int = 1,
        base_scale_on_predicted_mean: bool = True,
    ):
        super().__init__()
        # Per-component delta factors (matching PyTorch's DeltaFactor)
        self.delta_factor_xy = delta_factor_xy
        self.delta_factor_z = delta_factor_z
        self.delta_factor_scale = delta_factor_scale
        self.delta_factor_quaternion = delta_factor_quaternion
        self.delta_factor_color = delta_factor_color
        self.delta_factor_opacity = delta_factor_opacity
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_factor = scale_factor
        self.base_scale_on_predicted_mean = base_scale_on_predicted_mean
    
    def __call__(
        self,
        delta: mx.array,
        base_values: GaussianBaseValues,
        global_scale: Optional[mx.array] = None,
        flatten_output: bool = True,
    ) -> Gaussians3D:
        """Combine base and delta values.
        
        Args:
            delta: Predicted delta values [B, H, W, 14, num_layers]
                   Channels: [mean_x, mean_y, mean_z, scale_x, scale_y, scale_z,
                              quat_w, quat_x, quat_y, quat_z, color_r, color_g, color_b, opacity]
            base_values: Base Gaussian values
            global_scale: Global scale factor to unscale depth
            flatten_output: Whether to flatten spatial + layer dimensions
            
        Returns:
            3D Gaussians
        """
        # Upsample delta if needed (nearest neighbor)
        if self.scale_factor > 1:
            # delta: [B, H, W, 14, num_layers]
            delta = mx.repeat(mx.repeat(delta, self.scale_factor, axis=1), self.scale_factor, axis=2)
        
        # Split delta: [B, H, W, C, num_layers] -> transpose to [B, H, W, num_layers, C]
        delta = mx.transpose(delta, (0, 1, 2, 4, 3))  # -> [B, H, W, num_layers, 14]
        
        # Extract components
        # mean_delta: [B, H, W, num_layers, 3]
        mean_delta = delta[:, :, :, :, :3]
        # scale_delta: [B, H, W, num_layers, 3]
        scale_delta = delta[:, :, :, :, 3:6]
        # quat_delta: [B, H, W, num_layers, 4]
        quat_delta = delta[:, :, :, :, 6:10]
        # color_delta: [B, H, W, num_layers, 3]
        color_delta = delta[:, :, :, :, 10:13]
        # opacity_delta: [B, H, W, num_layers, 1]
        opacity_delta = delta[:, :, :, :, 13:14]
        
        # Mean activation (softplus-based as in PyTorch)
        # base_values have shape [B, H, W, num_layers]
        # mean_delta[:, :, :, :, 0] -> [B, H, W, num_layers]
        mean_x = base_values.mean_x_ndc + mean_delta[:, :, :, :, 0] * self.delta_factor_xy
        mean_y = base_values.mean_y_ndc + mean_delta[:, :, :, :, 1] * self.delta_factor_xy
        
        # Inverse depth activation with softplus
        inverse_z_base = base_values.mean_inverse_z_ndc  # [B, H, W, num_layers]
        inverse_z_delta = mean_delta[:, :, :, :, 2] * self.delta_factor_z  # [B, H, W, num_layers]
        
        # Apply softplus-based activation: softplus(inverse_softplus(base) + delta)
        # Approximate: inverse_z = softplus(log(exp(base) - 1) + delta)
        # Simplified: just compute the activation directly
        eps = 1e-4
        inverse_z_activated = mx.log(1.0 + mx.exp(
            mx.log(mx.maximum(mx.exp(inverse_z_base) - 1.0, eps)) + inverse_z_delta
        ))
        mean_z = 1.0 / (inverse_z_activated + 1e-3)  # [B, H, W, num_layers]
        
        # Construct mean vectors: [B, H, W, num_layers, 3]
        # Account for NDC to metric: multiply x,y by z
        # mean_x, mean_y, mean_z all have shape [B, H, W, num_layers]
        means = mx.stack([
            mean_z * mean_x,
            mean_z * mean_y,
            mean_z,
        ], axis=-1)
        
        # Scale activation
        if self.base_scale_on_predicted_mean:
            # Adjust base scale for z offset: base_scale * base_inv_z * new_z
            base_scales = base_values.scales * base_values.mean_inverse_z_ndc[:, :, :, :, None] * mean_z[:, :, :, :, None]
        else:
            base_scales = base_values.scales
        
        # Apply sigmoid-based scale activation
        constant_a = (self.max_scale - self.min_scale) / (1 - self.min_scale) / (self.max_scale - 1)
        # constant_b = inverse_sigmoid((1 - min_scale) / (max_scale - min_scale))
        ratio = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
        constant_b = mx.log(ratio / (1.0 - ratio + 1e-8))
        scale_factor_mult = (self.max_scale - self.min_scale) * mx.sigmoid(
            constant_a * scale_delta * self.delta_factor_scale + constant_b
        ) + self.min_scale
        scales = base_scales * scale_factor_mult
        
        # Quaternion: base + delta (normalization happens in rendering)
        quats = base_values.quaternions + quat_delta * self.delta_factor_quaternion
        
        # Color activation (sigmoid-based)
        # Clamp base to valid range for inverse sigmoid
        base_colors = mx.clip(base_values.colors, 0.01, 0.99)
        # inverse_sigmoid(x) = log(x / (1 - x))
        inv_sigmoid_base = mx.log(base_colors / (1.0 - base_colors + 1e-8))
        colors = mx.sigmoid(inv_sigmoid_base + color_delta * self.delta_factor_color)
        
        # Convert sRGB to linearRGB (matching PyTorch color_space='linearRGB')
        # Formula from Apple Metal Shading Language spec, Section 7.7.7:
        # if sRGB <= 0.04045: linear = sRGB / 12.92
        # else: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
        threshold = 0.04045
        colors_linear = mx.where(
            colors <= threshold,
            colors / 12.92,
            ((colors + 0.055) / 1.055) ** 2.4
        )
        
        # Opacity activation (sigmoid-based)
        base_opacities = mx.clip(base_values.opacities, 0.01, 0.99)
        inv_sigmoid_base_op = mx.log(base_opacities / (1.0 - base_opacities + 1e-8))
        opacities = mx.sigmoid(inv_sigmoid_base_op + opacity_delta * self.delta_factor_opacity)
        
        # Apply global scaling
        if global_scale is not None:
            means = means * global_scale[:, None, None, None, None]
            scales = scales * global_scale[:, None, None, None, None]
        
        if flatten_output:
            B = means.shape[0]
            # Flatten [B, H, W, num_layers, C] -> [B, N, C]
            means = means.reshape(B, -1, 3)
            scales = scales.reshape(B, -1, 3)
            quats = quats.reshape(B, -1, 4)
            colors_linear = colors_linear.reshape(B, -1, 3)
            opacities = opacities.reshape(B, -1, 1)
        
        return Gaussians3D(
            means=means,
            scales=scales,
            quaternions=quats,
            colors=colors_linear,
            opacities=opacities,
        )
