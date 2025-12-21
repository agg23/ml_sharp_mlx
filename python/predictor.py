"""Main Sharp predictor for 3D Gaussian prediction.

Combines all modules into the complete RGBGaussianPredictor (SharpPredictor).
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple

try:
    from .monodepth import (
        MonodepthWithEncodingAdaptor,
        MonodepthDensePredictionTransformer,
        create_monodepth_dpt,
        create_monodepth_adaptor,
    )
    from .gaussian import (
        Gaussians3D,
        GaussianBaseValues,
        MultiLayerInitializer,
        GaussianDensePredictionTransformer,
        GaussianComposer,
        ImageFeatures,
    )
    from .decoder import MultiresConvDecoder
    from .blocks import Sequential
except ImportError:
    from monodepth import (
        MonodepthWithEncodingAdaptor,
        MonodepthDensePredictionTransformer,
        create_monodepth_dpt,
        create_monodepth_adaptor,
    )
    from gaussian import (
        Gaussians3D,
        GaussianBaseValues,
        MultiLayerInitializer,
        GaussianDensePredictionTransformer,
        GaussianComposer,
        ImageFeatures,
    )
    from decoder import MultiresConvDecoder
    from blocks import Sequential


class DirectPredictionHead(nn.Module):
    """Prediction head that converts features to delta values.
    
    Returns combined delta tensor with shape [B, H, W, 14, num_layers] in NHWC format.
    """
    
    def __init__(self, feature_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        # PyTorch structure:
        # - geometry: 3 * num_layers (means delta)
        # - texture: (14 - 3) * num_layers = 11 * num_layers (scales, quaternions, colors, opacity)
        geometry_out = 3 * num_layers
        texture_out = (14 - 3) * num_layers
        
        self.geometry_prediction_head = nn.Conv2d(feature_dim, geometry_out, kernel_size=1)
        self.texture_prediction_head = nn.Conv2d(feature_dim, texture_out, kernel_size=1)
    
    def __call__(self, features: ImageFeatures) -> mx.array:
        """Convert image features to delta predictions.
        
        Returns:
            Combined delta tensor of shape [B, H, W, 14, num_layers]
        """
        # texture: [B, H, W, 11 * num_layers]
        texture = self.texture_prediction_head(features.texture_features)
        # geometry: [B, H, W, 3 * num_layers]
        geometry = self.geometry_prediction_head(features.geometry_features)
        
        B, H, W, _ = texture.shape
        
        # Unflatten to [B, H, W, C, num_layers]
        # geometry: [B, H, W, 3, num_layers]
        geometry = geometry.reshape(B, H, W, 3, self.num_layers)
        # texture: [B, H, W, 11, num_layers]
        texture = texture.reshape(B, H, W, 11, self.num_layers)
        
        # Concatenate along channel dim: [B, H, W, 14, num_layers]
        delta = mx.concatenate([geometry, texture], axis=3)
        
        return delta


class SharpPredictor(nn.Module):
    """Sharp predictor for 3D Gaussian Splatting from a single image.
    
    Given a single photograph, predicts the parameters of a 3D Gaussian
    representation of the depicted scene.
    """
    
    def __init__(
        self,
        init_model: MultiLayerInitializer,
        monodepth_model: MonodepthWithEncodingAdaptor,
        feature_model: GaussianDensePredictionTransformer,
        prediction_head: DirectPredictionHead,
        gaussian_composer: GaussianComposer,
    ):
        super().__init__()
        self.init_model = init_model
        self.monodepth_model = monodepth_model
        self.feature_model = feature_model
        self.prediction_head = prediction_head
        self.gaussian_composer = gaussian_composer
    
    def __call__(
        self,
        image: mx.array,
        disparity_factor: mx.array,
    ) -> Gaussians3D:
        """Predict 3D Gaussians from an image.
        
        Args:
            image: Input image (B, H, W, 3) in NHWC format, values in [0, 1]
            disparity_factor: Factor to convert depth to disparities (B,)
            
        Returns:
            Gaussians3D with predicted 3D Gaussian parameters
        """
        # Estimate depth
        monodepth_output = self.monodepth_model(image)
        monodepth_disparity = monodepth_output.disparity
        
        # Convert disparity to depth
        disparity_factor = disparity_factor[:, None, None, None]
        monodepth = disparity_factor / mx.clip(monodepth_disparity, 1e-4, 1e4)
        
        # Initialize base Gaussians
        init_output = self.init_model(image, monodepth)
        
        # Predict delta values
        image_features = self.feature_model(
            init_output.feature_input,
            encodings=monodepth_output.output_features,
        )
        delta_values = self.prediction_head(image_features)
        
        # Compose final Gaussians
        gaussians = self.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        
        return gaussians
    
    def internal_resolution(self) -> int:
        """Return the internal image size."""
        return self.monodepth_model.internal_resolution()


def create_predictor(
    num_layers: int = 2,
    init_stride: int = 2,
    dims_encoder: list = None,
    dims_decoder: list = None,
    feature_dim: int = 128,
    gaussian_stride: int = 2,
) -> SharpPredictor:
    """Create a Sharp predictor model.
    
    Args:
        num_layers: Number of Gaussian layers
        init_stride: Stride for initializer
        dims_encoder: Encoder dimensions
        dims_decoder: Decoder dimensions
        feature_dim: Feature dimension for Gaussian decoder
        gaussian_stride: Stride for Gaussian prediction
        
    Returns:
        SharpPredictor model
    """
    if dims_encoder is None:
        dims_encoder = [256, 256, 512, 1024, 1024]
    if dims_decoder is None:
        dims_decoder = [128, 128, 128, 128, 128]
    
    # Monodepth uses 256 for all decoder dims
    monodepth_dims_decoder = [256, 256, 256, 256, 256]
    
    # Create monodepth model with correct dims
    monodepth_predictor = create_monodepth_dpt(
        dims_encoder=dims_encoder,
        dims_decoder=monodepth_dims_decoder,  # Use 256 for monodepth
        num_disparity_channels=2,
        use_patch_overlap=True,
    )
    monodepth_model = create_monodepth_adaptor(
        monodepth_predictor,
        return_encoder_features=True,
        return_decoder_features=False,
        num_monodepth_layers=1,
        sorting_monodepth=False,
    )
    
    # Create initializer
    init_model = MultiLayerInitializer(
        num_layers=num_layers,
        stride=init_stride,
        base_depth=10.0,  # PyTorch uses 10.0
        scale_factor=1.0,
        disparity_factor=1.0,
        normalize_depth=True,
    )
    
    # Create Gaussian decoder
    # The feature_model.decoder uses encoder features only (5 levels), not encoder+decoder
    # Checkpoint shows: convs[0-4] with dims [256, 256, 512, 1024, 1024]
    gaussian_encoder_dims = dims_encoder  # Use encoder dims, not get_feature_dims()
    gaussian_decoder_dims = [128] * len(gaussian_encoder_dims)
    gaussian_decoder = MultiresConvDecoder(
        dims_encoder=gaussian_encoder_dims,
        dims_decoder=gaussian_decoder_dims,
        upsampling_mode="transposed_conv",
    )
    
    # dim_in = RGB (3) + disparity (2) = 5
    # dim_out = 32 (from checkpoint, feeds into prediction head)
    feature_model = GaussianDensePredictionTransformer(
        decoder=gaussian_decoder,
        dim_in=5,  # RGB + 2 disparity channels
        dim_out=32,  # Matches checkpoint texture/geometry_head output
        stride_out=gaussian_stride,
        norm_type="group_norm",
        norm_num_groups=8,
        use_depth_input=True,
    )
    
    # Create prediction head
    prediction_head = DirectPredictionHead(
        feature_dim=32,  # Matches feature_model dim_out
        num_layers=num_layers,
    )
    
    # Create Gaussian composer with per-component delta factors (matching PyTorch)
    gaussian_composer = GaussianComposer(
        delta_factor_xy=0.001,
        delta_factor_z=0.001,
        delta_factor_scale=1.0,
        delta_factor_quaternion=1.0,
        delta_factor_color=0.1,
        delta_factor_opacity=1.0,
        min_scale=0.0,
        max_scale=10.0,
        scale_factor=gaussian_stride // init_stride if gaussian_stride > init_stride else 1,
        base_scale_on_predicted_mean=True,
    )
    
    return SharpPredictor(
        init_model=init_model,
        monodepth_model=monodepth_model,
        feature_model=feature_model,
        prediction_head=prediction_head,
        gaussian_composer=gaussian_composer,
    )
