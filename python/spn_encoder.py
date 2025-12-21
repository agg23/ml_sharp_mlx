"""Sliding Pyramid Network encoder for Sharp MLX.

Creates multi-resolution encodings from Vision Transformers by:
1. Creating an image pyramid
2. Generating overlapping patches with a sliding window at each pyramid level
3. Creating batched encodings via ViT backbones
4. Producing multi-resolution encodings
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Dict, Optional

try:
    from .vit import VisionTransformer, create_dinov2_vit_large
    from .blocks import ConvTranspose2d, Sequential
except ImportError:
    from vit import VisionTransformer, create_dinov2_vit_large
    from blocks import ConvTranspose2d, Sequential


def split(image: mx.array, overlap_ratio: float = 0.25, patch_size: int = 384) -> mx.array:
    """Split the input into small patches with sliding window.
    
    Args:
        image: Input image (B, H, W, C) in NHWC format
        overlap_ratio: Overlap ratio between adjacent patches
        patch_size: Size of each patch
        
    Returns:
        Patches stacked along batch dimension (N*B, patch_size, patch_size, C)
    """
    B, H, W, C = image.shape
    patch_stride = int(patch_size * (1 - overlap_ratio))
    
    image_size = H  # Assume square images
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size
        
        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            patches.append(image[:, j0:j1, i0:i1, :])
    
    # Stack all patches: (num_patches * B, patch_size, patch_size, C)
    return mx.concatenate(patches, axis=0)


def merge(image_patches: mx.array, batch_size: int, padding: int = 3) -> mx.array:
    """Merge the patched input into an image with sliding window.
    
    Args:
        image_patches: Patches (num_patches * B, H, W, C)
        batch_size: Original batch size
        padding: Overlap padding to crop from edges
        
    Returns:
        Merged image (B, H_out, W_out, C)
    """
    steps = int(math.sqrt(image_patches.shape[0] // batch_size))
    
    idx = 0
    output_rows = []
    
    for j in range(steps):
        output_row_patches = []
        for i in range(steps):
            output = image_patches[batch_size * idx : batch_size * (idx + 1)]
            
            if padding != 0:
                if j != 0:
                    output = output[:, padding:, :, :]
                if i != 0:
                    output = output[:, :, padding:, :]
                if j != steps - 1:
                    output = output[:, :-padding, :, :]
                if i != steps - 1:
                    output = output[:, :, :-padding, :]
            
            output_row_patches.append(output)
            idx += 1
        
        output_row = mx.concatenate(output_row_patches, axis=2)  # Concat along W
        output_rows.append(output_row)
    
    output = mx.concatenate(output_rows, axis=1)  # Concat along H
    return output


def interpolate(x: mx.array, scale_factor: float, mode: str = "bilinear") -> mx.array:
    """Interpolate image tensor to match PyTorch F.interpolate.
    
    Args:
        x: Input (B, H, W, C) in NHWC format
        scale_factor: Scale factor for resizing
        mode: Interpolation mode ('bilinear' or 'nearest')
        
    Returns:
        Resized tensor matching PyTorch F.interpolate(align_corners=False)
    """
    B, H, W, C = x.shape
    new_H = int(H * scale_factor)
    new_W = int(W * scale_factor)
    
    if mode == "nearest":
        if scale_factor < 1:
            pool_size = int(1 / scale_factor)
            x = x[:, ::pool_size, ::pool_size, :]
        else:
            factor = int(scale_factor)
            x = mx.repeat(x, factor, axis=1)
            x = mx.repeat(x, factor, axis=2)
    elif mode == "bilinear":
        # Proper bilinear interpolation matching PyTorch align_corners=False
        # Create sampling grid in the source coordinate space
        # For align_corners=False: src_coord = (dst_coord + 0.5) / scale - 0.5
        
        # Generate destination coordinates
        y_dst = mx.arange(new_H, dtype=mx.float32)
        x_dst = mx.arange(new_W, dtype=mx.float32)
        
        # Map to source coordinates (align_corners=False behavior)
        y_src = (y_dst + 0.5) / scale_factor - 0.5
        x_src = (x_dst + 0.5) / scale_factor - 0.5
        
        # Clamp to valid range
        y_src = mx.clip(y_src, 0, H - 1)
        x_src = mx.clip(x_src, 0, W - 1)
        
        # Get integer and fractional parts
        y0 = mx.floor(y_src).astype(mx.int32)
        x0 = mx.floor(x_src).astype(mx.int32)
        y1 = mx.minimum(y0 + 1, H - 1)
        x1 = mx.minimum(x0 + 1, W - 1)
        
        fy = y_src - y0.astype(mx.float32)
        fx = x_src - x0.astype(mx.float32)
        
        # Reshape for broadcasting: [new_H] and [new_W]
        fy = fy[:, None, None]  # [new_H, 1, 1]
        fx = fx[None, :, None]  # [1, new_W, 1]
        
        # Gather pixels from 4 corners for all batches
        # x shape: [B, H, W, C]
        # We need x[:, y0, x0, :], x[:, y0, x1, :], x[:, y1, x0, :], x[:, y1, x1, :]
        
        # Create mesh of indices
        y0_grid = mx.broadcast_to(y0[:, None], (new_H, new_W))
        y1_grid = mx.broadcast_to(y1[:, None], (new_H, new_W))
        x0_grid = mx.broadcast_to(x0[None, :], (new_H, new_W))
        x1_grid = mx.broadcast_to(x1[None, :], (new_H, new_W))
        
        # Gather using advanced indexing for all batches at once
        # Create batch indices
        batch_outputs = []
        
        for b in range(B):
            # Get corners for this batch
            p00 = x[b, y0_grid, x0_grid, :]  # [new_H, new_W, C]
            p01 = x[b, y0_grid, x1_grid, :]
            p10 = x[b, y1_grid, x0_grid, :]
            p11 = x[b, y1_grid, x1_grid, :]
            
            # Bilinear interpolation
            interp = (1 - fy) * (1 - fx) * p00 + \
                     (1 - fy) * fx * p01 + \
                     fy * (1 - fx) * p10 + \
                     fy * fx * p11
            
            batch_outputs.append(interp)
        
        x = mx.stack(batch_outputs, axis=0)
    
    return x


class SlidingPyramidNetwork(nn.Module):
    """Sliding Pyramid Network encoder.
    
    Creates multi-resolution encodings from Vision Transformers.
    """
    
    def __init__(
        self,
        dims_encoder: List[int],
        patch_encoder: VisionTransformer,
        image_encoder: VisionTransformer,
        use_patch_overlap: bool = True,
    ):
        """Initialize Sliding Pyramid Network.
        
        Args:
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: ViT backbone for high-res patches.
            image_encoder: ViT backbone for low-res image.
            use_patch_overlap: Whether to use overlap between patches.
        """
        super().__init__()
        
        self.dims_encoder = dims_encoder
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.use_patch_overlap = use_patch_overlap
        
        base_embed_dim = patch_encoder.embed_dim
        lowres_embed_dim = image_encoder.embed_dim
        self.patch_size = patch_encoder.internal_resolution()
        
        # Intermediate feature ids (for DINOv2-L: layers 5, 11, 17, 23)
        self.patch_intermediate_features_ids = patch_encoder.intermediate_features_ids
        self.image_intermediate_features_ids = image_encoder.intermediate_features_ids
        
        # Upsampling blocks for patch encoder features
        self.upsample_latent0 = self._create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=dims_encoder[0],
            upsample_layers=3,
            dim_intermediate=dims_encoder[1],
        )
        self.upsample_latent1 = self._create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=dims_encoder[1],
            upsample_layers=2,
        )
        self.upsample0 = self._create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=dims_encoder[2],
            upsample_layers=1,
        )
        self.upsample1 = self._create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=dims_encoder[3],
            upsample_layers=1,
        )
        self.upsample2 = self._create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=dims_encoder[4],
            upsample_layers=1,
        )
        
        # Upsampling for image encoder features
        self.upsample_lowres = ConvTranspose2d(
            in_channels=lowres_embed_dim,
            out_channels=dims_encoder[4],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        
        # Fusion for combining patch and image encoder features
        self.fuse_lowres = nn.Conv2d(
            dims_encoder[4] + dims_encoder[4],
            dims_encoder[4],
            kernel_size=1,
            stride=1,
            padding=0,
        )
    
    def _create_project_upsample_block(
        self,
        dim_in: int,
        dim_out: int,
        upsample_layers: int,
        dim_intermediate: Optional[int] = None,
    ) -> nn.Module:
        """Create projection + upsampling block."""
        if dim_intermediate is None:
            dim_intermediate = dim_out
        
        layers = []
        
        # 1x1 projection (bias=False to match PyTorch checkpoint)
        layers.append(nn.Conv2d(dim_in, dim_intermediate, kernel_size=1, stride=1, padding=0, bias=False))
        
        # Upsampling layers (ConvTranspose2d with kernel=2, stride=2)
        for i in range(upsample_layers):
            in_ch = dim_intermediate if i == 0 else dim_out
            layers.append(ConvTranspose2d(
                in_channels=in_ch,
                out_channels=dim_out,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ))
        
        return Sequential(*layers)
    
    def _create_pyramid(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Create a 3-level image pyramid.
        
        Args:
            x: Input image (B, H, W, C), typically 1536x1536
            
        Returns:
            x0: Original resolution (1536)
            x1: Half resolution (768)
            x2: Quarter resolution (384)
        """
        x0 = x
        x1 = interpolate(x, scale_factor=0.5, mode="bilinear")
        x2 = interpolate(x, scale_factor=0.25, mode="bilinear")
        return x0, x1, x2
    
    def internal_resolution(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_size * 4  # 384 * 4 = 1536
    
    def __call__(self, x: mx.array) -> List[mx.array]:
        """Encode input at multiple resolutions.
        
        Args:
            x: Input image (B, H, W, C), expected to be 1536x1536
            
        Returns:
            List of 5 multi-resolution feature maps
        """
        batch_size = x.shape[0]
        
        # Step 0: Create 3-level image pyramid
        x0, x1, x2 = self._create_pyramid(x)
        
        if self.use_patch_overlap:
            # 5x5 @ 384x384 at highest resolution
            x0_patches = split(x0, overlap_ratio=0.25, patch_size=self.patch_size)
            # 3x3 @ 384x384 at middle resolution
            x1_patches = split(x1, overlap_ratio=0.5, patch_size=self.patch_size)
            # 1x1 @ 384x384 at lowest resolution
            x2_patches = x2
            padding = 3
        else:
            x0_patches = split(x0, overlap_ratio=0.0, patch_size=self.patch_size)
            x1_patches = split(x1, overlap_ratio=0.0, patch_size=self.patch_size)
            x2_patches = x2
            padding = 0
        
        x0_tile_size = x0_patches.shape[0] // batch_size
        
        # Concatenate all patches
        x_pyramid_patches = mx.concatenate([x0_patches, x1_patches, x2_patches], axis=0)
        
        # Run patch encoder on all patches
        x_pyramid_encodings, patch_intermediate_features = self.patch_encoder(x_pyramid_patches)
        
        # Extract and reshape intermediate features for latent encodings
        if self.patch_intermediate_features_ids:
            x_latent0_encodings = self.patch_encoder.reshape_feature(
                patch_intermediate_features[self.patch_intermediate_features_ids[0]]
            )
            x_latent0_features = merge(
                x_latent0_encodings[: batch_size * x0_tile_size],
                batch_size=batch_size,
                padding=padding,
            )
            
            x_latent1_encodings = self.patch_encoder.reshape_feature(
                patch_intermediate_features[self.patch_intermediate_features_ids[1]]
            )
            x_latent1_features = merge(
                x_latent1_encodings[: batch_size * x0_tile_size],
                batch_size=batch_size,
                padding=padding,
            )
        else:
            x_latent0_features = None
            x_latent1_features = None
        
        # Split pyramid encodings back
        num_x0 = x0_patches.shape[0]
        num_x1 = x1_patches.shape[0]
        num_x2 = x2_patches.shape[0]
        
        x0_encodings = x_pyramid_encodings[:num_x0]
        x1_encodings = x_pyramid_encodings[num_x0:num_x0 + num_x1]
        x2_encodings = x_pyramid_encodings[num_x0 + num_x1:]
        
        # Merge patches back to feature maps
        x0_features = merge(x0_encodings, batch_size=batch_size, padding=padding)
        x1_features = merge(x1_encodings, batch_size=batch_size, padding=2 * padding)
        x2_features = x2_encodings
        
        # Run image encoder on low-res image
        x_lowres_features, _ = self.image_encoder(x2_patches)
        
        # Upsample all feature maps
        if x_latent0_features is not None:
            x_latent0_features = self.upsample_latent0(x_latent0_features)
        if x_latent1_features is not None:
            x_latent1_features = self.upsample_latent1(x_latent1_features)
        
        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)
        
        x_lowres_features = self.upsample_lowres(x_lowres_features)
        x_lowres_features = self.fuse_lowres(
            mx.concatenate([x2_features, x_lowres_features], axis=-1)
        )
        
        output = [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_lowres_features,
        ]
        
        return output


def create_spn_encoder(
    dims_encoder: List[int] = [256, 256, 512, 1024, 1024],
    use_patch_overlap: bool = True,
) -> SlidingPyramidNetwork:
    """Create Sliding Pyramid Network encoder with DINOv2-L backbones.
    
    Args:
        dims_encoder: Output dimensions at each resolution level
        use_patch_overlap: Whether to use overlapping patches
        
    Returns:
        SlidingPyramidNetwork encoder
    """
    # Create patch encoder with intermediate feature extraction
    # DINOv2-L has 24 blocks, extract features at layers 5, 11, 17, 23
    patch_encoder = create_dinov2_vit_large(
        img_size=384,
        intermediate_features_ids=[5, 11, 17, 23],
    )
    
    # Create image encoder (no intermediate features needed)
    image_encoder = create_dinov2_vit_large(
        img_size=384,
        intermediate_features_ids=[],
    )
    
    return SlidingPyramidNetwork(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        use_patch_overlap=use_patch_overlap,
    )
