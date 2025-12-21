"""Vision Transformer implementation for Sharp MLX.

DINOv2-L/16 architecture with:
- 24 transformer blocks
- 1024 embedding dimension  
- 16 attention heads
- LayerScale with init 1e-5
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Dict, List


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding using Conv2d."""
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Conv2d with kernel_size=patch_size, stride=patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - NHWC format
        B, H, W, C = x.shape
        
        # Apply projection: (B, H, W, C) -> (B, H/P, W/P, D)
        x = self.proj(x)
        
        # Reshape to sequence: (B, H/P, W/P, D) -> (B, N, D) where N = (H/P) * (W/P)
        B, Hp, Wp, D = x.shape
        x = x.reshape(B, Hp * Wp, D)
        
        return x


class Attention(nn.Module):
    """Multi-head self attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        
        # QKV projection: (B, N, C) -> (B, N, 3*C)
        qkv = self.qkv(x)
        
        # Reshape to (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Transpose to (3, B, num_heads, N, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        
        # Split into q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        
        # Apply attention to values
        x = attn @ v
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    """Layer Scale from CaiT/DeiT-III."""
    
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_values
    
    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Block(nn.Module):
    """Transformer block with pre-norm and LayerScale."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)
        self.ls2 = LayerScale(dim, init_values=init_values)
    
    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for DINOv2.
    
    Supports extraction of intermediate features for multi-scale processing.
    """
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-5,
        intermediate_features_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.intermediate_features_ids = intermediate_features_ids or []
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = mx.zeros((1, 1, embed_dim))
        
        # Position embedding: class token + patches
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
            )
            for _ in range(depth)
        ]
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def reshape_feature(self, embeddings: mx.array) -> mx.array:
        """Discard class token and reshape 1D feature map to 2D grid (NHWC)."""
        B, seq_len, C = embeddings.shape
        
        # Remove class token (first token)
        embeddings = embeddings[:, 1:, :]
        
        # Calculate grid size
        grid_size = int(math.sqrt(seq_len - 1))
        
        # Reshape: (B, N, C) -> (B, H, W, C) - NHWC format
        embeddings = embeddings.reshape(B, grid_size, grid_size, C)
        
        return embeddings
    
    def __call__(self, x: mx.array) -> Tuple[mx.array, Dict[int, mx.array]]:
        """Forward pass with intermediate feature extraction.
        
        Args:
            x: Input image (B, H, W, C) in NHWC format
            
        Returns:
            Output features reshaped to 2D grid, and dict of intermediate features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Prepend class token
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls_tokens, x], axis=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks with intermediate feature extraction
        intermediate_features = {}
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.intermediate_features_ids:
                intermediate_features[idx] = x
        
        # Final norm
        x = self.norm(x)
        
        # Reshape to 2D grid (NHWC)
        x = self.reshape_feature(x)
        
        return x, intermediate_features
    
    def internal_resolution(self) -> int:
        """Return the internal image size."""
        return self.img_size


def create_dinov2_vit_large(
    img_size: int = 384,
    intermediate_features_ids: Optional[List[int]] = None,
) -> VisionTransformer:
    """Create DINOv2 ViT-Large/16 model."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1e-5,
        intermediate_features_ids=intermediate_features_ids,
    )
