"""Load Sharp weights from safetensors into MLX model.

This module handles mapping between PyTorch key names and MLX model structure.
No dependencies, pure MLX and safetensors.
"""

import re
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open


def load_safetensors(path: str | Path) -> Dict[str, mx.array]:
    """Load safetensors file into MLX arrays."""
    weights = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            weights[key] = mx.array(f.get_tensor(key))
    return weights


def map_mlx_to_pt_key(mlx_key: str) -> str:
    """Map MLX key to PyTorch key format.
    
    Handles differences:
    - MLX Sequential uses .layers.N, PyTorch uses just .N
    - MLX PyTorchGroupNorm uses .norm.weight/.norm.bias
    """
    pt_key = mlx_key
    pt_key = re.sub(r'\.residual\.layers\.(\d+)', r'.residual.\1', pt_key)
    pt_key = re.sub(r'(upsample\w*)\.layers\.(\d+)', r'\1.\2', pt_key)
    pt_key = re.sub(r'\.texture_head\.layers\.(\d+)', r'.texture_head.\1', pt_key)
    pt_key = re.sub(r'\.geometry_head\.layers\.(\d+)', r'.geometry_head.\1', pt_key)
    pt_key = re.sub(r'\.norm\.(weight|bias)$', r'.\1', pt_key)
    return pt_key


def load_weights(
    model,
    weights_path: str | Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load weights from safetensors into MLX model.
    
    Args:
        model: MLX model to load weights into
        weights_path: Path to safetensors file
        verbose: Print loading statistics
        
    Returns:
        Dictionary with loading statistics
    """
    weights = load_safetensors(weights_path)
    model_params = dict(tree_flatten(model.parameters()))
    
    loaded, missing, new_weights = [], [], {}
    unused = set(weights.keys())
    
    for mlx_key, mlx_param in model_params.items():
        pt_key = mlx_key if mlx_key in weights else map_mlx_to_pt_key(mlx_key)
        
        if pt_key in weights:
            pt_weight = weights[pt_key]
            unused.discard(pt_key)
            
            if pt_weight.shape == mlx_param.shape:
                new_weights[mlx_key] = pt_weight
                loaded.append(mlx_key)
            else:
                missing.append(mlx_key)
        else:
            missing.append(mlx_key)
    
    if new_weights:
        model.update(tree_unflatten(list(new_weights.items())))
    
    if verbose:
        print(f"Loaded {len(loaded)}/{len(model_params)} parameters")
    
    return {
        "loaded": len(loaded),
        "missing": len(missing),
        "unused": len(unused),
    }
