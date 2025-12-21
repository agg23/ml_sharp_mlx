#!/usr/bin/env python3
"""Convert Sharp PyTorch weights to MLX safetensors format.

Dependencies:
- torch (required to load .pt checkpoint)
- numpy, safetensors

Usage:
    python -m sharp_mlx.convert -i sharp.pt -o sharp.safetensors
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for weight conversion. Install with: pip install torch")

from safetensors.numpy import save_file


def convert_conv2d_weight(weight: np.ndarray) -> np.ndarray:
    """Convert Conv2d weight from OIHW to OHWI format."""
    return weight.transpose(0, 2, 3, 1)


def convert_conv_transpose2d_weight(weight: np.ndarray) -> np.ndarray:
    """Convert ConvTranspose2d weight from IOHW to OHWI format."""
    return weight.transpose(1, 2, 3, 0)


def is_conv2d_weight(key: str, shape: tuple) -> bool:
    """Check if weight is a Conv2d weight."""
    if len(shape) != 4:
        return False
    if 'deconv' in key:
        return False
    if 'upsample' in key and 'weight' in key and shape[2] == shape[3] == 2:
        return False
    return True


def is_conv_transpose2d_weight(key: str, shape: tuple) -> bool:
    """Check if weight is a ConvTranspose2d weight."""
    if len(shape) != 4:
        return False
    if 'deconv' in key:
        return True
    if 'upsample' in key and shape[2] == shape[3] == 2:
        return True
    if 'head.1.weight' in key and shape[2] == shape[3] == 2:
        return True
    return False


def convert_state_dict(state_dict: Dict[str, "torch.Tensor"], verbose: bool = False) -> Dict[str, np.ndarray]:
    """Convert PyTorch state dict to MLX-compatible numpy arrays."""
    mlx_weights = {}
    
    for key, value in state_dict.items():
        np_value = value.cpu().numpy()
        original_shape = np_value.shape
        
        if is_conv_transpose2d_weight(key, np_value.shape):
            np_value = convert_conv_transpose2d_weight(np_value)
            if verbose:
                print(f"  ConvTranspose2d: {key}: {original_shape} -> {np_value.shape}")
        elif is_conv2d_weight(key, np_value.shape):
            np_value = convert_conv2d_weight(np_value)
            if verbose:
                print(f"  Conv2d: {key}: {original_shape} -> {np_value.shape}")
        elif len(np_value.shape) == 4 and np_value.shape[2] <= 16 and np_value.shape[3] <= 16:
            np_value = convert_conv2d_weight(np_value)
            if verbose:
                print(f"  Conv2d (auto): {key}: {original_shape} -> {np_value.shape}")
        
        mlx_weights[key] = np_value
    
    return mlx_weights


def load_pytorch_checkpoint(path: Path) -> Dict[str, "torch.Tensor"]:
    """Load PyTorch checkpoint and extract state dict."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'model' in checkpoint:
            return checkpoint['model']
        return checkpoint
    return checkpoint


def convert(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    """Convert PyTorch checkpoint to MLX safetensors.
    
    Args:
        input_path: Path to PyTorch .pt checkpoint
        output_path: Path to output .safetensors file
        verbose: Print conversion details
    """
    if verbose:
        print(f"Loading: {input_path}")
    state_dict = load_pytorch_checkpoint(input_path)
    
    if verbose:
        print(f"Converting {len(state_dict)} tensors...")
    mlx_weights = convert_state_dict(state_dict, verbose=verbose)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(mlx_weights, str(output_path))
    
    if verbose:
        size_gb = output_path.stat().st_size / 1e9
        print(f"Saved: {output_path} ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Sharp PyTorch weights to MLX")
    parser.add_argument("-i", "--input", type=Path, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("-o", "--output", type=Path, default=Path("sharp.safetensors"), help="Output path")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1
    
    convert(args.input, args.output, verbose=not args.quiet)
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
