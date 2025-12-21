"""Generate 3D Gaussians from an image using Sharp MLX.

Dependencies:
- mlx, numpy, PIL (required)
- torch (optional, for exact resize matching Apple's preprocessing)

Usage:
    python -m sharp_mlx.generate -i image.jpg -o output/
"""

import time
import argparse
from pathlib import Path
import numpy as np

import mlx.core as mx
from PIL import Image

try:
    from .predictor import create_predictor
    from .gaussian_utils import Gaussians3D, unproject_gaussians, save_ply
    from .weights import load_weights
except ImportError:
    from predictor import create_predictor
    from gaussian_utils import Gaussians3D, unproject_gaussians, save_ply
    from weights import load_weights


def load_image(path: Path) -> tuple[np.ndarray, float, int, int]:
    """Load image and compute focal length."""
    img = Image.open(path).convert("RGB")
    image = np.array(img).astype(np.float32) / 255.0
    
    height, width = image.shape[:2]
    f_px = max(width, height) * 1.2  # Default focal length estimation
    
    return image, f_px, height, width


def resize_image(image: np.ndarray, target_size: int = 1536, use_torch: bool = True) -> np.ndarray:
    """Resize image to target size.
    
    Args:
        image: Input image (H, W, 3) in [0, 1]
        target_size: Target size for both dimensions
        use_torch: If True, use PyTorch for exact match with Apple's preprocessing.
                   If False, use PIL (slightly different results).
    """
    if use_torch:
        try:
            import torch
            import torch.nn.functional as F
            # HWC -> CHW -> NCHW
            image_pt = torch.from_numpy(image.copy()).float().permute(2, 0, 1)[None]
            resized_pt = F.interpolate(image_pt, size=(target_size, target_size), 
                                        mode='bilinear', align_corners=True)
            return resized_pt[0].permute(1, 2, 0).numpy()
        except ImportError:
            pass  # Fall back to PIL
    
    # PIL fallback (slightly different interpolation)
    h, w = image.shape[:2]
    img_pil = Image.fromarray((image * 255).astype(np.uint8))
    img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img_resized).astype(np.float32) / 255.0


def generate(
    image: np.ndarray,
    model,
    f_px: float,
    orig_width: int,
    orig_height: int,
    internal_shape: tuple = (1536, 1536),
) -> Gaussians3D:
    """Run inference and return unprojected Gaussians.
    
    Args:
        image: Preprocessed image (H, W, 3) in [0, 1], resized to internal_shape
        model: Sharp predictor model with loaded weights
        f_px: Focal length in pixels
        orig_width: Original image width
        orig_height: Original image height
        internal_shape: Internal processing resolution
        
    Returns:
        Gaussians3D in metric space
    """
    # Convert to MLX array (NHWC)
    x = mx.array(image[None])
    disparity_factor = mx.array([f_px / orig_width])
    
    # Run model
    gaussians_ndc = model(x, disparity_factor)
    mx.eval(gaussians_ndc)
    
    # Prepare for unprojection
    opacities = gaussians_ndc.opacities
    if opacities.ndim == 3:
        opacities = mx.squeeze(opacities, axis=-1)
    
    gaussians_ndc_utils = Gaussians3D(
        mean_vectors=gaussians_ndc.means,
        singular_values=gaussians_ndc.scales,
        quaternions=gaussians_ndc.quaternions,
        colors=gaussians_ndc.colors,
        opacities=opacities,
    )
    
    # Build intrinsics
    scale_x = internal_shape[0] / orig_width
    scale_y = internal_shape[1] / orig_height
    intrinsics = mx.array([
        [f_px * scale_x, 0, orig_width / 2 * scale_x, 0],
        [0, f_px * scale_y, orig_height / 2 * scale_y, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=mx.float32)
    
    extrinsics = mx.eye(4, dtype=mx.float32)
    
    # Unproject to metric space
    return unproject_gaussians(gaussians_ndc_utils, extrinsics, intrinsics, internal_shape)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D Gaussians from an image")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="Output directory or .ply file path")
    parser.add_argument("-c", "--checkpoint", type=Path, default=Path("sharp.safetensors"))
    parser.add_argument("--no-torch", action="store_true", help="Don't use PyTorch for resize")
    args = parser.parse_args()
    
    # Check if output is a directory or file
    if args.output.suffix == '.ply':
        output_ply = args.output
        output_ply.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        output_ply = args.output / f"{args.input.stem}.ply"
    
    print(f"Loading image: {args.input}")
    image, f_px, orig_height, orig_width = load_image(args.input)
    print(f"  Size: {orig_width}x{orig_height}, focal: {f_px:.1f}px")
    
    internal_shape = (1536, 1536)
    image_resized = resize_image(image, 1536, use_torch=not args.no_torch)
    
    print(f"\nLoading model: {args.checkpoint}")
    start = time.perf_counter()
    model = create_predictor()
    load_weights(model, args.checkpoint, verbose=False)
    print(f"  Loaded in {time.perf_counter() - start:.2f}s")
    
    print("\nGenerating 3D Gaussians...")
    start = time.perf_counter()
    gaussians = generate(image_resized, model, f_px, orig_width, orig_height, internal_shape)
    gen_time = time.perf_counter() - start
    
    num_gaussians = gaussians.mean_vectors.shape[1]
    print(f"  Generated {num_gaussians:,} Gaussians in {gen_time:.2f}s")
    
    print(f"\nSaving: {output_ply}")
    start = time.perf_counter()
    save_ply(gaussians, f_px, internal_shape, output_ply)
    print(f"  Saved in {time.perf_counter() - start:.2f}s")
    
    print(f"\nDone!")


if __name__ == "__main__":
    main()
