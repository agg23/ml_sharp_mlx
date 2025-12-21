# Sharp MLX
# Apple Sharp model (3D Gaussian Splatting from single image) - MLX implementation
#
# Usage:
#   from sharp_mlx import create_predictor, load_weights
#   model = create_predictor()
#   load_weights(model, "sharp.safetensors")
#   gaussians = model(image, disparity_factor)

from .predictor import SharpPredictor, create_predictor
from .weights import load_weights
from .gaussian_utils import Gaussians3D, unproject_gaussians, save_ply

__all__ = [
    "SharpPredictor",
    "create_predictor", 
    "load_weights",
    "Gaussians3D",
    "unproject_gaussians",
    "save_ply",
]
