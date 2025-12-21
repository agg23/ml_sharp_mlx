"""Pure MLX implementation of Gaussian unprojection utilities.

Based on Apple's Sharp: https://github.com/apple/ml-tract
Original code Copyright (C) 2025 Apple Inc. Licensed under Apple Sample Code License.
"""

import mlx.core as mx
import numpy as np  # Only for PLY saving
from typing import NamedTuple, Tuple

class Gaussians3D(NamedTuple):
    """Represents a collection of 3D Gaussians (MLX version)."""
    mean_vectors: mx.array      # [B, N, 3]
    singular_values: mx.array   # [B, N, 3]
    quaternions: mx.array       # [B, N, 4]
    colors: mx.array            # [B, N, 3]
    opacities: mx.array         # [B, N]


# ============================================================================
# Linear Algebra Utilities
# ============================================================================

def eyes(dim: int, shape: Tuple[int, ...], dtype=mx.float32) -> mx.array:
    """Create batch of identity matrices."""
    eye = mx.eye(dim, dtype=dtype)
    return mx.broadcast_to(eye, shape + (dim, dim))


def get_cross_product_matrix(vectors: mx.array) -> mx.array:
    """Generate cross product matrix for vector exterior product.
    
    For vector v = [x, y, z], returns the skew-symmetric matrix:
    [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    """
    if vectors.shape[-1] != 3:
        raise ValueError("Only 3-dimensional vectors are supported")
    
    # Extract components
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]
    zeros = mx.zeros_like(x)
    
    # Build skew-symmetric matrix
    # Row 0: [0, -z, y]
    # Row 1: [z, 0, -x]
    # Row 2: [-y, x, 0]
    row0 = mx.stack([zeros, -z, y], axis=-1)
    row1 = mx.stack([z, zeros, -x], axis=-1)
    row2 = mx.stack([-y, x, zeros], axis=-1)
    
    return mx.stack([row0, row1, row2], axis=-2)



def rotation_matrices_from_quaternions(quaternions: mx.array) -> mx.array:
    """Convert batch of quaternions into rotation matrices.
    
    Args:
        quaternions: Array of shape [..., 4] with (w, x, y, z) convention
        
    Returns:
        Rotation matrices of shape [..., 3, 3]
    """
    shape = quaternions.shape[:-1]
    
    # Normalize quaternions
    norm = mx.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions = quaternions / norm
    
    w = quaternions[..., 0]
    x = quaternions[..., 1]
    y = quaternions[..., 2]
    z = quaternions[..., 3]
    
    # Compute rotation matrix elements directly (more efficient than Rodrigues)
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Build rotation matrix
    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)
    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = 1 - 2*(xx + yy)
    
    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    
    return mx.stack([row0, row1, row2], axis=-2)


def quaternions_from_rotation_matrices(matrices: mx.array) -> mx.array:
    """Convert batch of rotation matrices to quaternions (fully vectorized).
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        matrices: Rotation matrices of shape [..., 3, 3]
        
    Returns:
        Quaternions of shape [..., 4] with (w, x, y, z) convention
    """
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"matrices have invalid shape {matrices.shape}")
    
    original_shape = matrices.shape[:-2]
    m = mx.reshape(matrices, (-1, 3, 3))
    n = m.shape[0]
    
    # Compute trace and diagonal elements
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    m01, m02, m10 = m[:, 0, 1], m[:, 0, 2], m[:, 1, 0]
    m12, m20, m21 = m[:, 1, 2], m[:, 2, 0], m[:, 2, 1]
    
    # Allocate output
    quaternions = mx.zeros((n, 4), dtype=matrices.dtype)
    
    # Case 1: trace > 0
    case1 = trace > 0
    s1 = 0.5 / mx.sqrt(mx.maximum(trace + 1.0, 1e-10))
    w1 = 0.25 / s1
    x1 = (m21 - m12) * s1
    y1 = (m02 - m20) * s1
    z1 = (m10 - m01) * s1
    q1 = mx.stack([w1, x1, y1, z1], axis=-1)
    
    # Case 2: m00 > m11 and m00 > m22
    case2 = ~case1 & (m00 > m11) & (m00 > m22)
    s2 = 2.0 * mx.sqrt(mx.maximum(1.0 + m00 - m11 - m22, 1e-10))
    w2 = (m21 - m12) / s2
    x2 = 0.25 * s2
    y2 = (m01 + m10) / s2
    z2 = (m02 + m20) / s2
    q2 = mx.stack([w2, x2, y2, z2], axis=-1)
    
    # Case 3: m11 > m22
    case3 = ~case1 & ~case2 & (m11 > m22)
    s3 = 2.0 * mx.sqrt(mx.maximum(1.0 + m11 - m00 - m22, 1e-10))
    w3 = (m02 - m20) / s3
    x3 = (m01 + m10) / s3
    y3 = 0.25 * s3
    z3 = (m12 + m21) / s3
    q3 = mx.stack([w3, x3, y3, z3], axis=-1)
    
    # Case 4: else (m22 is largest)
    s4 = 2.0 * mx.sqrt(mx.maximum(1.0 + m22 - m00 - m11, 1e-10))
    w4 = (m10 - m01) / s4
    x4 = (m02 + m20) / s4
    y4 = (m12 + m21) / s4
    z4 = 0.25 * s4
    q4 = mx.stack([w4, x4, y4, z4], axis=-1)
    
    # Combine cases using where
    quaternions = mx.where(case1[:, None], q1, 
                    mx.where(case2[:, None], q2,
                      mx.where(case3[:, None], q3, q4)))
    
    # Normalize
    quaternions = quaternions / mx.linalg.norm(quaternions, axis=-1, keepdims=True)
    
    return mx.reshape(quaternions, original_shape + (4,))


# ============================================================================
# Gaussian Utilities
# ============================================================================


def compose_covariance_matrices(
    quaternions: mx.array, 
    singular_values: mx.array
) -> mx.array:
    """Compose covariance matrices from rotation and scale.
    
    Args:
        quaternions: Shape [..., 4] with (w, x, y, z) convention
        singular_values: Shape [..., 3] scales
        
    Returns:
        Covariance matrices of shape [..., 3, 3]
    """
    rotations = rotation_matrices_from_quaternions(quaternions)
    
    # Create diagonal matrix from singular values squared
    # Σ = R @ D² @ R.T
    scales_sq = singular_values ** 2
    
    # Build diagonal matrix efficiently
    zeros = mx.zeros_like(scales_sq[..., 0])
    row0 = mx.stack([scales_sq[..., 0], zeros, zeros], axis=-1)
    row1 = mx.stack([zeros, scales_sq[..., 1], zeros], axis=-1)
    row2 = mx.stack([zeros, zeros, scales_sq[..., 2]], axis=-1)
    diagonal_matrix = mx.stack([row0, row1, row2], axis=-2)
    
    return rotations @ diagonal_matrix @ mx.transpose(rotations, list(range(len(rotations.shape)-2)) + [-1, -2])


def decompose_covariance_matrices(
    covariance_matrices: mx.array
) -> Tuple[mx.array, mx.array]:
    """Decompose 3D covariance matrices into quaternions and singular values.
    
    Args:
        covariance_matrices: Shape [..., 3, 3]
        
    Returns:
        Tuple of (quaternions [..., 4], singular_values [..., 3])
    """
    original_dtype = covariance_matrices.dtype
    
    # SVD decomposition: Σ = U @ S @ V.T
    # Note: MLX SVD only runs on CPU currently
    U, singular_values_2, Vt = mx.linalg.svd(covariance_matrices, stream=mx.cpu)
    rotations = U
    
    # Check for reflection matrices and fix them
    # For MLX, we compute determinant via the formula for 3x3
    r = rotations
    det = (r[..., 0, 0] * (r[..., 1, 1] * r[..., 2, 2] - r[..., 1, 2] * r[..., 2, 1]) -
           r[..., 0, 1] * (r[..., 1, 0] * r[..., 2, 2] - r[..., 1, 2] * r[..., 2, 0]) +
           r[..., 0, 2] * (r[..., 1, 0] * r[..., 2, 1] - r[..., 1, 1] * r[..., 2, 0]))
    
    reflection_mask = det < 0
    num_reflections = int(mx.sum(reflection_mask).item())
    
    if num_reflections > 0:
        print(
            "Received %d reflection matrices from SVD. Flipping them to rotations.",
            num_reflections,
        )
        # Flip the last column to convert reflection to rotation
        flip_col = mx.ones_like(rotations[..., :, -1:])
        flip_col = mx.where(reflection_mask[..., None, None], -flip_col, flip_col)
        rotations = mx.concatenate([rotations[..., :, :-1], rotations[..., :, -1:] * flip_col], axis=-1)
    
    # Convert rotation matrices to quaternions
    quaternions = quaternions_from_rotation_matrices(rotations)
    
    # Singular values are sqrt of eigenvalues
    singular_values = mx.sqrt(singular_values_2)
    
    return quaternions, singular_values


def get_unprojection_matrix(
    extrinsics: mx.array,
    intrinsics: mx.array,
    image_shape: Tuple[int, int],
) -> mx.array:
    """Compute unprojection matrix to transform Gaussians to Euclidean space.
    
    Args:
        extrinsics: 4x4 extrinsics matrix
        intrinsics: 4x4 intrinsics matrix
        image_shape: (width, height) of the input image
        
    Returns:
        4x4 unprojection matrix
    """
    image_width, image_height = image_shape
    
    # NDC matrix: converts OpenCV pixel coords to NDC coords
    ndc_matrix = mx.array([
        [2.0 / image_width, 0.0, -1.0, 0.0],
        [0.0, 2.0 / image_height, -1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=intrinsics.dtype)
    
    return mx.linalg.inv(ndc_matrix @ intrinsics @ extrinsics, stream=mx.cpu)


def apply_transform(
    gaussians: Gaussians3D, 
    transform: mx.array
) -> Gaussians3D:
    """Apply an affine transformation to 3D Gaussians.
    
    Args:
        gaussians: The Gaussians to transform
        transform: Affine transform with shape [3, 4]
        
    Returns:
        Transformed Gaussians
    """
    transform_linear = transform[..., :3, :3]  # [3, 3]
    transform_offset = transform[..., :3, 3]   # [3]
    
    # Transform mean positions
    mean_vectors = gaussians.mean_vectors @ mx.transpose(transform_linear) + transform_offset
    
    # Transform covariance matrices: R @ Σ @ R.T
    covariance_matrices = compose_covariance_matrices(
        gaussians.quaternions, gaussians.singular_values
    )
    covariance_matrices = (
        transform_linear @ covariance_matrices @ mx.transpose(transform_linear)
    )
    
    # Decompose back to quaternions and scales
    quaternions, singular_values = decompose_covariance_matrices(covariance_matrices)
    
    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )


def unproject_gaussians(
    gaussians_ndc: Gaussians3D,
    extrinsics: mx.array,
    intrinsics: mx.array,
    image_shape: Tuple[int, int],
) -> Gaussians3D:
    """Unproject Gaussians from NDC space to world coordinates.
    
    Args:
        gaussians_ndc: Gaussians in NDC space
        extrinsics: 4x4 camera extrinsics
        intrinsics: 4x4 camera intrinsics
        image_shape: (width, height) of the image
        
    Returns:
        Gaussians in world/metric coordinates
    """
    unprojection_matrix = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    gaussians = apply_transform(gaussians_ndc, unprojection_matrix[:3])
    mx.eval(gaussians)  # Ensure computation completes
    return gaussians


# ============================================================================
# Color Space Utilities
# ============================================================================

def linearRGB2sRGB(linear_rgb: mx.array) -> mx.array:
    """Convert linearRGB to sRGB."""
    THRESHOLD = 0.0031308
    
    return mx.where(
        linear_rgb <= THRESHOLD,
        linear_rgb * 12.92,
        1.055 * mx.power(mx.maximum(linear_rgb, THRESHOLD), 1/2.4) - 0.055
    )


def sRGB2linearRGB(srgb: mx.array) -> mx.array:
    """Convert sRGB to linearRGB."""
    THRESHOLD = 0.04045
    
    return mx.where(
        srgb <= THRESHOLD,
        srgb / 12.92,
        mx.power((mx.maximum(srgb, THRESHOLD) + 0.055) / 1.055, 2.4)
    )


def convert_rgb_to_spherical_harmonics(rgb: mx.array) -> mx.array:
    """Convert RGB to degree-0 spherical harmonics."""
    import math
    coeff_degree0 = math.sqrt(1.0 / (4.0 * math.pi))
    return (rgb - 0.5) / coeff_degree0


def convert_spherical_harmonics_to_rgb(sh0: mx.array) -> mx.array:
    """Convert degree-0 spherical harmonics to RGB."""
    import math
    coeff_degree0 = math.sqrt(1.0 / (4.0 * math.pi))
    return sh0 * coeff_degree0 + 0.5


# ============================================================================
# PLY I/O
# ============================================================================

def save_ply(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: Tuple[int, int],
    path,
) -> None:
    """Save Gaussians to a PLY file compatible with 3DGS viewers.
    
    Uses pure binary writing (no plyfile dependency).
    """
    from pathlib import Path
    import struct
    
    path = Path(path)
    
    # Convert MLX arrays to numpy
    xyz = np.array(gaussians.mean_vectors).reshape(-1, 3).astype(np.float32)
    scales = np.array(gaussians.singular_values).reshape(-1, 3).astype(np.float32)
    quaternions = np.array(gaussians.quaternions).reshape(-1, 4).astype(np.float32)
    colors_linear = np.array(gaussians.colors).reshape(-1, 3).astype(np.float32)
    opacities = np.array(gaussians.opacities).reshape(-1).astype(np.float32)
    
    N = len(xyz)
    
    # Convert linearRGB to sRGB
    THRESHOLD = 0.0031308
    colors_srgb = np.where(
        colors_linear <= THRESHOLD,
        colors_linear * 12.92,
        1.055 * np.power(np.maximum(colors_linear, THRESHOLD), 1/2.4) - 0.055
    )
    
    # Convert sRGB to spherical harmonics
    coeff = np.sqrt(1.0 / (4.0 * np.pi))
    colors_sh = ((colors_srgb - 0.5) / coeff).astype(np.float32)
    
    # Convert opacity to logit (inverse sigmoid)
    opacities_clamped = np.clip(opacities, 1e-6, 1 - 1e-6)
    opacity_logits = np.log(opacities_clamped / (1.0 - opacities_clamped)).astype(np.float32)
    
    # Convert scales to log
    scale_logits = np.log(np.maximum(scales, 1e-10)).astype(np.float32)
    
    # Normalize quaternions
    quat_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = (quaternions / np.maximum(quat_norms, 1e-8)).astype(np.float32)
    
    # Build PLY header
    image_height, image_width = image_shape
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
element extrinsic 16
property float extrinsic
element intrinsic 9
property float intrinsic
element image_size 2
property uint image_size
element frame 2
property int frame
element disparity 2
property float disparity
element color_space 1
property uchar color_space
element version 3
property uchar version
end_header
"""
    
    # Build vertex data as contiguous array for fast writing
    # Order: xyz (3), colors (3), opacity (1), scales (3), quaternions (4) = 14 floats per vertex
    vertex_data = np.zeros((N, 14), dtype=np.float32)
    vertex_data[:, 0:3] = xyz
    vertex_data[:, 3:6] = colors_sh
    vertex_data[:, 6] = opacity_logits
    vertex_data[:, 7:10] = scale_logits
    vertex_data[:, 10:14] = quaternions
    
    with open(path, 'wb') as f:
        # Write header
        f.write(header.encode('utf-8'))
        
        # Write vertex data (bulk write is much faster than loop)
        f.write(vertex_data.tobytes())
        
        # Write extrinsic (16 floats - identity matrix)
        extrinsic = np.eye(4, dtype=np.float32).flatten()
        f.write(extrinsic.tobytes())
        
        # Write intrinsic (9 floats - 3x3 matrix)
        intrinsic = np.array([
            f_px, 0, image_width * 0.5,
            0, f_px, image_height * 0.5,
            0, 0, 1
        ], dtype=np.float32)
        f.write(intrinsic.tobytes())
        
        # Write image_size (2 uints)
        f.write(struct.pack('<II', image_width, image_height))
        
        # Write frame (2 ints - num frames, num gaussians)
        f.write(struct.pack('<ii', 1, N))
        
        # Write disparity quantiles (2 floats)
        depths = xyz[:, 2]
        disparities = 1.0 / np.maximum(depths, 1e-6)
        q10, q90 = np.quantile(disparities, [0.1, 0.9])
        f.write(struct.pack('<ff', q10, q90))
        
        # Write color_space (1 uchar - 1 for sRGB with SH conversion)
        f.write(struct.pack('<B', 1))
        
        # Write version (3 uchars - 1, 5, 0)
        f.write(struct.pack('<BBB', 1, 5, 0))
