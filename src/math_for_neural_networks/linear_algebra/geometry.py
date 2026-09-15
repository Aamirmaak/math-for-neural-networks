"""
Vector geometry: projections and linear transformations.

Mathematics:
- Vector projection of a onto b: proj_b(a) = ((a · b) / (b · b)) * b
- Orthogonal component: a - proj_b(a)
- Reconstruction: a = proj_b(a) + (a - proj_b(a))

Linear transformations:
- Scaling: multiply coordinates by a factor
- Rotation: rotate vectors by an angle
- Reflection: mirror vectors across an axis
- Shear: displace coordinates proportionally

Neural network connection:
- Projection: understanding how representations decompose into components
- Linear transformations: y = Wx is exactly a linear transformation
- Each layer applies a linear transformation followed by nonlinearity
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.linear_algebra.vectors import (
    _to_vector,
    dot_product,
)


def project_vector(
    a: Sequence[float] | NDArray,
    b: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """Compute the projection of vector a onto vector b.

    proj_b(a) = ((a · b) / (b · b)) * b

    This decomposes a into:
    - Parallel component: proj_b(a)  (lies along b)
    - Orthogonal component: a - proj_b(a)  (perpendicular to b)

    Reconstruction: a = proj_b(a) + (a - proj_b(a))

    Neural network connection:
    - Understanding representation geometry
    - How much of one representation lies in the direction of another
    - Principal component analysis uses projections

    Args:
        a: Vector to project (1D).
        b: Direction to project onto (1D).

    Returns:
        Projected vector (1D), same shape as a.

    Raises:
        ValueError: If vectors have different lengths or b is zero.
    """
    va = _to_vector(a)
    vb = _to_vector(b)

    if va.shape != vb.shape:
        raise ValueError(f"Vector shapes must match for projection, got {va.shape} and {vb.shape}.")

    # b · b (denominator)
    dot_bb = dot_product(vb, vb)

    if dot_bb == 0.0:
        raise ValueError("Cannot project onto the zero vector. The zero vector has no direction.")

    # proj_b(a) = ((a · b) / (b · b)) * b
    dot_ab = dot_product(va, vb)
    scalar = dot_ab / dot_bb
    return scalar * vb


def orthogonal_component(
    a: Sequence[float] | NDArray,
    b: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """Compute the component of a orthogonal to b.

    orth_b(a) = a - proj_b(a)

    The orthogonal component is perpendicular to b, meaning:
    (a - proj_b(a)) · b = 0

    Args:
        a: Input vector (1D).
        b: Direction vector (1D).

    Returns:
        Orthogonal component (1D), same shape as a.
    """
    va = _to_vector(a)
    proj = project_vector(va, b)
    return va - proj


def verify_projection(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> dict[str, Any]:
    """Verify the decomposition a = proj_b(a) + orth_b(a).

    Returns a dictionary with verification metrics:
    - parallel: the projection (parallel to b)
    - orthogonal: the orthogonal component
    - reconstruction_error: ||a - (proj + orth)||
    - orthogonality_check: |orth · b| (should be ~0)
    - sum_check: ||a - (parallel + orthogonal)||

    Args:
        a: Input vector (1D).
        b: Direction vector (1D).

    Returns:
        Dictionary with verification results.
    """
    va = _to_vector(a)
    proj = project_vector(va, b)
    orth = orthogonal_component(va, b)

    # Reconstruction check: a = proj + orth
    reconstruction_error = float(np.sqrt(dot_product(va - (proj + orth), va - (proj + orth))))

    # Orthogonality check: orth · b should be 0
    orthogonality_check = abs(dot_product(orth, b))

    return {
        "parallel": proj,
        "orthogonal": orth,
        "reconstruction_error": reconstruction_error,
        "orthogonality_check": orthogonality_check,
    }


# --- Linear Transformations ---


def scaling_matrix(sx: float, sy: float) -> NDArray[np.float64]:
    """Create a 2D scaling transformation matrix.

    M = [[sx,  0],
         [ 0, sy]]

    Transforms: (x, y) -> (sx * x, sy * y)

    Neural network connection: Scaling activations is similar to
    multiplying by a diagonal weight matrix.

    Args:
        sx: Scale factor for x-axis.
        sy: Scale factor for y-axis.

    Returns:
        2x2 scaling matrix.
    """
    return np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)


def rotation_matrix(angle_rad: float) -> NDArray[np.float64]:
    """Create a 2D rotation matrix for counterclockwise rotation.

    M = [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]

    Transforms: rotates vectors by angle theta counterclockwise.

    Properties:
    - Orthogonal: M^T M = I
    - Preserves lengths: ||Mx|| = ||x||
    - Preserves angles between vectors

    Neural network connection: Rotation matrices are orthogonal and
    preserve norm — understanding them helps with weight initialization
    and understanding rotation invariance.

    Args:
        angle_rad: Rotation angle in radians.

    Returns:
        2x2 rotation matrix.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def reflection_matrix_x() -> NDArray[np.float64]:
    """Create a 2D reflection matrix across the x-axis.

    M = [[1,  0],
         [0, -1]]

    Transforms: (x, y) -> (x, -y)
    """
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def reflection_matrix_y() -> NDArray[np.float64]:
    """Create a 2D reflection matrix across the y-axis.

    M = [[-1, 0],
         [ 0, 1]]

    Transforms: (x, y) -> (-x, y)
    """
    return np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)


def shear_matrix(shear_x: float, shear_y: float) -> NDArray[np.float64]:
    """Create a 2D shear transformation matrix.

    M = [[1,       shear_y],
         [shear_x,      1]]

    Transforms: (x, y) -> (x + shear_y * y, shear_x * x + y)

    Neural network connection: Shear transforms distort the input space,
    similar to how non-linear activations warp representations.

    Args:
        shear_x: Shear factor in x-direction.
        shear_y: Shear factor in y-direction.

    Returns:
        2x2 shear matrix.
    """
    return np.array([[1.0, shear_y], [shear_x, 1.0]], dtype=np.float64)


def apply_transformation(
    matrix: NDArray,
    vector: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """Apply a linear transformation matrix to a vector.

    Computes: result = M @ v

    Args:
        matrix: Transformation matrix (2D).
        vector: Input vector (1D).

    Returns:
        Transformed vector (1D).
    """
    from math_for_neural_networks.linear_algebra.matrices import _to_matrix
    from math_for_neural_networks.linear_algebra.operations import matrix_vector_multiply

    mat = _to_matrix(matrix)
    vec = _to_vector(vector)
    return matrix_vector_multiply(mat, vec)
