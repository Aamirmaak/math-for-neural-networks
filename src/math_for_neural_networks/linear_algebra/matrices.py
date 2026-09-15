"""
Matrix operations for the Math for Neural Networks toolkit.

Matrices are represented as 2D NumPy arrays with shape (m, n).

Neural network connection:
- Weight matrices transform input representations: y = Wx + b
- Each row of W is a neuron's weight vector
- Matrix multiplication IS the dense/linear layer

Mathematics:
- Matrix addition: (A + B)_ij = A_ij + B_ij
- Scalar multiplication: (c * A)_ij = c * A_ij
- Transpose: (A^T)_ij = A_ji
- Matrix multiplication: (AB)_ij = sum_k A_ik * B_kj
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _to_matrix(
    data: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Convert input to a 2D NumPy float64 array.

    Args:
        data: A nested sequence of numbers or a NumPy array.

    Returns:
        2D NumPy array with dtype float64.

    Raises:
        ValueError: If the input is not 2-dimensional or is empty.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Input must be a matrix, not a scalar.")
    if arr.ndim == 1:
        raise ValueError(
            f"Input is 1D with shape {arr.shape}. For vectors, use the vector functions."
        )
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2-dimensional, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Matrix must be non-empty.")
    return arr


def matrix_add(
    a: Sequence[Sequence[float]] | NDArray,
    b: Sequence[Sequence[float]] | NDArray,
) -> NDArray[np.float64]:
    """Compute matrix addition: result_ij = A_ij + B_ij.

    Both matrices must have the same shape.

    Args:
        a: First matrix (2D).
        b: Second matrix (2D).

    Returns:
        New matrix of same shape.

    Raises:
        ValueError: If matrices have different shapes.
    """
    ma = _to_matrix(a)
    mb = _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError(f"Matrix shapes must match for addition, got {ma.shape} and {mb.shape}.")
    return ma + mb


def matrix_subtract(
    a: Sequence[Sequence[float]] | NDArray,
    b: Sequence[Sequence[float]] | NDArray,
) -> NDArray[np.float64]:
    """Compute matrix subtraction: result_ij = A_ij - B_ij.

    Both matrices must have the same shape.

    Args:
        a: First matrix (2D).
        b: Second matrix (2D).

    Returns:
        New matrix of same shape.

    Raises:
        ValueError: If matrices have different shapes.
    """
    ma = _to_matrix(a)
    mb = _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError(
            f"Matrix shapes must match for subtraction, got {ma.shape} and {mb.shape}."
        )
    return ma - mb


def scalar_multiply_matrix(c: float, m: Sequence[Sequence[float]] | NDArray) -> NDArray[np.float64]:
    """Compute scalar multiplication: result_ij = c * A_ij.

    Neural network connection: Scaling weight matrices during
    learning rate updates.

    Args:
        c: Scalar multiplier.
        m: Input matrix (2D).

    Returns:
        New matrix of same shape, scaled by c.
    """
    mat = _to_matrix(m)
    return c * mat


def transpose(m: Sequence[Sequence[float]] | NDArray) -> NDArray[np.float64]:
    """Compute the matrix transpose: result_ij = A_ji.

    For a matrix A with shape (m, n), transpose has shape (n, m).

    Neural network connection:
    - Backpropagation requires transposing weight matrices
    - Attention: computing K^T for QK^T
    - Many gradient formulas involve transposes

    Mathematical property:
    - (A^T)^T = A
    - (AB)^T = B^T A^T

    Args:
        m: Input matrix (2D).

    Returns:
        Transposed matrix with swapped dimensions.
    """
    mat = _to_matrix(m)
    return mat.T


def identity_matrix(n: int) -> NDArray[np.float64]:
    """Create an n x n identity matrix.

    The identity matrix I satisfies: AI = IA = A for any compatible A.

    Args:
        n: Dimension of the identity matrix (n x n).

    Returns:
        Identity matrix of shape (n, n).

    Raises:
        ValueError: If n <= 0.
    """
    if n <= 0:
        raise ValueError(f"Identity matrix dimension must be positive, got {n}.")
    return np.eye(n, dtype=np.float64)
