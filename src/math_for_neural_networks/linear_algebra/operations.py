"""
Core linear algebra operations for the Math for Neural Networks toolkit.

This module implements matrix multiplication and matrix-vector multiplication
with educational clarity, showing the mathematical formulas directly.

Neural network connection:
- Matrix-vector multiplication IS a dense/linear layer
- Matrix-matrix multiplication processes batches in parallel
- The formula C_ij = sum_k A_ik * B_kj is the foundation of neural computation

Mathematics:
- Matrix-vector: y = Ax, y_i = sum_j A_ij * x_j
- Matrix-matrix: C = AB, C_ij = sum_k A_ik * B_kj
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.linear_algebra.matrices import _to_matrix
from math_for_neural_networks.linear_algebra.vectors import _to_vector


def matrix_vector_multiply(
    m: Sequence[Sequence[float]] | NDArray,
    v: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """Compute matrix-vector multiplication: y = Ax.

    For matrix A with shape (m, n) and vector x with shape (n,),
    the result y has shape (m,).

    Mathematical formula:
        y_i = sum_j A_ij * x_j

    This is the fundamental operation of a neural network linear layer:
        y = Wx + b
    where W is the weight matrix and x is the input.

    Args:
        m: Matrix with shape (m, n).
        v: Vector with shape (n,).

    Returns:
        Result vector with shape (m,).

    Raises:
        ValueError: If dimensions are incompatible.
    """
    mat = _to_matrix(m)
    vec = _to_vector(v)

    if mat.shape[1] != vec.shape[0]:
        raise ValueError(
            f"Incompatible dimensions for matrix-vector multiplication: "
            f"matrix has {mat.shape[1]} columns, vector has {vec.shape[0]} elements. "
            f"Matrix columns must equal vector length."
        )

    # Educational implementation: y_i = sum_j A_ij * x_j
    m_rows, n_cols = mat.shape
    result = np.zeros(m_rows, dtype=np.float64)
    for i in range(m_rows):
        for j in range(n_cols):
            result[i] += mat[i, j] * vec[j]

    return result


def matrix_multiply(
    a: Sequence[Sequence[float]] | NDArray,
    b: Sequence[Sequence[float]] | NDArray,
) -> NDArray[np.float64]:
    """Compute matrix multiplication: C = AB.

    For matrix A with shape (m, n) and matrix B with shape (n, p),
    the result C has shape (m, p).

    Mathematical formula:
        C_ij = sum_k A_ik * B_kj

    Neural network connection:
    - Processing a batch of inputs: Y = XW^T
    - Composing two linear transformations
    - Attention mechanism: scores = QK^T

    IMPORTANT: Matrix multiplication is NOT commutative in general.
    AB != BA (shapes may differ, and even when shapes match, values differ).

    Mathematical properties:
    - Associative: (AB)C = A(BC)
    - Distributive: A(B + C) = AB + AC
    - NOT commutative: AB != BA in general

    Args:
        a: Left matrix with shape (m, n).
        b: Right matrix with shape (n, p).

    Returns:
        Result matrix with shape (m, p).

    Raises:
        ValueError: If dimensions are incompatible.
    """
    ma = _to_matrix(a)
    mb = _to_matrix(b)

    if ma.shape[1] != mb.shape[0]:
        raise ValueError(
            f"Incompatible dimensions for matrix multiplication: "
            f"left matrix has {ma.shape[1]} columns, "
            f"right matrix has {mb.shape[0]} rows. "
            f"Left columns must equal right rows."
        )

    m, n = ma.shape
    n2, p = mb.shape

    # Educational implementation: C_ij = sum_k A_ik * B_kj
    result = np.zeros((m, p), dtype=np.float64)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += ma[i, k] * mb[k, j]

    return result
