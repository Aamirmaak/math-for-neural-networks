"""
Vector operations for the Math for Neural Networks toolkit.

Vectors are represented as 1D NumPy arrays with shape (n,).

Neural network connection:
- Vectors represent feature vectors, embeddings, and hidden states
- Dot products between vectors compute attention scores and weighted sums
- Vector addition combines representations (residual connections)
- Scalar multiplication scales feature activations

Mathematics:
- Vector addition: (a + b)_i = a_i + b_i
- Scalar multiplication: (c * v)_i = c * v_i
- Dot product: a · b = sum(a_i * b_i)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _to_vector(data: Sequence[float] | NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert input to a 1D NumPy float64 array.

    Args:
        data: A sequence of numbers or a NumPy array.

    Returns:
        1D NumPy array with dtype float64.

    Raises:
        ValueError: If the input is empty or not 1-dimensional.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Input must be a vector, not a scalar.")
    if arr.ndim != 1:
        raise ValueError(
            f"Input must be 1-dimensional, got shape {arr.shape}. "
            "For matrices, use the matrix functions."
        )
    if arr.size == 0:
        raise ValueError("Vector must be non-empty.")
    return arr


def vector_add(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> NDArray[np.float64]:
    """Compute vector addition: result_i = a_i + b_i.

    Both vectors must have the same length.

    Neural network connection: Residual connections add a skip connection
    vector to the main path: h = f(x) + x

    Args:
        a: First vector (1D).
        b: Second vector (1D).

    Returns:
        New vector of same length.

    Raises:
        ValueError: If vectors have different lengths.
    """
    va = _to_vector(a)
    vb = _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"Vector shapes must match for addition, got {va.shape} and {vb.shape}.")
    return va + vb


def vector_subtract(
    a: Sequence[float] | NDArray, b: Sequence[float] | NDArray
) -> NDArray[np.float64]:
    """Compute vector subtraction: result_i = a_i - b_i.

    Both vectors must have the same length.

    Neural network connection: Computing the difference between two
    representations, e.g., in attention residual updates.

    Args:
        a: First vector (1D).
        b: Second vector (1D).

    Returns:
        New vector of same length.

    Raises:
        ValueError: If vectors have different lengths.
    """
    va = _to_vector(a)
    vb = _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError(
            f"Vector shapes must match for subtraction, got {va.shape} and {vb.shape}."
        )
    return va - vb


def scalar_multiply(c: float, v: Sequence[float] | NDArray) -> NDArray[np.float64]:
    """Compute scalar multiplication: result_i = c * v_i.

    Neural network connection: Scaling activations by a weight or
    learning rate in gradient updates: w = w - lr * grad

    Args:
        c: Scalar multiplier.
        v: Input vector (1D).

    Returns:
        New vector of same length, scaled by c.
    """
    vec = _to_vector(v)
    return c * vec


def dot_product(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> float:
    """Compute the dot product: a · b = sum(a_i * b_i).

    The dot product measures the degree to which two vectors point
    in the same direction. It equals ||a|| * ||b|| * cos(theta).

    Neural network connection:
    - Neuron computation: z = sum(w_i * x_i) + b = w · x + b
    - Attention: score = q · k (scaled dot-product attention)
    - Cosine similarity is normalized dot product

    Mathematical properties:
    - Commutative: a · b = b · a
    - Distributive: a · (b + c) = a · b + a · c
    - Non-negative for self-dot: a · a = ||a||^2 >= 0

    Args:
        a: First vector (1D).
        b: Second vector (1D).

    Returns:
        Scalar result.

    Raises:
        ValueError: If vectors have different lengths.
    """
    va = _to_vector(a)
    vb = _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError(
            f"Vector shapes must match for dot product, got {va.shape} and {vb.shape}."
        )
    # Educational implementation: sum(a_i * b_i)
    result = 0.0
    for ai, bi in zip(va, vb, strict=True):
        result += ai * bi
    return float(result)
