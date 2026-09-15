"""
Vector norms and distance for the Math for Neural Networks toolkit.

Neural network connection:
- L2 norm: weight regularization (L2 penalty), gradient clipping
- L1 norm: sparse representations, L1 regularization
- Distance: measuring similarity/dissimilarity in representation space
- Normalizing vectors to unit length is common in embedding spaces

Mathematics:
- L1 norm: ||v||_1 = sum(|v_i|)
- L2 norm: ||v||_2 = sqrt(sum(v_i^2))
- Euclidean distance: d(a, b) = ||a - b||_2

Norm properties:
1. Non-negativity: ||v|| >= 0
2. Definiteness: ||v|| = 0 if and only if v = 0
3. Homogeneity: ||c * v|| = |c| * ||v||
4. Triangle inequality: ||a + b|| <= ||a|| + ||b||
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.linear_algebra.vectors import _to_vector


def l1_norm(v: Sequence[float] | NDArray) -> float:
    """Compute the L1 (Manhattan) norm: ||v||_1 = sum(|v_i|).

    The L1 norm measures the total absolute deviation from zero.

    Neural network connection:
    - L1 regularization encourages sparsity in weights
    - Used in Lasso regression
    - Manhattan distance between points

    Args:
        v: Input vector (1D).

    Returns:
        L1 norm as a non-negative scalar.
    """
    vec = _to_vector(v)
    # Educational implementation: sum of absolute values
    result = 0.0
    for vi in vec:
        result += abs(vi)
    return result


def l2_norm(v: Sequence[float] | NDArray) -> float:
    """Compute the L2 (Euclidean) norm: ||v||_2 = sqrt(sum(v_i^2)).

    The L2 norm measures the straight-line distance from the origin.

    Neural network connection:
    - L2 regularization (weight decay): penalizes large weights
    - Gradient clipping by norm: rescale gradient if ||g|| > threshold
    - Normalizing embeddings to unit length

    Args:
        v: Input vector (1D).

    Returns:
        L2 norm as a non-negative scalar.
    """
    vec = _to_vector(v)
    # Educational implementation: sqrt(sum of squares)
    sum_sq = 0.0
    for vi in vec:
        sum_sq += vi * vi
    return float(np.sqrt(sum_sq))


def euclidean_distance(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> float:
    """Compute the Euclidean distance between two vectors: d(a,b) = ||a - b||_2.

    Neural network connection:
    - Distance in representation space
    - k-nearest neighbors uses distance
    - Clustering algorithms minimize within-cluster distances

    Args:
        a: First vector (1D).
        b: Second vector (1D).

    Returns:
        Non-negative distance scalar.

    Raises:
        ValueError: If vectors have different lengths.
    """
    va = _to_vector(a)
    vb = _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"Vector shapes must match for distance, got {va.shape} and {vb.shape}.")
    # d(a, b) = ||a - b||_2
    diff = va - vb
    sum_sq = 0.0
    for di in diff:
        sum_sq += di * di
    return float(np.sqrt(sum_sq))
