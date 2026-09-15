"""
Cosine similarity for the Math for Neural Networks toolkit.

Mathematics:
    cos(theta) = (a · b) / (||a|| * ||b||)

The cosine similarity measures the cosine of the angle between two vectors.
It ranges from -1 (opposite directions) to 1 (same direction),
with 0 indicating orthogonality.

Neural network connection:
- Embeddings: cosine similarity measures semantic similarity
  - "king" - "man" + "woman" ≈ "woman" (word2vec arithmetic)
- Attention: scaled dot-product attention uses normalized dot products
- Retrieval: finding similar documents/items by embedding similarity
- Representation learning: objectives that maximize cosine similarity
  of similar pairs and minimize it for dissimilar pairs

Zero vector handling:
- If either vector is zero, cosine similarity is undefined (0/0)
- We raise ValueError explicitly rather than returning NaN silently
- This forces the caller to handle the degenerate case
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.linear_algebra.vectors import _to_vector, dot_product


def cosine_similarity(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> float:
    """Compute cosine similarity between two vectors.

    cos(theta) = (a · b) / (||a|| * ||b||)

    Returns a value in [-1, 1]:
    -  1.0: vectors point in the same direction (angle = 0)
    -  0.0: vectors are orthogonal (angle = 90 degrees)
    - -1.0: vectors point in opposite directions (angle = 180 degrees)

    Neural network connection:
    - Semantic similarity between word/sentence embeddings
    - Attention score computation (before softmax scaling)
    - Contrastive learning objectives

    Args:
        a: First vector (1D).
        b: Second vector (1D).

    Returns:
        Cosine similarity in [-1, 1].

    Raises:
        ValueError: If vectors have different lengths or either is zero.
    """
    va = _to_vector(a)
    vb = _to_vector(b)

    if va.shape != vb.shape:
        raise ValueError(
            f"Vector shapes must match for cosine similarity, got {va.shape} and {vb.shape}."
        )

    # Compute dot product: a · b
    dot_ab = dot_product(va, vb)

    # Compute norms: ||a|| and ||b||
    norm_a = float(np.sqrt(dot_product(va, va)))
    norm_b = float(np.sqrt(dot_product(vb, vb)))

    # Handle zero vectors explicitly
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            "Cosine similarity is undefined for zero vectors. "
            f"||a|| = {norm_a}, ||b|| = {norm_b}. "
            "Handle zero vectors before calling cosine_similarity."
        )

    # cos(theta) = (a · b) / (||a|| * ||b||)
    return dot_ab / (norm_a * norm_b)
