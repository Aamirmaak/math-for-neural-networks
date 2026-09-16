"""
Attention Mathematics
=====================

Scaled dot-product attention is the core operation in Transformers.

Mathematics:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    Q = queries   (n_q x d_k)
    K = keys      (n_k x d_k)
    V = values    (n_k x d_v)
    d_k = key dimension (used for scaling)

Shape flow:
    QK^T:      (n_q x d_k) @ (d_k x n_k) = (n_q x n_k)
    Scaled:    (n_q x n_k) / sqrt(d_k)
    Weights:   softmax along last axis -> (n_q x n_k), rows sum to 1
    Output:    (n_q x n_k) @ (n_k x d_v) = (n_q x d_v)

Why scale by 1/sqrt(d_k)?
    Without scaling, dot products grow with d_k.
    Large dot products push softmax into saturation (tiny gradients).
    Scaling by 1/sqrt(d_k) keeps variance of dot products ~1
    regardless of d_k.

Neural network connection:
    Attention is how Transformers "look at" relevant parts of the input.
    Q, K, V are linear transformations of the input embeddings.
    Multi-head attention runs multiple attention operations in parallel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """Compute numerically stable softmax along specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax output (probabilities sum to 1 along axis).
    """
    x = np.asarray(x, dtype=np.float64)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    result: NDArray = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return result


def scaled_dot_product_attention(
    Q: NDArray, K: NDArray, V: NDArray, mask: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: Queries, shape (n_q, d_k).
        K: Keys, shape (n_k, d_k).
        V: Values, shape (n_k, d_v).
        mask: Optional boolean mask, shape (n_q, n_k).
              True positions are allowed, False positions are masked.

    Returns:
        Tuple of:
            - Attention output, shape (n_q, d_v)
            - Attention weights, shape (n_q, n_k)

    Raises:
        ValueError: If shapes are incompatible.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError(f"Q, K, V must be 2D, got Q={Q.ndim}D, K={K.ndim}D, V={V.ndim}D")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(f"Q and K must have same key dimension: Q={Q.shape}, K={K.shape}")
    if K.shape[0] != V.shape[0]:
        raise ValueError(f"K and V must have same sequence length: K={K.shape}, V={V.shape}")

    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    output = weights @ V

    return output, weights


def attention_weights(Q: NDArray, K: NDArray, mask: NDArray | None = None) -> NDArray:
    """Compute attention weights without values (for visualization).

    Args:
        Q: Queries, shape (n_q, d_k).
        K: Keys, shape (n_k, d_k).
        mask: Optional boolean mask.

    Returns:
        Attention weights, shape (n_q, n_k).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if Q.ndim != 2 or K.ndim != 2:
        raise ValueError(f"Q and K must be 2D, got Q={Q.ndim}D, K={K.ndim}D")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(f"Q and K must have same key dimension: Q={Q.shape}, K={K.shape}")

    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scores = np.where(mask, scores, -1e9)

    return softmax(scores, axis=-1)
