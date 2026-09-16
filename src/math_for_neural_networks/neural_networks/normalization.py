"""
Normalization Mathematics
=========================

Normalization stabilizes neural network training by reducing
internal covariate shift (change in layer input distributions).

Layer Normalization:

    For a feature vector x in R^d:

    mu = mean(x)           scalar
    sigma^2 = mean((x - mu)^2)   scalar
    x_hat = (x - mu) / sqrt(sigma^2 + epsilon)
    y = gamma * x_hat + beta

Where:
    gamma = learnable scale parameter (initialized to 1)
    beta = learnable shift parameter (initialized to 0)
    epsilon = small constant for numerical stability (typically 1e-5)

Why Layer Norm (not Batch Norm)?
    - Works for variable-length sequences (NLP)
    - No dependence on batch statistics
    - Each sample is normalized independently
    - Used in Transformers, GPT, BERT

Neural network connection:
    Normalization is applied after linear transformation + activation.
    It stabilizes training and allows higher learning rates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def layer_norm(
    x: NDArray,
    gamma: NDArray | None = None,
    beta: NDArray | None = None,
    eps: float = 1e-5,
) -> NDArray:
    """Compute Layer Normalization.

    For a 1D input x in R^d:
        mu = mean(x)
        sigma^2 = mean((x - mu)^2)
        x_hat = (x - mu) / sqrt(sigma^2 + eps)
        y = gamma * x_hat + beta

    For a 2D input x in R^(batch, d):
        Normalization is applied independently to each sample.

    Args:
        x: Input vector or batch. Shape (d,) or (batch, d).
        gamma: Scale parameter. Shape (d,). Defaults to ones.
        beta: Shift parameter. Shape (d,). Defaults to zeros.
        eps: Small constant for numerical stability.

    Returns:
        Normalized output. Same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        mu = np.mean(x)
        var = np.mean((x - mu) ** 2)
        x_hat = (x - mu) / np.sqrt(var + eps)
    elif x.ndim == 2:
        mu = np.mean(x, axis=1, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.ndim}D")

    if gamma is None:
        gamma = np.ones(x.shape[-1], dtype=np.float64)
    if beta is None:
        beta = np.zeros(x.shape[-1], dtype=np.float64)

    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError(f"gamma and beta must be 1D, got gamma={gamma.ndim}D, beta={beta.ndim}D")
    if gamma.shape[0] != x.shape[-1] or beta.shape[0] != x.shape[-1]:
        raise ValueError(
            f"gamma/beta size mismatch: x has {x.shape[-1]} features, "
            f"gamma has {gamma.shape[0]}, beta has {beta.shape[0]}"
        )

    result: NDArray = gamma * x_hat + beta
    return result


def layer_norm_stats(x: NDArray, eps: float = 1e-5) -> dict[str, float | NDArray]:
    """Compute layer normalization statistics (for inspection).

    Returns:
        Dictionary with mean, variance, normalized values, and eps.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.ndim}D")

    mu = float(np.mean(x))
    var = float(np.mean((x - mu) ** 2))
    x_hat = (x - mu) / np.sqrt(var + eps)

    return {
        "mean": mu,
        "variance": var,
        "x_hat": x_hat,
        "eps": eps,
    }
