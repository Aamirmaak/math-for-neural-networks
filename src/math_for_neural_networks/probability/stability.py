"""
Numerical Stability Utilities
==============================

Stable implementations for operations that involve exp, log, and softmax.

Key stability techniques:
1. Log-Sum-Exp: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
   Prevents overflow when computing log of sum of exponentials.

2. Softmax: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
   Subtracting max prevents overflow. Result is identical to naive version.

3. Stable log softmax: log(softmax(x_i)) = x_i - log(Σ exp(x_j))
   = x_i - max(x) - log(Σ exp(x_j - max(x)))

4. Binary cross-entropy with logits: BCE with sigmoid built in for stability.

Why this matters for ML:
- Neural network outputs (logits) can be very large
- Naive exp(large_number) overflows to inf
- Naive softmax produces NaN for large inputs
- Language models work with log-probabilities throughout
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def log_sum_exp(x: NDArray) -> float:
    """Compute log(Σ exp(x_i)) in a numerically stable way.

    Uses the identity:
        log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))

    This prevents overflow from exp(large_number).

    Args:
        x: Input vector.

    Returns:
        log(Σ exp(x_i)).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.ndim}D")
    max_x = float(np.max(x))
    return max_x + float(np.log(np.sum(np.exp(x - max_x))))


def softmax(x: NDArray) -> NDArray:
    """Compute softmax(x_i) = exp(x_i) / Σ exp(x_j) in a numerically stable way.

    Subtracts max(x) before exponentiation to prevent overflow.
    The result is mathematically identical to the naive formula.

    Args:
        x: Input logits (1D array).

    Returns:
        Probability distribution (sums to 1).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.ndim}D")
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    result: NDArray = exp_x / np.sum(exp_x)
    return result


def log_softmax(x: NDArray) -> NDArray:
    """Compute log(softmax(x_i)) in a numerically stable way.

    Uses the identity:
        log(softmax(x_i)) = x_i - log(Σ exp(x_j))

    Computed as:
        x_i - max(x) - log(Σ exp(x_j - max(x)))

    Args:
        x: Input logits (1D array).

    Returns:
        Log-probability distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.ndim}D")
    max_x = float(np.max(x))
    lse = max_x + float(np.log(np.sum(np.exp(x - max_x))))
    return x - lse


def sigmoid(x: NDArray | float) -> NDArray | float:
    """Compute sigmoid(x) = 1 / (1 + exp(-x)) stably.

    For large positive x: sigmoid ≈ 1
    For large negative x: sigmoid ≈ 0

    Numerically stable implementation avoids exp overflow.

    Args:
        x: Input value(s).

    Returns:
        Sigmoid output in (0, 1).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    # For numerical stability: use different formulas for positive/negative
    result = np.where(
        x_arr >= 0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )
    return float(result) if result.ndim == 0 else result


def binary_cross_entropy_with_logits(logits: NDArray, targets: NDArray) -> float:
    """Compute binary cross-entropy with logits in a numerically stable way.

    Uses the identity:
        BCE = -[t × log(σ(x)) + (1-t) × log(1-σ(x))]
            = max(x, 0) - x×t + log(1 + exp(-|x|))

    This avoids computing log(σ(x)) directly, which is numerically unstable.

    Args:
        logits: Raw model outputs (before sigmoid).
        targets: Binary targets (0 or 1).

    Returns:
        Mean binary cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if logits.ndim != 1 or targets.ndim != 1:
        raise ValueError(f"logits and targets must be 1D, got {logits.ndim}D and {targets.ndim}D")
    if len(logits) != len(targets):
        raise ValueError("logits and targets must have same length")
    if np.any(targets < 0) or np.any(targets > 1):
        raise ValueError("targets must be in [0, 1]")

    # Numerically stable: max(x, 0) - x*t + log(1 + exp(-|x|))
    loss = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def cross_entropy_with_logits(logits: NDArray, targets: NDArray) -> float:
    """Compute cross-entropy loss with logits (softmax + cross-entropy).

    Uses the identity:
        CE = -Σ t_i × log(softmax(x_i))
           = -Σ t_i × (x_i - log(Σ exp(x_j)))
           = log(Σ exp(x_j)) - Σ t_i × x_i

    Where log(Σ exp(x_j)) is computed using log-sum-exp.

    Args:
        logits: Raw model outputs (before softmax), shape (n_classes,) or (batch, n_classes).
        targets: One-hot encoded targets, same shape as logits.

    Returns:
        Mean cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if logits.ndim == 1:
        # Single sample
        lse = log_sum_exp(logits)
        ce = lse - float(np.sum(targets * logits))
        return ce
    elif logits.ndim == 2:
        # Batch
        batch_size = logits.shape[0]
        losses = np.zeros(batch_size)
        for i in range(batch_size):
            lse = log_sum_exp(logits[i])
            losses[i] = lse - float(np.sum(targets[i] * logits[i]))
        return float(np.mean(losses))
    else:
        raise ValueError(f"logits must be 1D or 2D, got {logits.ndim}D")
