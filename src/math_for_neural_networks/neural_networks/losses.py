"""
Loss Functions
==============

Loss functions measure how far a model's predictions are from the targets.
They are the objective that optimization minimizes.

Mathematics:

    Mean Squared Error (MSE):
        L = (1/n) * sum((y_i - y_hat_i)^2)
        Use: regression

    Binary Cross-Entropy (BCE):
        L = -(1/n) * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
        Use: binary classification

    Categorical Cross-Entropy (CCE):
        L = -(1/n) * sum_i sum_k y_ik * log(q_ik)
        For one-hot targets: L = -log(q_c) where c is the correct class
        Use: multi-class classification

    Logits + Cross-Entropy (numerically stable):
        L = log(sum(exp(logits))) - sum(targets * logits)
        Avoids computing softmax then log separately

Neural network connection:
    Loss functions define what "good" means for the model.
    The gradient of the loss with respect to parameters drives learning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_squared_error(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute Mean Squared Error: (1/n) * sum((y_true - y_pred)^2).

    Properties:
        - Always non-negative
        - 0 iff y_true == y_pred
        - Sensitive to outliers (squares errors)
        - Differentiable everywhere

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MSE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    return float(np.mean((y_true - y_pred) ** 2))


def binary_cross_entropy(y_true: NDArray, y_pred: NDArray, eps: float = 1e-15) -> float:
    """Compute Binary Cross-Entropy: -(1/n) * sum(y*log(p) + (1-y)*log(1-p)).

    Clips y_pred to [eps, 1-eps] to avoid log(0).

    Properties:
        - Non-negative
        - 0 iff predictions match targets perfectly
        - Punishes confident wrong predictions heavily

    Args:
        y_true: Binary targets (0 or 1).
        y_pred: Predicted probabilities in (0, 1).
        eps: Small constant for numerical stability.

    Returns:
        Binary cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
    return float(np.mean(loss))


def categorical_cross_entropy(y_true: NDArray, y_pred: NDArray, eps: float = 1e-15) -> float:
    """Compute Categorical Cross-Entropy: -(1/n) * sum(y_true * log(y_pred)).

    Supports one-hot encoded targets and probability distribution targets.
    Clips y_pred to [eps, inf) to avoid log(0).

    Properties:
        - Non-negative
        - 0 iff predicted probabilities match targets
        - For one-hot: loss = -log(predicted probability of correct class)

    Args:
        y_true: True labels. Shape (n_classes,) for single sample,
                or (batch_size, n_classes) for batch.
                Can be one-hot encoded or probability distributions.
        y_pred: Predicted probabilities. Same shape as y_true.
        eps: Small constant for numerical stability.

    Returns:
        Categorical cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    y_pred_clipped = np.clip(y_pred, eps, None)
    if y_true.ndim == 1:
        loss = -float(np.sum(y_true * np.log(y_pred_clipped)))
    elif y_true.ndim == 2:
        batch_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        loss = float(np.mean(batch_losses))
    else:
        raise ValueError(f"y_true must be 1D or 2D, got {y_true.ndim}D")
    return loss


def cross_entropy_with_logits(logits: NDArray, targets: NDArray) -> float:
    """Compute cross-entropy loss directly from logits (numerically stable).

    Uses the identity:
        CE = log(sum(exp(logits))) - sum(targets * logits)

    This avoids computing softmax followed by log, which is less stable.

    Properties:
        - Mathematically equivalent to softmax + cross-entropy
        - More numerically stable for extreme logits
        - Gradient is simply: softmax(logits) - targets

    Args:
        logits: Raw model outputs, shape (n_classes,) or (batch, n_classes).
        targets: One-hot encoded targets, same shape as logits.

    Returns:
        Mean cross-entropy loss.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits={logits.shape}, targets={targets.shape}")

    if logits.ndim == 1:
        max_logit = float(np.max(logits))
        lse = max_logit + float(np.log(np.sum(np.exp(logits - max_logit))))
        return lse - float(np.sum(targets * logits))
    elif logits.ndim == 2:
        max_logits = np.max(logits, axis=1, keepdims=True)
        lse = max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
        losses = lse - np.sum(targets * logits, axis=1, keepdims=True)
        return float(np.mean(losses))
    else:
        raise ValueError(f"logits must be 1D or 2D, got {logits.ndim}D")
