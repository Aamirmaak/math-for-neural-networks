"""
Backpropagation Utilities
=========================

This module provides:
1. Gradient checking (analytical vs numerical)
2. Matrix backpropagation for affine layers
3. Backpropagation through activation functions
4. Backpropagation through loss functions
5. Manual backward pass implementations for neural network layers

Mathematical foundation:
    Backpropagation computes gradients of a loss function L with respect
    to each parameter by applying the chain rule in reverse order through
    the computational graph.

    For L = f(g(h(x))):
        dL/dx = dL/dh * dh/dg * dg/dx

    In matrix form for Z = XW^T + b:
        dL/dX = dL/dZ @ W
        dL/dW = dL/dZ^T @ X  (transposed to match W shape)
        dL/db = sum(dL/dZ, axis=0)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def gradient_check(
    func: Callable[[NDArray], float],
    grad_func: Callable[[NDArray], NDArray],
    x: NDArray,
    h: float = 1e-5,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> dict[str, object]:
    """Check analytical gradients against numerical (finite-difference) gradients.

    Uses central differences: df/dx ≈ (f(x+h) - f(x-h)) / (2h)

    Args:
        func: Function to differentiate. Takes NDArray input, returns float.
        grad_func: Gradient function. Takes NDArray input, returns NDArray gradient.
        x: Point at which to check gradients.
        h: Finite-difference step size.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        Dictionary with:
            - x: the input point
            - analytical: analytical gradient
            - numerical: numerical gradient
            - abs_error: absolute error per element
            - rel_error: relative error per element
            - passed: bool, True if all gradients pass tolerance check
            - max_abs_error: maximum absolute error
            - max_rel_error: maximum relative error
    """
    x = np.asarray(x, dtype=np.float64)
    analytical = np.asarray(grad_func(x), dtype=np.float64)
    numerical = np.zeros_like(x, dtype=np.float64)

    flat_x = x.flatten()
    flat_num = numerical.flatten()

    for i in range(flat_x.size):
        x_plus = flat_x.copy()
        x_minus = flat_x.copy()
        x_plus[i] += h
        x_minus[i] -= h

        x_plus_reshaped = x_plus.reshape(x.shape)
        x_minus_reshaped = x_minus.reshape(x.shape)

        f_plus = float(func(x_plus_reshaped))
        f_minus = float(func(x_minus_reshaped))
        flat_num[i] = (f_plus - f_minus) / (2.0 * h)

    numerical = flat_num.reshape(x.shape)

    abs_error = np.abs(analytical - numerical)

    denom = np.maximum(np.abs(analytical), np.abs(numerical))
    rel_error = np.where(denom > 0, abs_error / denom, 0.0)

    max_abs = float(np.max(abs_error))
    max_rel = float(np.max(rel_error))

    passed = bool(np.all(abs_error <= atol + rtol * np.abs(numerical)))

    return {
        "x": x,
        "analytical": analytical,
        "numerical": numerical,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "passed": passed,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def gradient_check_scalar(
    func: Callable[[float], float],
    grad_func: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> dict[str, object]:
    """Check analytical gradient against numerical gradient for a scalar function.

    Args:
        func: Scalar function: float -> float.
        grad_func: Gradient function: float -> float.
        x: Point at which to check.
        h: Finite-difference step size.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Dictionary with gradient check results.
    """
    analytical = grad_func(x)
    numerical = (func(x + h) - func(x - h)) / (2.0 * h)

    abs_error = abs(analytical - numerical)
    denom = max(abs(analytical), abs(numerical))
    rel_error = abs_error / denom if denom > 0 else 0.0

    passed = abs_error <= atol + rtol * abs(numerical)

    return {
        "x": x,
        "analytical": analytical,
        "numerical": numerical,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "passed": passed,
        "max_abs_error": abs_error,
        "max_rel_error": rel_error,
    }


def affine_backward(
    x: NDArray, W: NDArray, b: NDArray, dout: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute gradients for affine transformation Z = XW^T + b.

    Given upstream gradient dL/dZ, compute:
        dL/dX = dL/dZ @ W
        dL/db = sum(dL/dZ, axis=0)
        dL/dW = dL/dZ^T @ X  (then transpose to match W shape)

    Shape conventions (batch-first):
        x: (batch_size, input_features) or (input_features,)
        W: (output_features, input_features)
        b: (output_features,)
        dout: same shape as output = (batch_size, output_features) or (output_features,)

    Returns:
        Tuple of (dx, dW, db) with shapes matching x, W, b respectively.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)

    if x.ndim == 1:
        # Single sample: x is (input_features,)
        # Z = W @ x + b, shape (output_features,)
        # dL/dX = W^T @ dL/dZ
        dx = W.T @ dout
        # dL/dW = dL/dZ outer x => (output_features, input_features)
        dW = np.outer(dout, x)
        # dL/db = dL/dZ
        db = dout.copy()
    elif x.ndim == 2:
        # Batch: x is (batch_size, input_features)
        # Z = x @ W^T + b, shape (batch_size, output_features)
        # dL/dX = dL/dZ @ W
        dx = dout @ W
        # dL/dW = dL/dZ^T @ x => (output_features, input_features)
        dW = dout.T @ x
        # dL/db = sum over batch
        db = np.sum(dout, axis=0)
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.ndim}D")

    return dx, dW, db


def sigmoid_backward(x: NDArray, dout: NDArray) -> NDArray:
    """Compute gradient through sigmoid activation.

    For a = sigmoid(x):
        dL/dx = dL/da * da/dx = dL/da * a * (1 - a)

    Args:
        x: Pre-activation input (same shape as dout).
        dout: Upstream gradient dL/da.

    Returns:
        Gradient dL/dx with same shape as x.
    """
    a = 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    return np.asarray(dout, dtype=np.float64) * a * (1.0 - a)


def relu_backward(x: NDArray, dout: NDArray) -> NDArray:
    """Compute gradient through ReLU activation.

    For a = max(0, x):
        dL/dx = dL/da * da/dx = dL/da * (1 if x > 0 else 0)

    Args:
        x: Pre-activation input (same shape as dout).
        dout: Upstream gradient dL/da.

    Returns:
        Gradient dL/dx with same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return np.asarray(dout, dtype=np.float64) * (x_arr > 0).astype(np.float64)


def tanh_backward(x: NDArray, dout: NDArray) -> NDArray:
    """Compute gradient through tanh activation.

    For a = tanh(x):
        dL/dx = dL/da * da/dx = dL/da * (1 - a^2)

    Args:
        x: Pre-activation input (same shape as dout).
        dout: Upstream gradient dL/da.

    Returns:
        Gradient dL/dx with same shape as x.
    """
    a = np.tanh(np.asarray(x, dtype=np.float64))
    return np.asarray(dout, dtype=np.float64) * (1.0 - a**2)


def mse_backward(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Compute gradient of MSE loss with respect to predictions.

    For L = (1/n) * sum((y_true - y_pred)^2):
        dL/d(y_pred) = (2/n) * (y_pred - y_true)

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Gradient dL/d(y_pred) with same shape as y_pred.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.size
    result: NDArray = (2.0 / n) * (y_pred - y_true)
    return result


def softmax_backward(logits: NDArray, targets: NDArray) -> NDArray:
    """Compute gradient of cross-entropy loss with respect to logits.

    For L = -sum(target * log(softmax(logits))):
        dL/d(logits) = softmax(logits) - targets

    This is a key identity in neural networks.

    Args:
        logits: Raw model outputs (before softmax).
        targets: One-hot encoded targets.

    Returns:
        Gradient dL/d(logits) with same shape as logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # Numerically stable softmax
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    result: NDArray = probs - targets
    return result


def sigmoid_bce_backward(logits: NDArray, targets: NDArray) -> NDArray:
    """Compute gradient of sigmoid + BCE with respect to logits.

    For L = -(1/n) * sum(y * log(sigma(z)) + (1-y) * log(1-sigma(z))):
        dL/dz = (1/n) * (sigma(z) - y)

    This is the clean gradient identity for sigmoid + BCE.
    The 1/n factor comes from the mean over the batch.

    Args:
        logits: Raw model outputs (before sigmoid).
        targets: Binary targets (0 or 1).

    Returns:
        Gradient dL/d(logits) with same shape as logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    n = targets.size
    result: NDArray = (probs - targets) / n
    return result
