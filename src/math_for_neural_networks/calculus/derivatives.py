"""
Analytical derivatives for common functions used in neural networks.

Each function provides:
- The forward evaluation f(x)
- The analytical derivative f'(x)
- Documentation of the mathematical formula
- Neural network connection

Mathematics:
- d/dx x^n = n * x^(n-1)
- d/dx sin(x) = cos(x)
- d/dx cos(x) = -sin(x)
- d/dx exp(x) = exp(x)
- d/dx log(x) = 1/x
- d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
- d/dx tanh(x) = 1 - tanh(x)^2
- d/dx relu(x) = 1 if x > 0, 0 if x < 0, undefined at 0

Neural network connection:
- These are the building blocks of backpropagation
- Understanding these derivatives helps understand gradient flow
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Polynomial functions ---


def quadratic(x: float | NDArray) -> float | NDArray:
    """f(x) = x^2"""
    return x**2


def quadratic_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 2x

    Neural network: Similar to L2 penalty gradient
    """
    return 2 * x


def cubic(x: float | NDArray) -> float | NDArray:
    """f(x) = x^3"""
    return x**3


def cubic_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 3x^2"""
    return 3 * x**2


def polynomial(x: float | NDArray) -> float | NDArray:
    """f(x) = x^4 - 3x^2 + 2x - 1"""
    return x**4 - 3 * x**2 + 2 * x - 1


def polynomial_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 4x^3 - 6x + 2"""
    return 4 * x**3 - 6 * x + 2


# --- Trigonometric functions ---


def sin_func(x: float | NDArray) -> float | NDArray:
    """f(x) = sin(x)"""
    return np.sin(x)


def sin_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = cos(x)

    Neural network: Periodic activation functions, positional encodings
    """
    return np.cos(x)


def cos_func(x: float | NDArray) -> float | NDArray:
    """f(x) = cos(x)"""
    return np.cos(x)


def cos_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = -sin(x)"""
    return -np.sin(x)


# --- Exponential and logarithmic ---


def exp_func(x: float | NDArray) -> float | NDArray:
    """f(x) = e^x"""
    return np.exp(x)


def exp_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = e^x

    Neural network: Exponential appears in softmax, attention scaling
    """
    return np.exp(x)


def log_func(x: float | NDArray) -> float | NDArray:
    """f(x) = ln(x) for x > 0"""
    x_arr = np.asarray(x, dtype=np.float64)
    if np.any(x_arr <= 0):
        raise ValueError(f"log(x) requires x > 0, got {x}.")
    return np.log(x_arr)


def log_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 1/x for x > 0

    Neural network: Log probability in language modeling
    """
    return 1.0 / x


# --- Neural network activation functions ---


def sigmoid(x: float | NDArray) -> float | NDArray:
    """f(x) = 1 / (1 + e^(-x))

    Neural network: Output activation for binary classification,
    gates in LSTMs, attention gating.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = sigmoid(x) * (1 - sigmoid(x))

    The derivative of sigmoid depends on its own output.
    This creates the vanishing gradient problem for large |x|.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_func(x: float | NDArray) -> float | NDArray:
    """f(x) = tanh(x)

    Neural network: Hidden state activation in RNNs/LSTMs,
    centered output (mean ~0).
    """
    return np.tanh(x)


def tanh_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 1 - tanh(x)^2

    Similar to sigmoid but output centered at 0.
    Also suffers from vanishing gradients for large |x|.
    """
    t = np.tanh(x)
    return 1.0 - t**2


def relu(x: float | NDArray) -> float | NDArray:
    """f(x) = max(0, x)

    Neural network: Most common hidden layer activation.
    Sparse activation, no vanishing gradient for x > 0.
    """
    return np.maximum(0, x)


def relu_derivative(x: float | NDArray) -> float | NDArray:
    """f'(x) = 1 if x > 0, 0 if x < 0, undefined at x = 0

    In practice, we define it as 0 at x = 0.
    Dead neurons (always negative) have zero gradient.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return (x_arr > 0).astype(float)
