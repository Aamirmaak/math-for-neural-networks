"""
Activation Functions
====================

Activation functions introduce non-linearity into neural networks.
Without them, a stack of linear layers is still just a linear transformation.

Mathematics:

    Sigmoid:  sigma(x) = 1 / (1 + exp(-x))
              Range: (0, 1)
              Derivative: sigma(x) * (1 - sigma(x))
              Use: binary classification output, gates

    Tanh:     tanh(x)
              Range: (-1, 1)
              Derivative: 1 - tanh(x)^2
              Use: RNN hidden states, zero-centered output

    ReLU:     max(0, x)
              Range: [0, inf)
              Derivative: 1 if x > 0, 0 if x < 0
              Use: most common hidden layer activation

    GELU:     x * Phi(x) where Phi is the standard normal CDF
              Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
              Use: Transformers, BERT, GPT

Neural network connection:
    Activation functions are applied after each linear transformation.
    They determine whether a neuron "fires" and how strongly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sigmoid(x: NDArray | float) -> NDArray | float:
    """Compute sigmoid(x) = 1 / (1 + exp(-x)).

    Numerically stable: uses different formulas for positive/negative x
    to avoid exp overflow.

    Properties:
        - Range: (0, 1)
        - sigma(0) = 0.5
        - sigma(-x) = 1 - sigma(x)
        - Saturates for |x| > 5

    Args:
        x: Input value(s).

    Returns:
        Sigmoid output in (0, 1).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.where(
        x_arr >= 0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )
    return float(result) if result.ndim == 0 else result


def sigmoid_derivative(x: NDArray | float) -> NDArray | float:
    """Compute sigmoid derivative: sigma(x) * (1 - sigma(x)).

    The derivative depends on the sigmoid output itself.
    Maximum at x=0 where sigma(0)=0.5, so derivative max = 0.25.
    Approaches 0 for large |x| (vanishing gradient).

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: NDArray | float) -> NDArray | float:
    """Compute tanh(x).

    Properties:
        - Range: (-1, 1)
        - tanh(0) = 0
        - tanh(-x) = -tanh(x) (odd symmetry)
        - Zero-centered output (unlike sigmoid)

    Args:
        x: Input value(s).

    Returns:
        Tanh output in (-1, 1).
    """
    return np.tanh(x)


def tanh_derivative(x: NDArray | float) -> NDArray | float:
    """Compute tanh derivative: 1 - tanh(x)^2.

    Maximum at x=0 where tanh(0)=0, so derivative max = 1.
    Approaches 0 for large |x| (vanishing gradient).

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).
    """
    t = np.tanh(x)
    return 1.0 - t**2


def relu(x: NDArray | float) -> NDArray | float:
    """Compute ReLU(x) = max(0, x).

    Properties:
        - Range: [0, inf)
        - Identity for positive values
        - Outputs 0 for negative values (sparsity)
        - Not differentiable at x=0 (we define derivative as 0 there)

    Args:
        x: Input value(s).

    Returns:
        ReLU output.
    """
    return np.maximum(0, x)


def relu_derivative(x: NDArray | float) -> NDArray | float:
    """Compute ReLU derivative.

    Derivative:
        1 if x > 0
        0 if x < 0
        0 at x = 0 (convention; mathematically undefined)

    "Dying ReLU" problem: neurons that output 0 for all inputs
    have zero gradient and never recover.

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return (x_arr > 0).astype(float)


def gelu(x: NDArray | float) -> NDArray | float:
    """Compute GELU(x) = x * Phi(x) where Phi is the standard normal CDF.

    Uses the tanh approximation:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Properties:
        - Smooth approximation of ReLU
        - Non-zero for negative values (unlike ReLU)
        - Used in Transformers (BERT, GPT)

    Args:
        x: Input value(s).

    Returns:
        GELU output.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    result: NDArray = (
        0.5 * x_arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_arr + 0.044715 * x_arr**3)))
    )
    return result


def gelu_derivative(x: NDArray | float) -> NDArray | float:
    """Compute GELU derivative using the tanh approximation.

    Args:
        x: Input value(s).

    Returns:
        Derivative value(s).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    sqrt_2_pi = np.sqrt(2.0 / np.pi)
    tanh_arg = sqrt_2_pi * (x_arr + 0.044715 * x_arr**3)
    sech2 = 1.0 - np.tanh(tanh_arg) ** 2
    d_tanh = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x_arr**2)
    result: NDArray = 0.5 * (1.0 + np.tanh(tanh_arg)) + 0.5 * x_arr * sech2 * d_tanh
    return result
