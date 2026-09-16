"""
Gradient computation for scalar-valued multivariable functions.

Mathematics:

For f: ℝⁿ → ℝ, the gradient is:

    ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

The gradient points in the direction of steepest increase.
Its magnitude |∇f| is the rate of steepest increase.

Neural network connection:
- The gradient of the loss with respect to weights tells us
  how to update each weight to reduce the loss
- Gradient descent: w ← w - η * ∇L(w)
- Understanding gradients IS understanding how neural networks learn
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.calculus.partials import partial_derivative


def numerical_gradient(
    func: Callable[[NDArray], float],
    point: NDArray,
    h: float = 1e-5,
) -> NDArray:
    """Compute numerical gradient using central differences.

    ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

    Each partial derivative is computed via central difference:
    ∂f/∂xᵢ ≈ [f(x + h*eᵢ) - f(x - h*eᵢ)] / (2h)

    Args:
        func: Scalar function that takes an NDArray and returns a float.
        point: The point at which to evaluate the gradient.
        h: Step size for finite differences (must be > 0).

    Returns:
        Gradient vector as NDArray with same shape as point.

    Raises:
        ValueError: If h <= 0 or point is not 1-dimensional.
    """
    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f"Point must be 1-dimensional, got shape {point.shape}.")

    n = point.shape[0]
    grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        grad[i] = partial_derivative(func, point, i, h)

    return grad


def gradient_magnitude(gradient: NDArray) -> float:
    """Compute the magnitude (L2 norm) of a gradient vector.

    |∇f| = sqrt(sum((∂f/∂xᵢ)²))

    Neural network connection: Gradient magnitude indicates the
    overall sensitivity of the function to all inputs simultaneously.

    Args:
        gradient: Gradient vector.

    Returns:
        Magnitude as a non-negative scalar.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    return float(np.sqrt(np.sum(gradient**2)))


def gradient_direction(gradient: NDArray) -> NDArray:
    """Compute the unit vector in the direction of the gradient.

    û = ∇f / |∇f|

    Neural network connection: Gradient descent follows this direction
    (for maximization) or the negative (for minimization).

    Args:
        gradient: Gradient vector.

    Returns:
        Unit direction vector.

    Raises:
        ValueError: If gradient is zero (no direction defined).
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    mag = gradient_magnitude(gradient)
    if mag < 1e-15:
        raise ValueError(
            "Cannot compute direction of zero gradient. The function has a critical point here."
        )
    return gradient / mag
