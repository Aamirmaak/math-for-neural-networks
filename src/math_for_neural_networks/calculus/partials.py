"""
Partial derivatives for multivariable functions.

Mathematics:

For f(x₁, x₂, ..., xₙ), the partial derivative with respect to xᵢ is:

    ∂f/∂xᵢ = lim_{h→0} [f(x₁, ..., xᵢ+h, ..., xₙ) - f(x₁, ..., xᵢ-h, ..., xₙ)] / (2h)

All other variables are held constant.

Neural network connection:
- Each weight in a neural network has a partial derivative
- ∂loss/∂wᵢ measures how sensitive the loss is to weight wᵢ
- The gradient is the vector of all partial derivatives
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def partial_derivative(
    func: Callable[[NDArray], float],
    point: NDArray,
    variable_index: int,
    h: float = 1e-5,
) -> float:
    """Compute partial derivative using central difference.

    ∂f/∂xᵢ ≈ [f(x + h*eᵢ) - f(x - h*eᵢ)] / (2h)

    where eᵢ is the unit vector in the i-th direction.

    Args:
        func: Function that takes an NDArray and returns a float.
        point: The point at which to evaluate, as an NDArray.
        variable_index: Which variable to differentiate with respect to.
        h: Step size (must be > 0).

    Returns:
        Approximate partial derivative.

    Raises:
        ValueError: If h <= 0 or variable_index is out of bounds.
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}.")

    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f"Point must be 1-dimensional, got shape {point.shape}.")

    n = point.shape[0]
    if variable_index < 0 or variable_index >= n:
        raise ValueError(
            f"variable_index {variable_index} is out of bounds for {n}-dimensional point."
        )

    # Create perturbation vector: h in the variable_index direction
    e_i = np.zeros(n, dtype=np.float64)
    e_i[variable_index] = h

    return float((func(point + e_i) - func(point - e_i)) / (2 * h))


def partial_derivative_forward(
    func: Callable[[NDArray], float],
    point: NDArray,
    variable_index: int,
    h: float = 1e-5,
) -> float:
    """Compute partial derivative using forward difference.

    ∂f/∂xᵢ ≈ [f(x + h*eᵢ) - f(x)] / h

    Args:
        func: Function that takes an NDArray and returns a float.
        point: The point at which to evaluate.
        variable_index: Which variable to differentiate with respect to.
        h: Step size (must be > 0).

    Returns:
        Approximate partial derivative.
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}.")

    point = np.asarray(point, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError(f"Point must be 1-dimensional, got shape {point.shape}.")

    n = point.shape[0]
    if variable_index < 0 or variable_index >= n:
        raise ValueError(
            f"variable_index {variable_index} is out of bounds for {n}-dimensional point."
        )

    e_i = np.zeros(n, dtype=np.float64)
    e_i[variable_index] = h

    return float((func(point + e_i) - func(point)) / h)
