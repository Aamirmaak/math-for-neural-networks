"""
Numerical differentiation using finite differences.

Mathematics:

Forward difference:
    f'(x) ≈ [f(x+h) - f(x)] / h
    Error: O(h) — first-order approximation

Central difference:
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    Error: O(h²) — second-order approximation, generally more accurate

Step size h tradeoff:
- Too large: poor approximation (truncation error)
- Too small: floating-point cancellation (round-off error)
- Sweet spot: typically h ≈ 1e-5 to 1e-7 for float64

Neural network connection:
- Gradient checking in backpropagation uses finite differences
- Validates analytical gradients against numerical approximations
- Central differences are preferred for gradient checking
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def forward_difference(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """Compute derivative using forward difference.

    f'(x) ≈ [f(x+h) - f(x)] / h

    Error: O(h) — first-order accurate.

    Args:
        func: Function to differentiate.
        x: Point at which to compute derivative.
        h: Step size (must be > 0).

    Returns:
        Approximate derivative.

    Raises:
        ValueError: If h <= 0.
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    return (func(x + h) - func(x)) / h


def central_difference(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """Compute derivative using central difference.

    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    Error: O(h²) — second-order accurate, generally preferred.

    Neural network connection: Central differences are the standard
    method for gradient checking in backpropagation implementations.

    Args:
        func: Function to differentiate.
        x: Point at which to compute derivative.
        h: Step size (must be > 0).

    Returns:
        Approximate derivative.

    Raises:
        ValueError: If h <= 0.
    """
    if h <= 0:
        raise ValueError(f"Step size h must be positive, got {h}.")
    return (func(x + h) - func(x - h)) / (2 * h)


def numerical_derivative(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    method: str = "central",
) -> float:
    """Compute numerical derivative using specified method.

    Args:
        func: Function to differentiate.
        x: Point at which to compute derivative.
        h: Step size (must be > 0).
        method: "forward" or "central" (default: "central").

    Returns:
        Approximate derivative.

    Raises:
        ValueError: If h <= 0 or method is unknown.
    """
    if method == "forward":
        return forward_difference(func, x, h)
    elif method == "central":
        return central_difference(func, x, h)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'forward' or 'central'.")


def step_size_analysis(
    func: Callable[[float], float],
    x: float,
    true_derivative: float,
    h_values: NDArray | None = None,
) -> dict[str, NDArray]:
    """Analyze how step size affects numerical derivative accuracy.

    This experiment demonstrates the tradeoff between:
    - Truncation error (large h) — the approximation is poor
    - Round-off error (small h) — floating-point cancellation dominates

    Neural network connection: Understanding this tradeoff is essential
    for implementing reliable gradient checking.

    Args:
        func: Function to differentiate.
        x: Point at which to evaluate.
        true_derivative: Known analytical derivative for comparison.
        h_values: Array of step sizes to test. Default: logspace from 1e-1 to 1e-12.

    Returns:
        Dictionary with:
        - h_values: the step sizes tested
        - forward_errors: absolute errors for forward difference
        - central_errors: absolute errors for central difference
    """
    if h_values is None:
        h_values = np.logspace(-1, -12, 50)

    forward_errors = np.zeros(len(h_values))
    central_errors = np.zeros(len(h_values))

    for i, h in enumerate(h_values):
        fd = forward_difference(func, x, h)
        cd = central_difference(func, x, h)
        forward_errors[i] = abs(fd - true_derivative)
        central_errors[i] = abs(cd - true_derivative)

    return {
        "h_values": h_values,
        "forward_errors": forward_errors,
        "central_errors": central_errors,
    }
