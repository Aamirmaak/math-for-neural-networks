"""
Chain rule for function composition.

Mathematics:

For y = f(g(x)):

    dy/dx = (df/dg) * (dg/dx)

For multivariable composition y = f(g₁(x), g₂(x), ...):

    dy/dx = Σᵢ (∂f/∂gᵢ) * (dgᵢ/dx)

Neural network connection:
- The chain rule IS backpropagation
- Each layer computes a local derivative
- Backward pass multiplies local derivatives together
- This is how gradients flow from loss to parameters

Example:
    g(x) = x²        →  dg/dx = 2x
    f(g) = sin(g)     →  df/dg = cos(g)
    y = sin(x²)       →  dy/dx = cos(x²) * 2x
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def chain_rule_scalar(
    outer_func: Callable[[float], float],
    outer_deriv: Callable[[float], float],
    inner_func: Callable[[float], float],
    inner_deriv: Callable[[float], float],
    x: float,
) -> float:
    """Apply chain rule for scalar composition y = f(g(x)).

    dy/dx = f'(g(x)) * g'(x)

    Args:
        outer_func: f — the outer function.
        outer_deriv: f' — derivative of outer function.
        inner_func: g — the inner function.
        inner_deriv: g' — derivative of inner function.
        x: Point at which to evaluate.

    Returns:
        The chain rule derivative dy/dx.
    """
    g_x = inner_func(x)
    return outer_deriv(g_x) * inner_deriv(x)


def chain_rule_multi(
    outer_derivs: NDArray,
    inner_derivs: NDArray,
) -> float:
    """Apply chain rule for multivariable composition.

    For y = f(g₁(x), g₂(x), ..., gₙ(x)):

        dy/dx = Σᵢ (∂f/∂gᵢ) * (dgᵢ/dx)

    This is the foundation of backpropagation through a layer.

    Neural network connection:
        If z = Wx + b and loss = L(z)
        Then ∂L/∂x = W^T * ∂L/∂z

    Args:
        outer_derivs: Vector of ∂f/∂gᵢ (partial derivatives of outer func).
        inner_derivs: Vector of dgᵢ/dx (derivatives of inner funcs).

    Returns:
        The combined chain rule derivative.
    """
    outer_derivs = np.asarray(outer_derivs, dtype=np.float64)
    inner_derivs = np.asarray(inner_derivs, dtype=np.float64)
    return float(np.sum(outer_derivs * inner_derivs))


def demonstrate_chain_rule() -> dict[str, float | str]:
    """Demonstrate chain rule with the example y = sin(x²).

    g(x) = x²       →  g'(x) = 2x
    f(g) = sin(g)    →  f'(g) = cos(g)
    y = sin(x²)      →  dy/dx = cos(x²) * 2x

    Returns:
        Dictionary with demonstration values at x = 2.
    """
    x = 2.0
    # Inner function
    g_x = x**2
    g_prime = 2 * x

    # Outer function
    f_g = np.sin(g_x)
    f_prime = np.cos(g_x)

    # Chain rule
    chain_result = f_prime * g_prime

    # Verify with direct computation
    from math_for_neural_networks.calculus.finite_differences import central_difference

    def composed(x_val: float) -> float:
        return float(np.sin(x_val**2))

    numerical = central_difference(composed, x)

    return {
        "x": x,
        "g(x)": g_x,
        "g'(x)": g_prime,
        "f(g)": f_g,
        "f'(g)": f_prime,
        "chain_rule": chain_result,
        "numerical": numerical,
        "error": abs(chain_result - numerical),
    }


def neural_network_chain_rule_demo() -> dict[str, float]:
    """Demonstrate chain rule as it appears in a single neuron.

    Model: z = w*x + b, loss = (z - target)^2

    Chain rule path:
        ∂loss/∂w = ∂loss/∂z * ∂z/∂w
        ∂loss/∂b = ∂loss/∂z * ∂z/∂b

    Local derivatives:
        ∂loss/∂z = 2(z - target)
        ∂z/∂w = x
        ∂z/∂b = 1

    Returns:
        Dictionary with all values for a concrete example.
    """
    # Concrete example
    w, b, x = 2.0, 1.0, 3.0
    target = 10.0

    # Forward pass
    z = w * x + b  # z = 2*3 + 1 = 7

    # Loss
    loss = (z - target) ** 2  # (7-10)^2 = 9

    # Local derivatives
    dloss_dz = 2 * (z - target)  # 2*(7-10) = -6
    dz_dw = x  # 3
    dz_db = 1.0

    # Chain rule
    dloss_dw = dloss_dz * dz_dw  # -6 * 3 = -18
    dloss_db = dloss_dz * dz_db  # -6 * 1 = -6

    return {
        "w": w,
        "b": b,
        "x": x,
        "target": target,
        "z": z,
        "loss": loss,
        "dloss_dz": dloss_dz,
        "dz_dw": dz_dw,
        "dz_db": dz_db,
        "dloss_dw": dloss_dw,
        "dloss_db": dloss_db,
    }
