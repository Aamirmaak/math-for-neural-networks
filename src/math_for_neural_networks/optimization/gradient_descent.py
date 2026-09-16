"""
Gradient Descent
================

The foundational optimization algorithm for neural network training.

Core update rule:
    theta_(t+1) = theta_t - eta * nabla L(theta_t)

where:
    theta = parameters being optimized
    eta = learning rate (step size)
    nabla L(theta) = gradient of the loss function

WHY does this work?
- The gradient points in the direction of steepest ascent
- Subtracting the gradient moves toward steepest descent
- Small enough steps converge to a local minimum

WHY does learning rate matter?
- Too small: slow convergence (many iterations wasted)
- Right size: fast convergence (efficient descent)
- Too large: oscillation or divergence (overshoots minimum)

ML Connection:
This is EXACTLY what happens when you call optimizer.step() in PyTorch.
The gradient is computed by backpropagation, then the parameters are updated
using this rule. Understanding gradient descent is understanding the
fundamental mechanism of neural network training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.optimization.diagnostics import (
    OptResult,
    check_convergence,
    has_finite_values,
)


def gradient_descent(
    objective: Callable[[NDArray], float],
    gradient: Callable[[NDArray], NDArray],
    initial_position: NDArray,
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    grad_tol: float = 1e-6,
    obj_tol: float = 1e-12,
    param_tol: float = 1e-8,
    record_history: bool = True,
) -> OptResult:
    """Run gradient descent optimization.

    Update rule:
        theta_(t+1) = theta_t - eta * nabla f(theta_t)

    Convergence criteria (any one stops optimization):
        1. ||nabla f(theta)|| < grad_tol
        2. |f(theta_t) - f(theta_{t-1})| < obj_tol
        3. ||theta_t - theta_{t-1}|| < param_tol

    Args:
        objective: Function to minimize: f(x) -> float.
        gradient: Gradient function: f(x) -> nabla f(x).
        initial_position: Starting parameter values.
        learning_rate: Step size eta. Must be > 0.
        max_iterations: Maximum number of iterations. Must be > 0.
        grad_tol: Convergence tolerance for gradient norm.
        obj_tol: Convergence tolerance for objective change.
        param_tol: Convergence tolerance for parameter change.
        record_history: If True, record full history (slower but diagnostic).

    Returns:
        OptResult with final parameters, history, and diagnostics.

    Raises:
        ValueError: If learning_rate <= 0 or max_iterations <= 0.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    x = np.asarray(initial_position, dtype=np.float64).copy()
    result = OptResult(parameters=x.copy())

    prev_obj = float("inf")
    converged = False
    reason = ""

    for t in range(max_iterations):
        # Compute gradient
        grad = np.asarray(gradient(x), dtype=np.float64)

        # Check for non-finite gradient
        if not has_finite_values(grad):
            reason = f"non-finite gradient at iteration {t}"
            break

        grad_norm = float(np.linalg.norm(grad))

        # Compute objective value
        obj = float(objective(x))

        # Record history
        if record_history:
            result.objective_history.append(obj)
            result.gradient_norm_history.append(grad_norm)
            result.parameter_history.append(x.tolist())

        # Compute update step
        step = learning_rate * grad
        param_change = float(np.linalg.norm(step))

        # Check convergence (before update, using current step)
        obj_change = abs(obj - prev_obj) if prev_obj != float("inf") else float("inf")

        converged, reason = check_convergence(
            grad_norm, obj_change, param_change, grad_tol, obj_tol, param_tol
        )

        if converged:
            reason = f"converged at iteration {t + 1}: {reason}"
            break

        # Update parameters
        x_new = x - step

        # Check for non-finite parameters
        if not has_finite_values(x_new):
            reason = f"non-finite parameters at iteration {t + 1}"
            break

        x = x_new
        prev_obj = obj

    result.parameters = x.copy()
    result.iterations = t + 1 if "t" in dir() else 0
    result.converged = converged
    result.termination_reason = reason if reason else f"max_iterations ({max_iterations}) reached"

    return result
