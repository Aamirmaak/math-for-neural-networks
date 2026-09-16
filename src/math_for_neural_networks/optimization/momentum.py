"""
Momentum
========

Momentum accelerates gradient descent by accumulating directional information
from past gradients. It is the conceptual bridge between vanilla gradient
descent and adaptive methods like Adam.

Update rule (standard convention):
    v_t = beta * v_(t-1) + gradient(theta_t)
    theta_(t+1) = theta_t - eta * v_t

where:
    v = velocity (momentum buffer)
    beta = momentum coefficient (typically 0.9)
    eta = learning rate

WHY does momentum help?
- On consistent gradients: velocity accumulates, effective step size grows
  -> faster convergence through flat regions
- On noisy/oscillating gradients: positive and negative gradients cancel
  -> reduces oscillation perpendicular to the descent direction

Analogy:
Imagine a ball rolling down a hill. Without momentum, it stops immediately
when the slope flattens. With momentum, it keeps rolling, carrying energy
from steeper sections.

ML Connection:
Momentum is used in most practical neural network optimizers.
SGD with momentum is the baseline optimizer for training deep networks.
PyTorch: torch.optim.SGD(lr=0.01, momentum=0.9)
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


def momentum(
    objective: Callable[[NDArray], float],
    gradient: Callable[[NDArray], NDArray],
    initial_position: NDArray,
    learning_rate: float = 0.01,
    beta: float = 0.9,
    max_iterations: int = 1000,
    grad_tol: float = 1e-6,
    obj_tol: float = 1e-12,
    param_tol: float = 1e-8,
    record_history: bool = True,
) -> OptResult:
    """Run gradient descent with momentum.

    Update rule:
        v_t = beta * v_(t-1) + nabla f(theta_t)
        theta_(t+1) = theta_t - eta * v_t

    Args:
        objective: Function to minimize: f(x) -> float.
        gradient: Gradient function: f(x) -> nabla f(x).
        initial_position: Starting parameter values.
        learning_rate: Step size eta. Must be > 0.
        beta: Momentum coefficient. Must be in [0, 1). Typically 0.9.
        max_iterations: Maximum number of iterations.
        grad_tol: Convergence tolerance for gradient norm.
        obj_tol: Convergence tolerance for objective change.
        param_tol: Convergence tolerance for parameter change.
        record_history: If True, record full history.

    Returns:
        OptResult with final parameters, history, and diagnostics.

    Raises:
        ValueError: If learning_rate <= 0, beta not in [0, 1), or max_iterations <= 0.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    x = np.asarray(initial_position, dtype=np.float64).copy()
    v = np.zeros_like(x)  # velocity / momentum buffer

    result = OptResult(parameters=x.copy())

    prev_obj = float("inf")
    converged = False
    reason = ""

    for t in range(max_iterations):
        # Compute gradient
        grad = np.asarray(gradient(x), dtype=np.float64)

        if not has_finite_values(grad):
            reason = f"non-finite gradient at iteration {t}"
            break

        grad_norm = float(np.linalg.norm(grad))

        # Compute objective
        obj = float(objective(x))

        # Record history
        if record_history:
            result.objective_history.append(obj)
            result.gradient_norm_history.append(grad_norm)
            result.parameter_history.append(x.tolist())

        # Momentum update
        v = beta * v + grad
        step = learning_rate * v
        param_change = float(np.linalg.norm(step))

        # Parameter update
        x_new = x - step

        if not has_finite_values(x_new):
            reason = f"non-finite parameters at iteration {t + 1}"
            break

        # Check convergence (after computing step)
        obj_change = abs(obj - prev_obj) if prev_obj != float("inf") else float("inf")

        converged, reason = check_convergence(
            grad_norm, obj_change, param_change, grad_tol, obj_tol, param_tol
        )

        if converged:
            reason = f"converged at iteration {t + 1}: {reason}"
            break

        x = x_new
        prev_obj = obj

    result.parameters = x.copy()
    result.iterations = t + 1 if "t" in dir() else 0
    result.converged = converged
    result.termination_reason = reason if reason else f"max_iterations ({max_iterations}) reached"

    return result
