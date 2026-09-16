"""
Adam (Adaptive Moment Estimation)
==================================

Adam combines the ideas of Momentum and RMSProp:
- First moment (mean) of gradients -> direction (like momentum)
- Second moment (variance) of gradients -> per-parameter step scaling

Update rule:
    m_t = beta1 * m_(t-1) + (1 - beta1) * g_t          (first moment)
    v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2         (second moment)
    m_hat_t = m_t / (1 - beta1^t)                        (bias correction)
    v_hat_t = v_t / (1 - beta2^t)                        (bias correction)
    theta_(t+1) = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)

WHY bias correction?
- At t=0: m_0 = 0, v_0 = 0 (initialization)
- Early estimates are biased toward zero
- Dividing by (1 - beta^t) corrects this bias
- As t grows: (1 - beta^t) -> 1, correction vanishes

WHY epsilon?
- Prevents division by zero when v_hat is very small
- Typical value: 1e-8

Default hyperparameters:
    beta1 = 0.9   (first moment decay)
    beta2 = 0.999 (second moment decay)
    epsilon = 1e-8
    eta = 0.001   (learning rate)

ML Connection:
Adam is the default optimizer for most neural network training.
It adapts the learning rate per-parameter based on gradient history.
PyTorch: torch.optim.Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)
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


def adam(
    objective: Callable[[NDArray], float],
    gradient: Callable[[NDArray], NDArray],
    initial_position: NDArray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iterations: int = 1000,
    grad_tol: float = 1e-6,
    obj_tol: float = 1e-12,
    param_tol: float = 1e-8,
    record_history: bool = True,
) -> OptResult:
    """Run Adam optimization.

    Update rule:
        m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
        v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
        m_hat_t = m_t / (1 - beta1^(t+1))
        v_hat_t = v_t / (1 - beta2^(t+1))
        theta_(t+1) = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)

    Note: We use (t+1) in bias correction because iteration counting starts
    at 0, but the first gradient is computed at the initial position.

    Args:
        objective: Function to minimize: f(x) -> float.
        gradient: Gradient function: f(x) -> nabla f(x).
        initial_position: Starting parameter values.
        learning_rate: Step size eta. Must be > 0.
        beta1: First moment decay. Must be in [0, 1).
        beta2: Second moment decay. Must be in [0, 1).
        epsilon: Small constant for numerical stability. Must be > 0.
        max_iterations: Maximum number of iterations.
        grad_tol: Convergence tolerance for gradient norm.
        obj_tol: Convergence tolerance for objective change.
        param_tol: Convergence tolerance for parameter change.
        record_history: If True, record full history.

    Returns:
        OptResult with final parameters, history, and diagnostics.

    Raises:
        ValueError: If hyperparameters are invalid.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    x = np.asarray(initial_position, dtype=np.float64).copy()
    m = np.zeros_like(x)  # first moment estimate
    v = np.zeros_like(x)  # second moment estimate

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

        # Adam update
        t_step = t + 1  # 1-indexed for bias correction
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)

        # Bias correction
        m_hat = m / (1.0 - beta1**t_step)
        v_hat = v / (1.0 - beta2**t_step)

        # Compute update step
        step = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        param_change = float(np.linalg.norm(step))

        # Parameter update
        x_new = x - step

        if not has_finite_values(x_new):
            reason = f"non-finite parameters at iteration {t + 1}"
            break

        # Check convergence (after update, using actual step)
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


def adam_step(
    x: NDArray,
    grad: NDArray,
    m: NDArray,
    v: NDArray,
    t: int,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[NDArray, NDArray, NDArray]:
    """Single Adam step (useful for testing individual updates).

    Args:
        x: Current parameters.
        grad: Current gradient.
        m: First moment estimate.
        v: Second moment estimate.
        t: Current timestep (1-indexed).
        learning_rate: Step size.
        beta1: First moment decay.
        beta2: Second moment decay.
        epsilon: Numerical stability constant.

    Returns:
        Tuple of (new_x, new_m, new_v).
    """
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad**2)

    m_hat = m_new / (1.0 - beta1**t)
    v_hat = v_new / (1.0 - beta2**t)

    x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x_new, m_new, v_new
