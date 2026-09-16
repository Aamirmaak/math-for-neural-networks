"""
Optimization Diagnostics
========================

Data structures for optimization results and convergence information.

An OptResult captures everything about an optimization run:
- Final parameters
- Objective history (value at each iteration)
- Gradient norm history
- Parameter history
- Iteration count
- Whether it converged
- Why it stopped (convergence reason)

This provides transparent diagnostics without hiding important behavior.

ML Connection:
In neural network training, these diagnostics tell you:
- Is the loss decreasing? (objective_history)
- Are gradients vanishing or exploding? (gradient_norm_history)
- Did training converge or hit max iterations?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class OptResult:
    """Result of an optimization run.

    Attributes:
        parameters: Final parameter values.
        objective_history: Objective function value at each iteration.
        gradient_norm_history: L2 norm of gradient at each iteration.
        parameter_history: Parameter values at each iteration.
        iterations: Number of iterations performed.
        converged: Whether optimization converged (vs hit max iterations).
        termination_reason: Human-readable reason for stopping.
    """

    parameters: NDArray
    objective_history: list[float] = field(default_factory=list)
    gradient_norm_history: list[float] = field(default_factory=list)
    parameter_history: list[list[float]] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    termination_reason: str = ""

    @property
    def final_objective(self) -> float:
        """Final objective function value."""
        if self.objective_history:
            return self.objective_history[-1]
        return float("nan")

    @property
    def final_gradient_norm(self) -> float:
        """Final gradient norm."""
        if self.gradient_norm_history:
            return self.gradient_norm_history[-1]
        return float("nan")

    @property
    def objective_improvement(self) -> float:
        """Total improvement in objective value."""
        if len(self.objective_history) >= 2:
            return self.objective_history[0] - self.objective_history[-1]
        return 0.0


def check_convergence(
    gradient_norm: float,
    objective_change: float,
    parameter_change: float,
    grad_tol: float,
    obj_tol: float,
    param_tol: float,
) -> tuple[bool, str]:
    """Check if any convergence criterion is met.

    Convergence criteria (any one is sufficient):
        1. Gradient norm < grad_tol
        2. Objective change < obj_tol
        3. Parameter change < param_tol

    Args:
        gradient_norm: Current L2 norm of gradient.
        objective_change: Absolute change in objective from previous iteration.
        parameter_change: L2 norm of parameter change from previous iteration.
        grad_tol: Tolerance for gradient norm criterion.
        obj_tol: Tolerance for objective change criterion.
        param_tol: Tolerance for parameter change criterion.

    Returns:
        Tuple of (converged: bool, reason: str).
    """
    if gradient_norm < grad_tol:
        return True, f"gradient_norm ({gradient_norm:.2e}) < tol ({grad_tol:.2e})"
    if objective_change < obj_tol and objective_change >= 0:
        return True, f"objective_change ({objective_change:.2e}) < tol ({obj_tol:.2e})"
    if parameter_change < param_tol:
        return True, f"parameter_change ({parameter_change:.2e}) < tol ({param_tol:.2e})"
    return False, ""


def has_finite_values(arr: NDArray) -> bool:
    """Check if all values in array are finite (not NaN or Inf).

    Args:
        arr: Array to check.

    Returns:
        True if all values are finite.
    """
    return bool(np.all(np.isfinite(arr)))
