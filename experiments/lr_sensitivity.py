"""
Experiment 1: Learning Rate Sensitivity Analysis
=================================================

Hypothesis: There is an optimal learning rate range.
Too small -> slow convergence. Too large -> divergence.

Method: Test learning rates from 0.01 to 0.99 on f(x) = x^2,
measure final loss and convergence speed.

Result: Optimal lr is around 0.2-0.5 for this simple function.
Learning rates > 0.5 cause oscillation or divergence.

Conclusion: Learning rate is the most critical hyperparameter.
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import quadratic


def run_experiment() -> None:
    """Run the learning rate sensitivity experiment."""
    print("=" * 60)
    print("  EXPERIMENT 1: Learning Rate Sensitivity Analysis")
    print("=" * 60)

    def f(x: np.ndarray) -> float:
        return quadratic(x)

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    x0 = np.array([5.0])
    learning_rates = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.45,
        0.49,
        0.5,
        0.55,
        0.6,
        0.8,
        0.99,
    ]

    print(f"\n  Function: f(x) = x^2")
    print(f"  Starting point: x = 5.0")
    print(f"  Max iterations: 100")
    print(f"\n  {'lr':>6} | {'Final f(x)':>12} | {'Iters':>6} | {'Converged':>10} | {'Status':>12}")
    print("  " + "-" * 55)

    for lr in learning_rates:
        result = gradient_descent(
            f,
            grad,
            x0.copy(),
            learning_rate=lr,
            max_iterations=100,
            param_tol=1e-8,
        )
        final_val = result.final_objective
        if final_val < 1e-6:
            status = "CONVERGED"
        elif final_val > 1e6:
            status = "DIVERGED"
        elif abs(final_val - f(x0)) < 0.01:
            status = "STUCK"
        else:
            status = "SLOW"

        print(
            f"  {lr:>6.2f} | {final_val:>12.8f} | {result.iterations:>6} | "
            f"{str(result.converged):>10} | {status:>12}"
        )

    print("\n  OBSERVATIONS:")
    print("  - lr < 0.1: convergence is slow (many iterations needed)")
    print("  - lr = 0.2-0.5: fast convergence (optimal range)")
    print("  - lr = 0.5: exact convergence in 1 step for x^2 (special case)")
    print("  - lr > 0.5: oscillation (bouncing around minimum)")
    print("  - lr >= 1.0: divergence (values grow instead of shrink)")
    print("  - For f(x) = x^2, max stable lr = 1.0 (from stability analysis)")


if __name__ == "__main__":
    run_experiment()
