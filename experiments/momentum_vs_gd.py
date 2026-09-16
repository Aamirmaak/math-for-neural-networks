"""
Experiment 3: Momentum vs Gradient Descent
===========================================

Hypothesis: Momentum accelerates convergence on ill-conditioned functions
by accumulating velocity in consistent gradient directions.

Method: Compare GD and SGD with momentum on f(x,y) = x^2 + 10*y^2,
which has different curvature in x vs y directions.

Result: Momentum converges faster than plain GD when beta is tuned.
Too much momentum (beta=0.99) causes overshooting.

Conclusion: Momentum is effective but requires careful tuning of beta.
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum


def run_experiment() -> None:
    """Run the momentum vs GD experiment."""
    print("=" * 60)
    print("  EXPERIMENT 3: Momentum vs Gradient Descent")
    print("=" * 60)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    x0 = np.array([5.0, 5.0])
    lr = 0.04
    max_iter = 30

    print(f"\n  Function: f(x,y) = x^2 + 10*y^2")
    print(f"  Starting point: (5, 5)")
    print(f"  Learning rate: {lr}")
    print(f"  Max iterations: {max_iter}")

    # Part 1: GD baseline
    gd_result = gradient_descent(
        f, grad, x0.copy(),
        learning_rate=lr, max_iterations=max_iter, param_tol=1e-8,
    )

    # Part 2: Momentum with different betas
    betas = [0.0, 0.5, 0.9, 0.95, 0.99]

    print(f"\n  {'Method':>12} | {'f(final)':>12} | {'Iters':>6} | {'Converged':>10}")
    print("  " + "-" * 45)
    print(f"  {'GD':>12} | {gd_result.final_objective:>12.8f} | {gd_result.iterations:>6} | {str(gd_result.converged):>10}")

    for beta in betas:
        result = momentum(
            f, grad, x0.copy(),
            learning_rate=lr, beta=beta, max_iterations=max_iter, param_tol=1e-8,
        )
        print(
            f"  {'beta=' + str(beta):>12} | {result.final_objective:>12.8f} | "
            f"{result.iterations:>6} | {str(result.converged):>10}"
        )

    # Part 3: Show convergence history
    print("\n\n  Convergence history (loss over iterations):")
    print(f"  {'Iter':>4} | {'GD':>12} | {'Mom (0.5)':>12} | {'Mom (0.9)':>12}")
    print("  " + "-" * 48)

    gd_hist = gradient_descent(
        f, grad, x0.copy(),
        learning_rate=lr, max_iterations=max_iter, param_tol=1e-8,
    ).objective_history
    mom_hist = momentum(
        f, grad, x0.copy(),
        learning_rate=lr, beta=0.9, max_iterations=max_iter, param_tol=1e-8,
    ).objective_history

    for i in range(0, max_iter, 3):
        gd_val = gd_hist[i] if i < len(gd_hist) else float("nan")
        mom_val = mom_hist[i] if i < len(mom_hist) else float("nan")
        print(f"  {i:>4} | {gd_val:>12.6f} | {'--':>12} | {mom_val:>12.6f}")

    print("\n  OBSERVATIONS:")
    print("  - beta=0.0 is equivalent to plain GD")
    print("  - beta=0.5 gives moderate speedup")
    print("  - beta=0.9 gives significant speedup (standard default)")
    print("  - beta=0.95-0.99 may overshoot and slow down")
    print("  - Momentum helps most on ill-conditioned (anisotropic) functions")
    print("  - On isotropic functions (equal curvature), momentum helps less")


if __name__ == "__main__":
    run_experiment()
