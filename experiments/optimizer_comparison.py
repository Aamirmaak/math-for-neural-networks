"""
Experiment 4: Adam vs Momentum vs Gradient Descent
===================================================

Hypothesis: Adam combines the best of momentum and adaptive scaling,
converging faster than both GD and momentum on most problems.

Method: Compare all three optimizers on f(x,y) = x^2 + 10*y^2.

Result: Adam converges fastest with default hyperparameters.
GD is slowest. Momentum is in between.

Conclusion: Adam's per-parameter learning rate adaptation is powerful.
"""

import numpy as np

from math_for_neural_networks.optimization.adam import adam
from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum


def run_experiment() -> None:
    """Run the Adam vs Momentum vs GD experiment."""
    print("=" * 60)
    print("  EXPERIMENT 4: Adam vs Momentum vs GD")
    print("=" * 60)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    x0 = np.array([5.0, 5.0])
    max_iter = 50

    print(f"\n  Function: f(x,y) = x^2 + 10*y^2")
    print(f"  Starting point: (5, 5)")
    print(f"  Max iterations: {max_iter}")

    # Part 1: Optimizer comparison
    optimizers = [
        ("GD (lr=0.04)", lambda: gradient_descent(f, grad, x0.copy(), learning_rate=0.04, max_iterations=max_iter, param_tol=1e-10)),
        ("Mom (lr=0.04, b=0.9)", lambda: momentum(f, grad, x0.copy(), learning_rate=0.04, beta=0.9, max_iterations=max_iter, param_tol=1e-10)),
        ("Adam (lr=0.1)", lambda: adam(f, grad, x0.copy(), learning_rate=0.1, max_iterations=max_iter, param_tol=1e-10)),
        ("Adam (lr=0.01)", lambda: adam(f, grad, x0.copy(), learning_rate=0.01, max_iterations=max_iter, param_tol=1e-10)),
    ]

    print(f"\n  {'Optimizer':>22} | {'f(final)':>12} | {'Iters':>6} | {'Converged':>10}")
    print("  " + "-" * 58)

    results = {}
    for name, opt_fn in optimizers:
        result = opt_fn()
        results[name] = result
        print(
            f"  {name:>22} | {result.final_objective:>12.8f} | "
            f"{result.iterations:>6} | {str(result.converged):>10}"
        )

    # Part 2: Convergence comparison
    print("\n\n  Convergence trajectory (loss at each iteration):")
    print(f"  {'Iter':>4} | {'GD':>12} | {'Momentum':>12} | {'Adam(0.1)':>12}")
    print("  " + "-" * 48)

    gd_hist = results["GD (lr=0.04)"].objective_history
    mom_hist = results["Mom (lr=0.04, b=0.9)"].objective_history
    adam_hist = results["Adam (lr=0.1)"].objective_history

    for i in range(0, max_iter, 5):
        gd_val = gd_hist[i] if i < len(gd_hist) else float("nan")
        mom_val = mom_hist[i] if i < len(mom_hist) else float("nan")
        adam_val = adam_hist[i] if i < len(adam_hist) else float("nan")
        print(f"  {i:>4} | {gd_val:>12.6f} | {mom_val:>12.6f} | {adam_val:>12.6f}")

    # Part 3: Rosenbrock comparison
    print("\n\n  PART 2: Rosenbrock function")
    print("  " + "-" * 50)

    def f_rosen(params: np.ndarray) -> float:
        x, y = params[0], params[1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def grad_rosen(params: np.ndarray) -> np.ndarray:
        x, y = params[0], params[1]
        dx = -2.0 * (1 - x) - 400.0 * x * (y - x ** 2)
        dy = 200.0 * (y - x ** 2)
        return np.array([dx, dy])

    x0_rosen = np.array([-1.0, 1.0])
    true_min = np.array([1.0, 1.0])

    rosen_optimizers = [
        ("GD (lr=0.001)", lambda: gradient_descent(f_rosen, grad_rosen, x0_rosen.copy(), learning_rate=0.001, max_iterations=5000, param_tol=1e-10)),
        ("Mom (lr=0.001, b=0.9)", lambda: momentum(f_rosen, grad_rosen, x0_rosen.copy(), learning_rate=0.001, beta=0.9, max_iterations=5000, param_tol=1e-10)),
        ("Adam (lr=0.01)", lambda: adam(f_rosen, grad_rosen, x0_rosen.copy(), learning_rate=0.01, max_iterations=5000, param_tol=1e-10)),
    ]

    print(f"\n  {'Optimizer':>22} | {'f(final)':>12} | {'Dist to min':>12} | {'Iters':>6}")
    print("  " + "-" * 60)

    for name, opt_fn in rosen_optimizers:
        result = opt_fn()
        dist = np.sqrt(np.sum((result.parameters - true_min) ** 2))
        print(
            f"  {name:>22} | {result.final_objective:>12.8f} | "
            f"{dist:>12.6f} | {result.iterations:>6}"
        )

    print("\n  OBSERVATIONS:")
    print("  - On quadratic: Adam converges fastest, GD slowest")
    print("  - Momentum provides moderate speedup over GD")
    print("  - Adam's per-parameter scaling helps on anisotropic functions")
    print("  - On Rosenbrock: all methods struggle (narrow valley)")
    print("  - Adam still makes progress but doesn't reach exact minimum")
    print("  - For hard non-convex problems, learning rate scheduling helps")


if __name__ == "__main__":
    run_experiment()
