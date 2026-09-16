"""
Experiment 5: Initialization Effect on Convergence
===================================================

Hypothesis: Starting far from the minimum requires more iterations
but still converges for convex functions. For non-convex functions,
starting point may affect which local minimum is found.

Method: Test different starting points on quadratic and Rosenbrock.

Result: For quadratic (convex), all starting points converge to the
same minimum. For Rosenbrock (non-convex), different starts may
converge to different points or not converge at all.

Conclusion: Initialization matters more for non-convex problems.
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import quadratic, rosenbrock


def run_experiment() -> None:
    """Run the initialization effect experiment."""
    print("=" * 60)
    print("  EXPERIMENT 5: Initialization Effect on Convergence")
    print("=" * 60)

    # Part 1: Quadratic (convex)
    print("\n  PART 1: Quadratic f(x,y) = x^2 + y^2 (convex)")
    print("  " + "-" * 50)

    def f_quad(params: np.ndarray) -> float:
        return quadratic(params)

    def grad_quad(params: np.ndarray) -> np.ndarray:
        return 2.0 * params

    starting_points = [
        ("Near (0.5, 0.5)", np.array([0.5, 0.5])),
        ("Medium (3, 3)", np.array([3.0, 3.0])),
        ("Far (10, 10)", np.array([10.0, 10.0])),
        ("Very far (50, 50)", np.array([50.0, 50.0])),
        ("Asymmetric (10, 1)", np.array([10.0, 1.0])),
    ]

    print(f"  Learning rate: 0.3, Max iterations: 100")
    print(f"\n  {'Start':>20} | {'Final (x,y)':>18} | {'f(final)':>12} | {'Iters':>6}")
    print("  " + "-" * 65)

    for label, start in starting_points:
        result = gradient_descent(
            f_quad,
            grad_quad,
            start,
            learning_rate=0.3,
            max_iterations=100,
            param_tol=1e-8,
        )
        print(
            f"  {label:>20} | "
            f"({result.parameters[0]:>7.4f}, {result.parameters[1]:>7.4f}) | "
            f"{result.final_objective:>12.8f} | "
            f"{result.iterations:>6}"
        )

    # Part 2: Rosenbrock (non-convex)
    print("\n\n  PART 2: Rosenbrock (non-convex)")
    print("  " + "-" * 50)

    def f_rosen(params: np.ndarray) -> float:
        return rosenbrock(params)

    def grad_rosen(params: np.ndarray) -> np.ndarray:
        x, y = params[0], params[1]
        dx = -2.0 * (1 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])

    starting_points_rosen = [
        ("(-1, 1)", np.array([-1.0, 1.0])),
        ("(0, 0)", np.array([0.0, 0.0])),
        ("(2, 3)", np.array([2.0, 3.0])),
        ("(-2, 4)", np.array([-2.0, 4.0])),
        ("(3, 2)", np.array([3.0, 2.0])),
    ]

    true_min = np.array([1.0, 1.0])
    print(f"  Learning rate: 0.001, Max iterations: 5000")
    print(
        f"\n  {'Start':>12} | {'Final (x,y)':>18} | {'f(final)':>12} | {'Dist':>8} | {'Iters':>6}"
    )
    print("  " + "-" * 65)

    for label, start in starting_points_rosen:
        result = gradient_descent(
            f_rosen,
            grad_rosen,
            start,
            learning_rate=0.001,
            max_iterations=5000,
            param_tol=1e-10,
        )
        dist = np.sqrt(np.sum((result.parameters - true_min) ** 2))
        print(
            f"  {label:>12} | "
            f"({result.parameters[0]:>7.4f}, {result.parameters[1]:>7.4f}) | "
            f"{result.final_objective:>12.8f} | "
            f"{dist:>8.4f} | "
            f"{result.iterations:>6}"
        )

    # Part 3: Quantify the effect
    print("\n\n  PART 3: Iterations vs Distance from Minimum")
    print("  " + "-" * 50)

    distances = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    print(f"\n  For f(x) = x^2, starting at x = d:")
    print(f"  {'Distance':>10} | {'Iterations':>10} | {'Ratio':>10}")
    print("  " + "-" * 35)

    for d in distances:
        result = gradient_descent(
            lambda x: float(np.sum(x**2)),
            lambda x: 2.0 * x,
            np.array([d]),
            learning_rate=0.3,
            max_iterations=200,
            param_tol=1e-8,
        )
        # Analytical: iterations ~ log(d/tol) / log(1/(1-2*lr))
        ratio = result.iterations / max(1, int(np.log(d / 1e-8) / np.log(1 / 0.4)))
        print(f"  {d:>10.1f} | {result.iterations:>10} | {ratio:>10.2f}")

    print("\n  OBSERVATIONS:")
    print("  - Quadratic (convex): all starting points converge to same minimum")
    print("  - Further starting points need more iterations (logarithmic scaling)")
    print("  - Rosenbrock (non-convex): starting point affects final result")
    print("  - Different starts on Rosenbrock converge to different distances")
    print("  - Initialization is more critical for non-convex optimization")
    print("  - In neural networks: random initialization affects training outcome")


if __name__ == "__main__":
    run_experiment()
