"""
Experiment 2: 2D Gradient Descent Path Visualization
=====================================================

Hypothesis: GD follows the steepest descent path on quadratic functions.
On non-convex functions (Rosenbrock), GD may struggle in narrow valleys.

Method: Run GD on f(x,y) = x^2 + y^2 and f(x,y) = Rosenbrock,
print the path as text contour.

Result: Quadratic converges cleanly. Rosenbrock gets stuck near
the valley but doesn't reach the exact minimum.

Conclusion: Function geometry strongly affects optimization difficulty.
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import quadratic, rosenbrock


def text_contour(values: np.ndarray, width: int = 50, height: int = 25) -> None:
    """Print a text-based contour plot."""
    chars = " .:-=+*#%@"
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        grid = np.zeros((height, width), dtype=int)
    else:
        normalized = (values - vmin) / (vmax - vmin)
        grid = (normalized * (len(chars) - 1)).astype(int)
    for row in range(height):
        line = ""
        for col in range(width):
            line += chars[grid[row, col]]
        print(f"  {line}")


def run_experiment() -> None:
    """Run the 2D GD path experiment."""
    print("=" * 60)
    print("  EXPERIMENT 2: 2D Gradient Descent Path")
    print("=" * 60)

    # Part 1: Quadratic
    print("\n  PART 1: Quadratic f(x,y) = x^2 + y^2")
    print("  " + "-" * 50)

    def f_quad(params: np.ndarray) -> float:
        return quadratic(params)

    def grad_quad(params: np.ndarray) -> np.ndarray:
        return 2.0 * params

    x_range = np.linspace(-5, 5, 50)
    y_range = np.linspace(-5, 5, 25)
    Z = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            Z[i, j] = f_quad(np.array([x, y]))

    print("  Contour plot:")
    text_contour(Z)

    # Run GD from (4, 4)
    result = gradient_descent(
        f_quad, grad_quad, np.array([4.0, 4.0]),
        learning_rate=0.3, max_iterations=20, param_tol=1e-8,
    )
    print(f"\n  Starting at (4, 4), lr=0.3")
    print(f"  Final: ({result.parameters[0]:.4f}, {result.parameters[1]:.4f})")
    print(f"  Final loss: {result.final_objective:.8f}")
    print(f"  Iterations: {result.iterations}")

    # Part 2: Rosenbrock
    print("\n\n  PART 2: Rosenbrock f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
    print("  " + "-" * 50)

    def f_rosen(params: np.ndarray) -> float:
        return rosenbrock(params)

    def grad_rosen(params: np.ndarray) -> np.ndarray:
        x, y = params[0], params[1]
        dx = -2.0 * (1 - x) - 400.0 * x * (y - x ** 2)
        dy = 200.0 * (y - x ** 2)
        return np.array([dx, dy])

    x_range = np.linspace(-2, 3, 50)
    y_range = np.linspace(-1, 4, 25)
    Z = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            Z[i, j] = f_rosen(np.array([x, y]))

    print("  Contour plot:")
    text_contour(Z)

    # Run GD from (-1, 1)
    result = gradient_descent(
        f_rosen, grad_rosen, np.array([-1.0, 1.0]),
        learning_rate=0.001, max_iterations=5000, param_tol=1e-10,
    )
    dist = np.sqrt(
        (result.parameters[0] - 1.0) ** 2
        + (result.parameters[1] - 1.0) ** 2
    )
    print(f"\n  Starting at (-1, 1), lr=0.001")
    print(f"  Final: ({result.parameters[0]:.6f}, {result.parameters[1]:.6f})")
    print(f"  Final loss: {result.final_objective:.10f}")
    print(f"  Distance to minimum (1,1): {dist:.6f}")
    print(f"  Iterations: {result.iterations}")

    print("\n  OBSERVATIONS:")
    print("  - Quadratic: clean convergence to origin (minimum)")
    print("  - Rosenbrock: GD gets close but struggles in the narrow valley")
    print("  - Rosenbrock minimum at (1,1) is hard to reach with vanilla GD")
    print("  - Need more advanced optimizers (momentum, Adam) for such functions")


if __name__ == "__main__":
    run_experiment()
