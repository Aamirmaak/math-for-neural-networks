"""
2D Gradient Descent — Visualizing the Optimization Path
=======================================================

This example demonstrates gradient descent on 2D functions
where we can visualize the optimization path and understand
how the algorithm navigates the loss landscape.

This example demonstrates:
1. Gradient descent on quadratic functions
2. Rosenbrock function (non-convex valley)
3. Effect of starting point
4. Path visualization (text-based)
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import (
    quadratic,
    rosenbrock,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def text_contour(values: np.ndarray, width: int = 40, height: int = 20) -> None:
    """Print a text-based contour plot of a 2D function grid."""
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


def example_1_quadratic_landscape() -> None:
    """Gradient descent on a quadratic function with visualization."""
    print_section("Example 1: Quadratic Function Landscape")

    print("""
    Function: f(x,y) = x^2 + y^2 (circular contours)
    Minimum at origin (0, 0)
    """)

    def f_2d(params: np.ndarray) -> float:
        return quadratic(params)

    def grad_2d(params: np.ndarray) -> np.ndarray:
        return 2.0 * params

    # Create contour grid
    x_range = np.linspace(-5, 5, 40)
    y_range = np.linspace(-5, 5, 20)
    Z = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            Z[i, j] = f_2d(np.array([x, y]))

    print("  Contour plot of f(x,y) = x^2 + y^2:")
    text_contour(Z)

    # Run gradient descent from different starting points
    starting_points = [
        np.array([4.0, 4.0]),
        np.array([-3.0, 2.0]),
        np.array([0.0, 5.0]),
    ]

    for start in starting_points:
        result = gradient_descent(
            f_2d, grad_2d, start,
            learning_rate=0.3, max_iterations=20, param_tol=1e-8,
        )
        print(f"\n  Start: ({start[0]:.1f}, {start[1]:.1f})")
        print(f"  End:   ({result.parameters[0]:.4f}, {result.parameters[1]:.4f})")
        print(f"  Loss:  {result.final_objective:.8f}, Iterations: {result.iterations}")


def example_2_rosenbrock() -> None:
    """Gradient descent on the Rosenbrock function."""
    print_section("Example 2: Rosenbrock Function (Non-Convex Valley)")

    print("""
    Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    Minimum at (1, 1) where f = 0

    This is a classic optimization test function.
    The minimum lies in a narrow, curved valley.
    Gradient descent often struggles here because the
    gradient points across the valley, not along it.
    """)

    def f_2d(params: np.ndarray) -> float:
        return rosenbrock(params)

    def grad_2d(params: np.ndarray) -> np.ndarray:
        x, y = params[0], params[1]
        dx = -2.0 * (1 - x) - 400.0 * x * (y - x ** 2)
        dy = 200.0 * (y - x ** 2)
        return np.array([dx, dy])

    starting_points = [
        np.array([-1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 3.0]),
    ]

    for start in starting_points:
        result = gradient_descent(
            f_2d, grad_2d, start,
            learning_rate=0.001, max_iterations=5000, param_tol=1e-10,
        )
        dist = np.sqrt(
            (result.parameters[0] - 1.0) ** 2
            + (result.parameters[1] - 1.0) ** 2
        )
        print(f"\n  Start: ({start[0]:.1f}, {start[1]:.1f})")
        print(f"  End:   ({result.parameters[0]:.6f}, {result.parameters[1]:.6f})")
        print(f"  Loss:  {result.final_objective:.10f}")
        print(f"  Distance to minimum: {dist:.6f}")
        print(f"  Iterations: {result.iterations}")


def example_3_starting_point_effect() -> None:
    """Show how starting point affects convergence."""
    print_section("Example 3: Starting Point Effect")

    print("""
    Function: f(x,y) = x^2 + 10*y^2
    Different starting points converge at different rates.
    """)

    def f_2d(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad_2d(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    starts = [
        ("Near minimum", np.array([0.5, 0.5])),
        ("Far in x", np.array([10.0, 0.5])),
        ("Far in y", np.array([0.5, 10.0])),
        ("Far in both", np.array([10.0, 10.0])),
    ]

    print(f"  {'Start':>14} | {'(x, y)':>16} | {'f(x,y)':>12} | {'Iters':>6}")
    print("  " + "-" * 55)

    for label, start in starts:
        result = gradient_descent(
            f_2d, grad_2d, start,
            learning_rate=0.04, max_iterations=100, param_tol=1e-8,
        )
        print(
            f"  {label:>14} | "
            f"({result.parameters[0]:>7.4f}, {result.parameters[1]:>7.4f}) | "
            f"{result.final_objective:>12.8f} | "
            f"{result.iterations:>6}"
        )


if __name__ == "__main__":
    print("2D GRADIENT DESCENT - VISUALIZING THE OPTIMIZATION PATH")
    print("=" * 55)

    example_1_quadratic_landscape()
    example_2_rosenbrock()
    example_3_starting_point_effect()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Quadratic functions have circular/spherical contours - easy for GD
    2. Rosenbrock has a narrow curved valley - GD struggles
    3. Starting point affects convergence speed
    4. Non-convex functions may have local minima or saddle points
    5. Real neural networks have high-dimensional, non-convex landscapes
    """)
