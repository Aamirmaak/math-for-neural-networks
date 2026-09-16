"""
Gradient Descent — The Workhorse of Optimization
=================================================

Gradient descent finds the minimum of a function by iteratively
moving in the direction of steepest descent (negative gradient).

This example demonstrates:
1. Basic gradient descent on a 1D function
2. Multi-dimensional gradient descent
3. Effect of learning rate
4. Convergence behavior
5. Connection to neural network training
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import (
    gradient_descent,
    has_finite_values,
)
from math_for_neural_networks.optimization.objectives import quadratic


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_1d_gradient_descent() -> None:
    """Gradient descent on f(x) = x^2."""
    print_section("Example 1: 1D Gradient Descent on f(x) = x^2")

    print("""
    Function: f(x) = x^2
    Gradient: f'(x) = 2x
    Update rule: x <- x - lr * 2x

    Starting at x = 5.0 with learning rate = 0.1
    The minimum is at x = 0.0
    """)

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    x = np.array([5.0])
    lr = 0.1

    print(f"  {'Step':>4} | {'x':>8} | {'f(x)':>10} | {'gradient':>10}")
    print("  " + "-" * 40)

    for step in range(8):
        g = grad(x)
        print(f"  {step:>4} | {x[0]:>8.4f} | {f(x):>10.4f} | {g[0]:>10.4f}")
        x = x - lr * g

    print(f"\n  After 7 steps, x = {x[0]:.6f}, f(x) = {f(x):.8f}")
    print("  We converge toward x = 0 (the minimum).")


def example_2_2d_gradient_descent() -> None:
    """Gradient descent on f(x,y) = x^2 + 10*y^2."""
    print_section("Example 2: 2D Gradient Descent (Anisotropic)")

    print("""
    Function: f(x,y) = x^2 + 10*y^2
    Gradient: [2x, 20y]

    This function is elongated in the y-direction (10x steeper).
    A small learning rate works for y but is slow for x.
    A large learning rate works for x but may diverge for y.
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    params = np.array([5.0, 5.0])
    lr = 0.04

    print(f"  Learning rate: {lr}")
    print(f"  {'Step':>4} | {'(x, y)':>16} | {'f(x,y)':>12}")
    print("  " + "-" * 40)

    for step in range(10):
        g = grad(params)
        print(f"  {step:>4} | ({params[0]:>6.3f}, {params[1]:>6.3f}) | {f(params):>12.4f}")
        params = params - lr * g

    print(f"\n  Final: ({params[0]:.6f}, {params[1]:.6f}), f = {f(params):.8f}")
    print("  x converges faster than y because the function is less steep in x.")


def example_3_learning_rate_comparison() -> None:
    """Compare different learning rates."""
    print_section("Example 3: Learning Rate Comparison")

    print("""
    Function: f(x) = x^2
    Starting at x = 5.0

    Too small lr: slow convergence
    Just right lr: fast convergence
    Too large lr: oscillation or divergence
    """)

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    learning_rates = [0.05, 0.3, 0.6, 0.95]
    steps = 15

    for lr in learning_rates:
        x = np.array([5.0])
        values = [f(x)]
        for _ in range(steps):
            x = x - lr * grad(x)
            values.append(f(x))

        status = "OK" if values[-1] < 0.01 else (
            "DIVERGED" if values[-1] > 1000 else "SLOW"
        )
        print(f"  lr={lr:>5.2f}: f(0)={values[0]:>10.2f} -> f({steps})={values[-1]:>10.6f}  [{status}]")


def example_4_use_optimization_module() -> None:
    """Use the gradient_descent function from the optimization module."""
    print_section("Example 4: Using the Gradient Descent Module")

    print("""
    The gradient_descent() function provides a complete implementation
    with convergence checking, history tracking, and diagnostics.
    """)

    def objective(params: np.ndarray) -> float:
        return quadratic(params)

    def gradient(params: np.ndarray) -> np.ndarray:
        return 2.0 * params

    x0 = np.array([4.0, -3.0])
    result = gradient_descent(
        objective, gradient, x0,
        learning_rate=0.1,
        max_iterations=50,
        param_tol=1e-8,
    )

    print(f"  Starting point: {x0}")
    print(f"  Learning rate: 0.1")
    print(f"  Max iterations: 50")
    print(f"  Tolerance: 1e-8")
    print(f"\n  Results:")
    print(f"    Final parameters: [{result.parameters[0]:.6f}, {result.parameters[1]:.6f}]")
    print(f"    Final loss: {result.final_objective:.10f}")
    print(f"    Iterations: {result.iterations}")
    print(f"    Converged: {result.converged}")
    print(f"    Finite values: {has_finite_values(result.parameters)}")


if __name__ == "__main__":
    print("GRADIENT DESCENT - THE WORKHORSE OF OPTIMIZATION")
    print("=" * 55)

    example_1_1d_gradient_descent()
    example_2_2d_gradient_descent()
    example_3_learning_rate_comparison()
    example_4_use_optimization_module()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Gradient descent follows the negative gradient to find minima
    2. Learning rate controls step size: too small = slow, too large = diverge
    3. Anisotropic functions (different steepness per dimension) are tricky
    4. The optimization module handles convergence checking automatically
    5. Neural network training is just gradient descent on the loss function
    """)
