"""
Gradients — Multi-Dimensional Derivatives
==========================================

The gradient is the vector of all partial derivatives of a function.
It points in the direction of steepest ascent.

This example demonstrates:
1. What a gradient means geometrically
2. How to compute gradients numerically
3. Gradient magnitude and direction
4. Connection to gradient descent in neural networks
"""

import numpy as np

from math_for_neural_networks.calculus.gradients import (
    gradient_direction,
    gradient_magnitude,
    numerical_gradient,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_gradient_basics() -> None:
    """Basic gradient computation."""
    print_section("Example 1: Gradient of x² + y²")

    print("""
    Function: f(x,y) = x² + y²
    Gradient: ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

    At point (3, 4): ∇f = [6, 8]
    This means:
    - Moving in x direction: function increases at rate 6
    - Moving in y direction: function increases at rate 8
    """)

    def f(point: np.ndarray) -> float:
        return point[0] ** 2 + point[1] ** 2

    points = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([3.0, 4.0]),
        np.array([-2.0, 1.0]),
    ]

    print(f"  {'Point':>12} | {'f(point)':>10} | {'∇f (numerical)':>20} | {'∇f (analytical)':>20}")
    print("  " + "-" * 70)

    for point in points:
        grad = numerical_gradient(f, point)
        analytical = 2 * point
        print(
            f"  ({point[0]:>5.1f}, {point[1]:>5.1f}) | "
            f"{f(point):>10.1f} | "
            f"[{grad[0]:>7.3f}, {grad[1]:>7.3f}] | "
            f"[{analytical[0]:>7.3f}, {analytical[1]:>7.3f}]"
        )


def example_2_gradient_direction() -> None:
    """Gradient direction and magnitude."""
    print_section("Example 2: Gradient Direction and Magnitude")

    print("""
    Gradient magnitude: |∇f| = √(Σ(∂f/∂xᵢ)²)
    Gradient direction: ∇f / |∇f| (unit vector pointing uphill)

    The gradient always points TOWARD steepest ASCENT.
    For MINIMIZATION, we move OPPOSITE to the gradient.
    """)

    def f(point: np.ndarray) -> float:
        return point[0] ** 2 + point[1] ** 2

    points = [
        np.array([3.0, 4.0]),
        np.array([1.0, 1.0]),
        np.array([-2.0, 3.0]),
    ]

    for point in points:
        grad = numerical_gradient(f, point)
        mag = gradient_magnitude(grad)
        direction = gradient_direction(grad)

        print(f"\n  Point: ({point[0]}, {point[1]})")
        print(f"  Gradient: [{grad[0]:.3f}, {grad[1]:.3f}]")
        print(f"  Magnitude: {mag:.3f}")
        print(f"  Direction: [{direction[0]:.3f}, {direction[1]:.3f}]")
        print(f"  Direction is unit vector: {abs(np.linalg.norm(direction) - 1.0) < 1e-10}")


def example_3_gradient_descent() -> None:
    """Simulate gradient descent on a simple function."""
    print_section("Example 3: Gradient Descent Simulation")

    print("""
    We minimize f(x,y) = x² + y² using gradient descent.
    Starting at (3, 4), learning rate η = 0.1

    Update rule: [x,y] ← [x,y] - η · ∇f(x,y)
    """)

    def f(point: np.ndarray) -> float:
        return point[0] ** 2 + point[1] ** 2

    position = np.array([3.0, 4.0])
    lr = 0.1

    print(f"  {'Step':>4} | {'Position':>12} | {'f(x,y)':>10} | {'∇f':>16}")
    print("  " + "-" * 50)

    for step in range(6):
        grad = numerical_gradient(f, position)
        print(f"  {step:>4} | ({position[0]:>5.2f}, {position[1]:>5.2f}) | {f(position):>10.3f} | [{grad[0]:>7.3f}, {grad[1]:>7.3f}]")
        position = position - lr * grad

    print(f"\n  After 5 steps, f(x,y) = {f(position):.6f} (started at {f(np.array([3.0, 4.0])):.3f})")
    print("  The function value decreases as we follow the negative gradient.")


def example_4_nn_gradient() -> None:
    """Gradient in a simple neural network context."""
    print_section("Example 4: Gradient in Neural Network")

    print("""
    Consider a single neuron: output = σ(w₁x₁ + w₂x₂ + b)
    Loss = (output - target)²

    We need: ∂Loss/∂w₁, ∂Loss/∂w₂, ∂Loss/∂b

    This is a multi-variable optimization problem!
    """)

    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def loss(params: np.ndarray) -> float:
        w1, w2, b = params[0], params[1], params[2]
        x1, x2 = 1.0, 2.0  # fixed inputs
        target = 1.0
        output = sigmoid(w1 * x1 + w2 * x2 + b)
        return (output - target) ** 2

    # Compute gradient of loss w.r.t. [w1, w2, b]
    params = np.array([0.5, 0.5, 0.0])  # initial weights
    grad = numerical_gradient(loss, params, h=1e-5)

    print(f"  Initial weights: w1={params[0]}, w2={params[1]}, b={params[2]}")
    print(f"  Loss: {loss(params):.6f}")
    print(f"  Gradient: ∂L/∂w1={grad[0]:.4f}, ∂L/∂w2={grad[1]:.4f}, ∂L/∂b={grad[2]:.4f}")

    # One step of gradient descent
    lr = 1.0
    new_params = params - lr * grad
    print(f"\n  After one gradient descent step (lr={lr}):")
    print(f"  New weights: w1={new_params[0]:.4f}, w2={new_params[1]:.4f}, b={new_params[2]:.4f}")
    print(f"  New loss: {loss(new_params):.6f}")
    print(f"  Loss decreased: {loss(new_params) < loss(params)}")


if __name__ == "__main__":
    print("GRADIENTS — MULTI-DIMENSIONAL DERIVATIVES")
    print("=" * 55)

    example_1_gradient_basics()
    example_2_gradient_direction()
    example_3_gradient_descent()
    example_4_nn_gradient()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Gradient = vector of all partial derivatives
    2. Points toward steepest ascent (negative gradient → descent)
    3. Gradient magnitude = steepness of the function
    4. Gradient descent follows -∇f to find minima
    5. Neural network training = high-dimensional gradient descent
    """)
