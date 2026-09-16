"""
Derivatives — Intuition and Neural Network Connection
=====================================================

A derivative measures the instantaneous rate of change of a function.
Geometrically, it is the slope of the tangent line at a point.

This example demonstrates:
1. What a derivative means visually
2. How to compute derivatives analytically
3. How to verify derivatives numerically
4. Why derivatives matter for neural networks
"""

import numpy as np

from math_for_neural_networks.calculus.derivatives import (
    quadratic,
    quadratic_derivative,
    sigmoid,
    sigmoid_derivative,
    sin_func,
    sin_derivative,
    exp_func,
    exp_derivative,
)
from math_for_neural_networks.calculus.finite_differences import (
    central_difference,
    forward_difference,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_basic_derivative() -> None:
    """The derivative of x² is 2x."""
    print_section("Example 1: Derivative of x²")

    print("""
    Function: f(x) = x²
    Derivative: f'(x) = 2x

    The derivative tells us: "How fast is x² changing at point x?"
    """)

    test_points = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    print(f"  {'x':>6} | {'f(x)':>8} | {'f(x+h)-f(x)':>14} | {'Δx':>6} | {'Secant slope':>14} | {'f''(x)':>8}")
    print("  " + "-" * 72)

    for x in test_points:
        h = 0.01
        secant = (quadratic(x + h) - quadratic(x)) / h
        print(f"  {x:>6.1f} | {quadratic(x):>8.1f} | {(quadratic(x+h)):`>14.4f}` | {h:>6.2f} | {secant:>14.4f} | {quadratic_derivative(x):>8.1f}")

    print("""
    As h → 0, the secant slope approaches the tangent slope (derivative).
    At x=0: slope=0 (flat, minimum)
    At x=3: slope=6 (steep, increasing)
    """)


def example_2_activation_derivatives() -> None:
    """Sigmoid and tanh derivatives — critical for backpropagation."""
    print_section("Example 2: Activation Function Derivatives")

    print("""
    SIGMOID: σ(x) = 1/(1+e⁻ˣ)
    Derivative: σ'(x) = σ(x)·(1 - σ(x))

    This elegant formula means: "The derivative depends on the output!"
    - At x=0 (output=0.5): derivative = 0.5 × 0.5 = 0.25 (maximum slope)
    - At x=±5 (output≈0/1): derivative ≈ 0 (flat, no learning)
    """)

    x_vals = [-5.0, -2.0, 0.0, 2.0, 5.0]
    print(f"  {'x':>6} | {'σ(x)':>8} | {'σ''(x) analytical':>18} | {'σ''(x) numerical':>18}")
    print("  " + "-" * 56)

    for x in x_vals:
        analytical = sigmoid_derivative(x)
        numerical = central_difference(sigmoid, x, h=1e-7)
        print(f"  {x:>6.1f} | {sigmoid(x):>8.4f} | {analytical:>18.6f} | {numerical:>18.6f}")

    print("""
    Key insight: When sigmoid output is near 0 or 1, gradients vanish.
    This is the VANISHING GRADIENT PROBLEM — why ReLU was invented!
    """)


def example_3_numerical_verification() -> None:
    """Verify analytical derivatives using finite differences."""
    print_section("Example 3: Numerical Verification")

    print("""
    We can verify any analytical derivative using the definition:

        f'(x) = lim[h→0] (f(x+h) - f(x)) / h

    Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    This is O(h²) accurate — much better than forward difference.
    """)

    functions = [
        (sin_func, sin_derivative, "sin(x)"),
        (exp_func, exp_derivative, "eˣ"),
        (quadratic, quadratic_derivative, "x²"),
    ]

    x_val = 1.0
    print(f"  Verification at x = {x_val}:\n")
    print(f"  {'Function':>10} | {'Analytical':>12} | {'Numerical':>12} | {'Error':>12}")
    print("  " + "-" * 52)

    for func, deriv, name in functions:
        analytical = deriv(x_val)
        numerical = central_difference(func, x_val, h=1e-7)
        error = abs(analytical - numerical)
        print(f"  {name:>10} | {analytical:>12.8f} | {numerical:>12.8f} | {error:>12.2e}")

    print("\n  All errors should be < 1e-6, confirming our derivatives are correct.")


def example_4_nn_connection() -> None:
    """Why derivatives matter for neural network training."""
    print_section("Example 4: Connection to Neural Networks")

    print("""
    In a neural network, we want to minimize a loss function L.

    Gradient descent updates: w_new = w_old - η · ∂L/∂w
    where η is the learning rate.

    The derivative ∂L/∂w tells us:
    - DIRECTION: Should w increase or decrease?
    - MAGNITUDE: How much should w change?

    Example: Sigmoid derivative shows why training stalls

    Network: x → sigmoid(wx) → loss
    Gradient: ∂L/∂w = ∂L/∂a · σ'(wx) · x

    If sigmoid output ≈ 0 or ≈ 1, then σ'(wx) ≈ 0
    → Gradient ≈ 0 → No learning!
    """)

    # Demonstrate the vanishing gradient problem
    print("  Sigmoid output vs derivative:\n")
    for x in [-6, -3, 0, 3, 6]:
        s = sigmoid(x)
        d = sigmoid_derivative(x)
        bar_len = int(d * 40)
        bar = "█" * bar_len
        print(f"  x={x:+d}: σ(x)={s:.3f}, σ'(x)={d:.4f} |{bar}")

    print("""
    Maximum gradient (0.25) only occurs at x=0.
    For large |x|, sigmoid saturates → gradients vanish.

    SOLUTION: ReLU activation
    ReLU'(x) = 1 for all x > 0 — no saturation!
    """)


if __name__ == "__main__":
    print("DERIVATIVES — INTUITION AND NEURAL NETWORK CONNECTION")
    print("=" * 55)

    example_1_basic_derivative()
    example_2_activation_derivatives()
    example_3_numerical_verification()
    example_4_nn_connection()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Derivatives measure rate of change (slope)
    2. Analytical derivatives are exact; numerical ones approximate
    3. Central differences are O(h²) accurate — use h ≈ 1e-5
    4. Sigmoid/tanh derivatives → vanishing gradient problem
    5. ReLU derivative is constant (1 or 0) → avoids vanishing gradients
    """)
