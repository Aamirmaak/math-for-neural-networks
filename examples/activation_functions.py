"""
Activation Functions
====================

Activation functions introduce non-linearity into neural networks.
Without them, a stack of linear layers is still just a linear transformation.

This example demonstrates:
1. Sigmoid: range, saturation, derivative
2. Tanh: zero-centered, symmetry, derivative
3. ReLU: sparsity, dying ReLU problem
4. GELU: smooth ReLU approximation
5. Comparison of all activations
"""

import numpy as np

from math_for_neural_networks.neural_networks.activations import (
    gelu,
    gelu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_sigmoid() -> None:
    """Sigmoid activation."""
    print_section("Example 1: Sigmoid")

    print("""
    Formula: sigma(x) = 1 / (1 + exp(-x))
    Range: (0, 1)
    Derivative: sigma(x) * (1 - sigma(x))
    Max derivative: 0.25 at x=0
    """)

    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    print(f"  {'x':>6} | {'sigma(x)':>10} | {'sigma_prime(x)':>15}")
    print("  " + "-" * 35)
    for x in x_values:
        print(f"  {x:>6.1f} | {sigmoid(x):>10.6f} | {sigmoid_derivative(x):>15.6f}")

    print("\n  Properties:")
    print("  - sigma(0) = 0.5")
    print("  - sigma(-x) = 1 - sigma(x)")
    print("  - Saturates for |x| > 5 (vanishing gradient)")
    print("  - Output is always positive (not zero-centered)")


def example_2_tanh() -> None:
    """Tanh activation."""
    print_section("Example 2: Tanh")

    print("""
    Formula: tanh(x)
    Range: (-1, 1)
    Derivative: 1 - tanh(x)^2
    Max derivative: 1.0 at x=0
    """)

    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    print(f"  {'x':>6} | {'tanh(x)':>10} | {'tanh_prime(x)':>15}")
    print("  " + "-" * 35)
    for x in x_values:
        print(f"  {x:>6.1f} | {tanh(x):>10.6f} | {tanh_derivative(x):>15.6f}")

    print("\n  Properties:")
    print("  - tanh(0) = 0 (zero-centered)")
    print("  - tanh(-x) = -tanh(x) (odd symmetry)")
    print("  - Saturates for |x| > 5 (vanishing gradient)")
    print("  - Zero-centered output (unlike sigmoid)")


def example_3_relu() -> None:
    """ReLU activation."""
    print_section("Example 3: ReLU")

    print("""
    Formula: max(0, x)
    Range: [0, inf)
    Derivative: 1 if x > 0, 0 if x < 0
    At x = 0: undefined (convention: 0)
    """)

    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    print(f"  {'x':>6} | {'ReLU(x)':>10} | {'ReLU_prime(x)':>15}")
    print("  " + "-" * 35)
    for x in x_values:
        print(f"  {x:>6.1f} | {relu(x):>10.1f} | {relu_derivative(x):>15.1f}")

    print("\n  Properties:")
    print("  - Identity for positive values")
    print("  - Outputs 0 for negative values (sparsity)")
    print("  - No vanishing gradient for x > 0")
    print("  - Dying ReLU: neurons that output 0 never recover")


def example_4_gelu() -> None:
    """GELU activation."""
    print_section("Example 4: GELU")

    print("""
    Formula: x * Phi(x) where Phi is standard normal CDF
    Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Range: (-inf, inf) but mostly [0, inf) for large x
    """)

    x_values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    print(f"  {'x':>6} | {'GELU(x)':>10} | {'GELU_prime(x)':>15}")
    print("  " + "-" * 35)
    for x in x_values:
        print(f"  {x:>6.1f} | {gelu(x):>10.6f} | {gelu_derivative(x):>15.6f}")

    print("\n  Properties:")
    print("  - Smooth approximation of ReLU")
    print("  - Non-zero for negative values")
    print("  - Used in Transformers (BERT, GPT)")


def example_5_comparison() -> None:
    """Compare all activations."""
    print_section("Example 5: Comparison")

    x = np.linspace(-5, 5, 11)
    print(f"  {'x':>6} | {'Sigmoid':>10} | {'Tanh':>10} | {'ReLU':>10} | {'GELU':>10}")
    print("  " + "-" * 55)
    for xi in x:
        print(
            f"  {xi:>6.1f} | {sigmoid(xi):>10.4f} | {tanh(xi):>10.4f} | "
            f"{relu(xi):>10.4f} | {gelu(xi):>10.4f}"
        )


if __name__ == "__main__":
    print("ACTIVATION FUNCTIONS")
    print("=" * 60)

    example_1_sigmoid()
    example_2_tanh()
    example_3_relu()
    example_4_gelu()
    example_5_comparison()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Sigmoid: (0,1), good for binary classification output
    2. Tanh: (-1,1), zero-centered, good for RNNs
    3. ReLU: [0,inf), most common hidden activation
    4. GELU: smooth ReLU, used in Transformers
    5. All suffer from vanishing gradients for large |x|
       (except ReLU for positive x)
    """)
