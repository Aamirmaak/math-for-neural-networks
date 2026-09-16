"""
Experiment 1: Activation Function Comparison
=============================================

Hypothesis: Different activation functions have different output ranges,
saturation behavior, and derivative characteristics that affect training.

Method: Compare sigmoid, tanh, ReLU, and GELU across a range of inputs.

Result: All activations produce different output ranges and derivative behaviors.
ReLU has the largest derivative range. Sigmoid and tanh saturate.

Conclusion: The choice of activation function affects gradient flow and training dynamics.
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


def run_experiment() -> None:
    """Run the activation function comparison experiment."""
    print("=" * 60)
    print("  EXPERIMENT 1: Activation Function Comparison")
    print("=" * 60)

    x = np.linspace(-5, 5, 100)

    # Output ranges
    print("\n  Output ranges:")
    print(f"    Sigmoid: [{sigmoid(x).min():.4f}, {sigmoid(x).max():.4f}]")
    print(f"    Tanh:    [{tanh(x).min():.4f}, {tanh(x).max():.4f}]")
    print(f"    ReLU:    [{relu(x).min():.4f}, {relu(x).max():.4f}]")
    print(f"    GELU:    [{gelu(x).min():.4f}, {gelu(x).max():.4f}]")

    # Derivative ranges
    print("\n  Derivative ranges:")
    print(f"    Sigmoid: [{sigmoid_derivative(x).min():.4f}, {sigmoid_derivative(x).max():.4f}]")
    print(f"    Tanh:    [{tanh_derivative(x).min():.4f}, {tanh_derivative(x).max():.4f}]")
    print(f"    ReLU:    [{relu_derivative(x).min():.4f}, {relu_derivative(x).max():.4f}]")
    print(f"    GELU:    [{gelu_derivative(x).min():.4f}, {gelu_derivative(x).max():.4f}]")

    # Saturation behavior
    print("\n  Saturation at |x|=5:")
    print(f"    Sigmoid(5):  {sigmoid(5.0):.6f} (near 1)")
    print(f"    Sigmoid(-5): {sigmoid(-5.0):.6f} (near 0)")
    print(f"    Tanh(5):     {tanh(5.0):.6f} (near 1)")
    print(f"    ReLU(5):     {relu(5.0):.1f} (no saturation)")
    print(f"    GELU(5):     {gelu(5.0):.4f} (near 5)")

    # Gradient behavior
    print("\n  Gradient at extreme values:")
    print(f"    Sigmoid derivative at x=10: {sigmoid_derivative(10.0):.8f} (vanishes)")
    print(f"    Tanh derivative at x=10:    {tanh_derivative(10.0):.8f} (vanishes)")
    print(f"    ReLU derivative at x=10:    {relu_derivative(10.0):.1f} (constant)")
    print(f"    GELU derivative at x=10:    {gelu_derivative(10.0):.4f} (near 1)")

    print("\n  OBSERVATIONS:")
    print("  - Sigmoid and tanh saturate for large |x| (vanishing gradient)")
    print("  - ReLU has constant gradient for x > 0 (no vanishing)")
    print("  - GELU is a smooth approximation of ReLU")
    print("  - ReLU outputs are not zero-centered (all >= 0)")
    print("  - Tanh is zero-centered (outputs in [-1, 1])")


if __name__ == "__main__":
    run_experiment()
