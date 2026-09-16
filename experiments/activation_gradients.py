"""
Experiment 5: Activation Gradient Behavior
===========================================

Hypothesis: Different activation functions have different gradient characteristics.

Method: Compare gradient magnitudes through sigmoid, tanh, and ReLU.

Result: Sigmoid and tanh gradients vanish for large inputs. ReLU has constant gradient.

Conclusion: Choice of activation affects gradient flow.
"""

import numpy as np

from math_for_neural_networks.neural_networks.activations import sigmoid, tanh, relu
from math_for_neural_networks.neural_networks.activations import (
    sigmoid_derivative,
    tanh_derivative,
    relu_derivative,
)


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 5: Activation Gradient Behavior")
    print("=" * 60)

    x = np.linspace(-5, 5, 11)

    print(f"\n  Gradient magnitudes for different activations:")
    print(f"  {'x':>6} | {'sigmoid':>10} | {'tanh':>10} | {'relu':>10}")
    print("  " + "-" * 40)

    for xi in x:
        s = abs(sigmoid_derivative(xi))
        t = abs(tanh_derivative(xi))
        r = abs(relu_derivative(xi))
        print(f"  {xi:>6.1f} | {s:>10.6f} | {t:>10.6f} | {r:>10.1f}")

    print(f"\n  Max gradients:")
    print(f"    sigmoid: {np.max(np.abs(sigmoid_derivative(x))):.6f}")
    print(f"    tanh:    {np.max(np.abs(tanh_derivative(x))):.6f}")
    print(f"    relu:    {np.max(np.abs(relu_derivative(x))):.1f}")

    print(f"\n  Gradient vanishing:")
    print(f"    sigmoid at x=5: {sigmoid_derivative(5.0):.8f} (very small)")
    print(f"    tanh at x=5:    {tanh_derivative(5.0):.8f} (very small)")
    print(f"    relu at x=5:    {relu_derivative(5.0):.1f} (constant)")

    print("\n  CONCLUSION: Sigmoid and tanh suffer from vanishing gradients.")
    print("  ReLU has constant gradient for positive inputs.")


if __name__ == "__main__":
    run_experiment()
