"""
Experiment 6: Vanishing Gradient Intuition
==========================================

Hypothesis: Deep networks with sigmoid activations suffer from vanishing gradients.

Method: Compute gradient through N layers of sigmoid activation.

Result: Gradient magnitude decreases exponentially with depth.

Conclusion: This demonstrates why deeper networks need careful initialization
or different activation functions.
"""

import numpy as np

from math_for_neural_networks.neural_networks.activations import sigmoid


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 6: Vanishing Gradient Intuition")
    print("=" * 60)

    print("\n  Gradient through N sigmoid layers:")
    print("  Each layer multiplies gradient by sigmoid'(z) <= 0.25")
    print()

    x = 0.5
    grad = 1.0

    print(f"  {'Layer':>5} | {'Gradient':>12} | {'Cumulative':>12}")
    print("  " + "-" * 35)

    for layer in range(10):
        # sigmoid'(x) <= 0.25 for any x
        local_grad = sigmoid(x) * (1 - sigmoid(x))
        grad *= local_grad
        print(f"  {layer:>5} | {local_grad:>12.8f} | {grad:>12.8f}")
        x += 0.1  # slight shift

    print(f"\n  After 10 layers: gradient = {grad:.2e}")
    print(f"  Initial gradient was 1.0")
    print(f"  Reduction factor: {grad / 1.0:.2e}")

    print("\n  CONCLUSION: Gradients shrink exponentially through sigmoid layers.")
    print("  This is the vanishing gradient problem.")
    print("  Solutions: ReLU, batch normalization, residual connections.")


if __name__ == "__main__":
    run_experiment()
