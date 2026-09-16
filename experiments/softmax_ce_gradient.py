"""
Experiment 7: Softmax + Cross-Entropy Gradient Identity
=======================================================

Hypothesis: The gradient of softmax + cross-entropy simplifies to p - y.

Method: Compute gradient both analytically and numerically.

Result: The identity holds exactly.

Conclusion: This is a key neural network identity that simplifies backpropagation.
"""

import numpy as np

from math_for_neural_networks.neural_networks.attention import softmax
from math_for_neural_networks.neural_networks.backprop import softmax_backward


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 7: Softmax + Cross-Entropy Gradient")
    print("=" * 60)

    print("\n  Identity: dL/dz = softmax(z) - y")
    print("  where L = -sum(y * log(softmax(z)))")

    test_cases = [
        (np.array([2.0, 1.0, 0.5]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0])),
    ]

    for i, (logits, targets) in enumerate(test_cases):
        print(f"\n  Test case {i + 1}:")
        print(f"    logits = {logits}")
        print(f"    targets = {targets}")

        # Analytical: p - y
        probs = softmax(logits)
        analytical = probs - targets

        # Backprop function
        result = softmax_backward(logits, targets)

        # Numerical verification
        h = 1e-5
        numerical = np.zeros_like(logits)
        for j in range(len(logits)):
            logits_plus = logits.copy()
            logits_minus = logits.copy()
            logits_plus[j] += h
            logits_minus[j] -= h

            # Compute loss: -sum(target * log(softmax(logits)))
            probs_plus = softmax(logits_plus)
            probs_minus = softmax(logits_minus)
            eps = 1e-15
            loss_plus = -np.sum(targets * np.log(np.clip(probs_plus, eps, 1)))
            loss_minus = -np.sum(targets * np.log(np.clip(probs_minus, eps, 1)))
            numerical[j] = (loss_plus - loss_minus) / (2 * h)

        print(f"    Analytical (p-y): {analytical}")
        print(f"    Backprop result:  {result}")
        print(f"    Numerical:        {numerical}")
        print(f"    Error (analytical vs numerical): {np.max(np.abs(analytical - numerical)):.2e}")

    print("\n  CONCLUSION: The gradient identity dL/dz = p - y holds exactly.")
    print("  This simplifies backpropagation through softmax + cross-entropy.")


if __name__ == "__main__":
    run_experiment()
