"""
Experiment 2: Cross-Entropy as Predicted Probability Changes
=============================================================

Hypothesis: Cross-entropy loss increases as predicted probability of true class decreases.

Method: For one-hot targets, vary the predicted probability of the true class
and measure cross-entropy loss.

Result: Cross-entropy = -log(p_true), which increases as p_true -> 0.

Conclusion: Penalizes confident wrong predictions heavily (infinite loss for p_true = 0).
"""

import numpy as np

from math_for_neural_networks.probability.information import cross_entropy


def run_experiment() -> None:
    """Run the cross-entropy experiment."""
    print("=" * 60)
    print("  EXPERIMENT 2: Cross-Entropy vs Predicted Probability")
    print("=" * 60)

    print("\n  Setup: 3-class problem, true class = 0")
    print("  We vary the predicted probability of the true class p0")
    print("  and distribute the remaining probability equally among other classes.")

    print(f"\n  {'p0':>6} | {'Other probs':>20} | {'H(p,q)':>10} | {'-log(p0)':>10}")
    print("  " + "-" * 55)

    for p0 in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]:
        remaining = (1.0 - p0) / 2.0
        predicted = np.array([p0, remaining, remaining])
        target = np.array([1.0, 0.0, 0.0])

        ce = cross_entropy(target, predicted)
        neg_log_p0 = -np.log(p0)

        print(
            f"  {p0:>6.3f} | [{remaining:.3f}, {remaining:.3f}] | {ce:>10.4f} | {neg_log_p0:>10.4f}"
        )

    print("\n  OBSERVATIONS:")
    print("  - Cross-entropy = -log(p0) for one-hot targets")
    print("  - Loss approaches infinity as p0 -> 0 (confident wrong prediction)")
    print("  - Loss approaches 0 as p0 -> 1 (perfect prediction)")
    print("  - This is why neural networks are penalized heavily for confident mistakes")


if __name__ == "__main__":
    run_experiment()
