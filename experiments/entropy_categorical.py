"""
Experiment 1: Entropy of Categorical Distributions
====================================================

Hypothesis: Entropy increases with uniformity and decreases with concentration.

Method: Compute entropy for categorical distributions with varying concentrations.

Result: Uniform distribution has maximum entropy H = log(k).
Concentrated distributions have lower entropy.

Conclusion: Entropy measures uncertainty; more uniform = more uncertain.
"""

import numpy as np

from math_for_neural_networks.probability.information import entropy


def run_experiment() -> None:
    """Run the entropy experiment."""
    print("=" * 60)
    print("  EXPERIMENT 1: Entropy of Categorical Distributions")
    print("=" * 60)

    k = 4  # number of categories

    # Different concentration levels
    concentrations = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  Distribution: {k} categories")
    print(f"  Concentration parameter alpha: lower = more uniform, higher = more peaked")

    print(f"\n  {'alpha':>8} | {'Distribution':>30} | {'H':>8}")
    print("  " + "-" * 50)

    for alpha in concentrations:
        # Generate Dirichlet-like distribution
        probs = np.ones(k) * alpha
        probs = probs / np.sum(probs)
        h = entropy(probs)
        print(f"  {alpha:>6.1f} | {str(np.round(probs, 3)):>30} | {h:>8.4f}")

    # Maximum entropy (uniform)
    uniform = np.ones(k) / k
    h_max = entropy(uniform)
    print(f"\n  Maximum entropy (uniform): H = {h_max:.4f} = log({k}) = {np.log(k):.4f}")

    # Concentrated
    concentrated = np.array([0.97, 0.01, 0.01, 0.01])
    h_min = entropy(concentrated)
    print(f"  Concentrated example:      H = {h_min:.4f}")

    print("\n  CONCLUSION: Entropy decreases as distribution becomes more peaked.")
    print("  This is why low-entropy models are more confident in their predictions.")


if __name__ == "__main__":
    run_experiment()
