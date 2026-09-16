"""
Experiment 3: Bernoulli MLE with Increasing Sample Size
========================================================

Hypothesis: MLE converges to true parameter as sample size increases.

Method: Generate Bernoulli data with known p, compute MLE for increasing n.

Result: MLE error decreases as O(1/sqrt(n)), consistent with statistical theory.

Conclusion: More data -> better parameter estimates (consistency of MLE).
"""

import numpy as np

from math_for_neural_networks.probability.likelihood import bernoulli_mle


def run_experiment() -> None:
    """Run the Bernoulli MLE experiment."""
    print("=" * 60)
    print("  EXPERIMENT 3: Bernoulli MLE Convergence")
    print("=" * 60)

    true_p = 0.6
    np.random.seed(42)

    print(f"\n  True parameter: p = {true_p}")
    print(f"  Method: Generate n samples, compute MLE, repeat 100 times, average error")

    sample_sizes = [5, 10, 25, 50, 100, 250, 500, 1000]
    n_repeats = 100

    print(f"\n  {'n':>6} | {'Mean MLE':>10} | {'Mean |Error|':>12} | {'Std |Error|':>12}")
    print("  " + "-" * 45)

    for n in sample_sizes:
        mles = []
        errors = []
        for _ in range(n_repeats):
            data = np.random.binomial(1, true_p, size=n)
            mle = bernoulli_mle(data)
            mles.append(mle)
            errors.append(abs(mle - true_p))

        mean_mle = np.mean(mles)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print(f"  {n:>6} | {mean_mle:>10.4f} | {mean_error:>12.4f} | {std_error:>12.4f}")

    print("\n  OBSERVATIONS:")
    print("  - Mean MLE approaches true p as n increases")
    print("  - Mean |error| decreases as O(1/sqrt(n))")
    print("  - This demonstrates CONSISTENCY of MLE")
    print("  - In practice: more training data -> better model estimates")


if __name__ == "__main__":
    run_experiment()
