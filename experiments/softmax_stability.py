"""
Experiment 2: Softmax Numerical Stability
==========================================

Hypothesis: Naive softmax computation overflows for large logits,
while the stable version (subtracting max) handles extreme values.

Method: Compare naive and stable softmax on increasingly large logits.

Result: Naive softmax produces NaN/inf for logits > 700.
Stable softmax produces correct results for any finite logits.

Conclusion: Always use the numerically stable softmax implementation.
"""

import numpy as np

from math_for_neural_networks.neural_networks.attention import softmax


def naive_softmax(x: np.ndarray) -> np.ndarray:
    """Naive softmax (numerically unstable)."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def run_experiment() -> None:
    """Run the softmax numerical stability experiment."""
    print("=" * 60)
    print("  EXPERIMENT 2: Softmax Numerical Stability")
    print("=" * 60)

    print("\n  Comparing naive vs stable softmax:")
    print(f"  {'Logits':>20} | {'Naive sum':>12} | {'Stable sum':>12} | {'Naive finite':>12}")
    print("  " + "-" * 65)

    test_cases = [
        np.array([1.0, 2.0, 3.0]),
        np.array([100.0, 200.0, 300.0]),
        np.array([500.0, 600.0, 700.0]),
        np.array([1000.0, 1001.0, 1002.0]),
        np.array([-1000.0, -1001.0, -1002.0]),
    ]

    for logits in test_cases:
        naive = naive_softmax(logits)
        stable = softmax(logits)
        naive_finite = np.all(np.isfinite(naive))
        print(
            f"  {str(logits):>20} | {np.sum(naive):>12.6f} | "
            f"{np.sum(stable):>12.6f} | {str(naive_finite):>12}"
        )

    # Demonstrate overflow
    print("\n  Overflow demonstration:")
    x = np.array([1000.0, 1001.0, 1002.0])
    try:
        naive_result = naive_softmax(x)
        print(f"    Naive: {naive_result}")
    except Exception as e:
        print(f"    Naive: {type(e).__name__}: {e}")

    stable_result = softmax(x)
    print(f"    Stable: {stable_result}")
    print(f"    Sum: {np.sum(stable_result):.6f}")

    print("\n  OBSERVATIONS:")
    print("  - Naive softmax overflows for logits > ~700")
    print("  - Stable softmax (subtracting max) works for any finite input")
    print("  - The stable version is mathematically identical")
    print("  - Always use the stable implementation in practice")


if __name__ == "__main__":
    run_experiment()
