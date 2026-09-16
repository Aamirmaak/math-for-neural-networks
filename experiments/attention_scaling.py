"""
Experiment 3: Attention Scaling
================================

Hypothesis: Without scaling by 1/sqrt(d_k), attention scores grow with
dimension, pushing softmax into saturation and producing near-one-hot weights.

Method: Compute attention with and without scaling for increasing d_k.

Result: Without scaling, attention becomes more peaked as d_k increases.
With scaling, attention remains smooth regardless of d_k.

Conclusion: Scaling by 1/sqrt(d_k) is essential for stable attention.
"""

import numpy as np

from math_for_neural_networks.neural_networks.attention import softmax


def run_experiment() -> None:
    """Run the attention scaling experiment."""
    print("=" * 60)
    print("  EXPERIMENT 3: Attention Scaling")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 5  # sequence length

    print(f"\n  Sequence length: {n}")
    print(f"  Testing different key dimensions (d_k):")

    d_k_values = [4, 16, 64, 256]

    print(f"\n  {'d_k':>6} | {'Without scaling':>20} | {'With scaling':>20} | {'Max weight (no scale)':>22}")
    print("  " + "-" * 75)

    for d_k in d_k_values:
        q = rng.standard_normal(d_k)
        K = rng.standard_normal((n, d_k))

        # Without scaling
        scores_no_scale = q @ K.T
        weights_no_scale = softmax(scores_no_scale)

        # With scaling
        scores_scaled = q @ K.T / np.sqrt(d_k)
        weights_scaled = softmax(scores_scaled)

        max_weight_no_scale = np.max(weights_no_scale)

        print(
            f"  {d_k:>6} | {str(np.round(weights_no_scale, 4)):>20} | "
            f"{str(np.round(weights_scaled, 4)):>20} | {max_weight_no_scale:>22.6f}"
        )

    print("\n  Effect on softmax output:")
    print(f"  {'d_k':>6} | {'Max score (no scale)':>20} | {'Max score (scaled)':>20}")
    print("  " + "-" * 50)

    for d_k in d_k_values:
        q = rng.standard_normal(d_k)
        K = rng.standard_normal((n, d_k))
        scores_no_scale = q @ K.T
        scores_scaled = q @ K.T / np.sqrt(d_k)
        print(
            f"  {d_k:>6} | {np.max(scores_no_scale):>20.4f} | "
            f"{np.max(scores_scaled):>20.4f}"
        )

    print("\n  OBSERVATIONS:")
    print("  - Without scaling, scores grow with d_k")
    print("  - Large scores push softmax toward one-hot (peaked)")
    print("  - With scaling, scores remain ~O(1) regardless of d_k")
    print("  - Scaling preserves gradient flow through attention")
    print("  - This is why Transformers use 1/sqrt(d_k) scaling")


if __name__ == "__main__":
    run_experiment()
