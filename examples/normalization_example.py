"""
Layer Normalization
===================

Layer normalization stabilizes neural network training by normalizing
activations to have zero mean and unit variance.

    mu = mean(x)
    sigma^2 = mean((x - mu)^2)
    x_hat = (x - mu) / sqrt(sigma^2 + eps)
    y = gamma * x_hat + beta

This example demonstrates:
1. Basic layer normalization
2. Effect of gamma and beta parameters
3. Batch normalization vs layer normalization
4. Connection to Transformers
"""

import numpy as np

from math_for_neural_networks.neural_networks.normalization import layer_norm, layer_norm_stats


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_basic() -> None:
    """Basic layer normalization."""
    print_section("Example 1: Basic Layer Norm")

    print("""
    For input x:
        mu = mean(x)
        sigma^2 = mean((x - mu)^2)
        x_hat = (x - mu) / sqrt(sigma^2 + eps)
        y = gamma * x_hat + beta
    """)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = layer_norm_stats(x)

    print(f"  Input:   {x}")
    print(f"  Mean:    {stats['mean']:.4f}")
    print(f"  Variance: {stats['variance']:.4f}")
    print(f"  Normalized: {stats['x_hat']}")

    y = layer_norm(x)
    print(f"  Output (gamma=1, beta=0): {y}")


def example_2_gamma_beta() -> None:
    """Effect of gamma and beta."""
    print_section("Example 2: Gamma and Beta")

    print("""
    gamma and beta are learnable parameters:
    - gamma scales the normalized output
    - beta shifts the normalized output

    Initialized to gamma=1, beta=0 (identity).
    During training, these are learned.
    """)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    gamma_1 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    beta_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y_1 = layer_norm(x, gamma=gamma_1, beta=beta_1)

    gamma_2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    beta_2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    y_2 = layer_norm(x, gamma=gamma_2, beta=beta_2)

    print(f"  Input:      {x}")
    print(f"  gamma=2, beta=1:  {y_1}")
    print(f"  gamma=0.5, beta=-1: {y_2}")


def example_3_batch() -> None:
    """Batch layer normalization."""
    print_section("Example 3: Batch Layer Norm")

    print("""
    For batched input, each sample is normalized independently.
    """)

    X = np.array([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0],
    ])

    Y = layer_norm(X)

    print(f"  Input (2 samples):")
    print(f"    Sample 1: {X[0]}")
    print(f"    Sample 2: {X[1]}")
    print(f"\n  Output (normalized independently):")
    print(f"    Sample 1: {Y[0]}")
    print(f"    Sample 2: {Y[1]}")
    print(f"\n  Sample 1 mean: {np.mean(Y[0]):.6f}")
    print(f"  Sample 2 mean: {np.mean(Y[1]):.6f}")


def example_4_nn_connection() -> None:
    """Connection to Transformers."""
    print_section("Example 4: Connection to Transformers")

    print("""
    In Transformers:
    1. Input -> Multi-Head Attention -> Add & Norm
    2. -> Feed-Forward -> Add & Norm

    Layer Norm is applied:
    - After attention (before feed-forward)
    - After feed-forward (before next layer)

    This stabilizes training and enables deep networks.
    """)

    rng = np.random.default_rng(42)
    x = rng.standard_normal(8)
    y = layer_norm(x)

    print(f"  Input (8 features): {np.round(x, 3)}")
    print(f"  Output (normalized): {np.round(y, 3)}")
    print(f"  Output mean: {np.mean(y):.6f}")
    print(f"  Output std:  {np.std(y, ddof=0):.6f}")


if __name__ == "__main__":
    print("LAYER NORMALIZATION")
    print("=" * 60)

    example_1_basic()
    example_2_gamma_beta()
    example_3_batch()
    example_4_nn_connection()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Layer norm: normalize to zero mean, unit variance
    2. gamma and beta are learnable scale and shift parameters
    3. Each sample is normalized independently
    4. Used in Transformers to stabilize training
    5. Epsilon prevents division by zero
    """)
