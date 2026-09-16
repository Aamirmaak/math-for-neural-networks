"""
Affine Transformation and Linear Layer Mathematics
==================================================

The fundamental building block of neural networks is the affine transformation:

    z = Wx + b

This example demonstrates:
1. Single-sample affine transformation
2. Batched affine transformation
3. Shape conventions
4. Why bias matters
5. Connection to neural network layers
"""

import numpy as np

from math_for_neural_networks.neural_networks.layers import affine_transform


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_single_sample() -> None:
    """Single-sample affine transformation."""
    print_section("Example 1: Single Sample")

    print("""
    z = Wx + b

    x = input vector (3 features)
    W = weight matrix (2 output features x 3 input features)
    b = bias vector (2 output features)
    z = output vector (2 features)
    """)

    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    b = np.array([0.1, 0.2])

    z = affine_transform(x, W, b)

    print(f"  Input x:  {x}  shape={x.shape}")
    print(f"  Weights W:\n{W}  shape={W.shape}")
    print(f"  Bias b:   {b}  shape={b.shape}")
    print(f"  Output z: {z}  shape={z.shape}")
    print(f"\n  Manual: Wx + b = {W @ x + b}")


def example_2_why_bias() -> None:
    """Demonstrate why bias is needed."""
    print_section("Example 2: Why Bias Matters")

    print("""
    Without bias: z = Wx (linear only)
    With bias:    z = Wx + b (can shift output)

    Bias allows the model to fit data that doesn't pass through the origin.
    """)

    x = np.array([1.0, 2.0])
    W = np.array([[1.0, 0.0], [0.0, 1.0]])

    z_no_bias = affine_transform(x, W)
    z_with_bias = affine_transform(x, W, b=np.array([10.0, 20.0]))

    print(f"  Without bias: {z_no_bias}")
    print(f"  With bias:    {z_with_bias}")
    print(f"  Bias shifts the output by [10, 20]")


def example_3_batch() -> None:
    """Batched affine transformation."""
    print_section("Example 3: Batched Input")

    print("""
    For batch processing:
        X: (batch_size, input_features)
        W: (output_features, input_features)
        b: (output_features,)
        Z: (batch_size, output_features)

    Z = XW^T + b
    """)

    X = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])
    W = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    b = np.array([0.0, 0.0])

    Z = affine_transform(X, W, b)

    print(f"  Input X: shape={X.shape}")
    print(f"  Weights W: shape={W.shape}")
    print(f"  Output Z: shape={Z.shape}")
    print(f"\n  Z = XW^T + b:")
    print(f"  {Z}")


def example_4_neural_network_connection() -> None:
    """Connect to neural network layers."""
    print_section("Example 4: Neural Network Connection")

    print("""
    A neural network layer is:
        output = activation(input @ W^T + b)

    The affine transformation is the "linear" part.
    The activation adds non-linearity.

    Example: 3-input, 4-hidden, 2-output network
    """)

    rng = np.random.default_rng(42)
    x = rng.standard_normal(3)
    W1 = rng.standard_normal((4, 3))
    b1 = rng.standard_normal(4)
    W2 = rng.standard_normal((2, 4))
    b2 = rng.standard_normal(2)

    h = affine_transform(x, W1, b1)
    print(f"  Input: shape={x.shape}")
    print(f"  Hidden: shape={h.shape} (after first layer)")
    print(f"  Output: affine_transform(hidden, W2, b2)")

    z = affine_transform(h, W2, b2)
    print(f"  Output: shape={z.shape}")


if __name__ == "__main__":
    print("AFFINE TRANSFORMATION AND LINEAR LAYER MATHEMATICS")
    print("=" * 60)

    example_1_single_sample()
    example_2_why_bias()
    example_3_batch()
    example_4_neural_network_connection()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Affine transformation: z = Wx + b
    2. W determines output dimension (rows) and input dimension (columns)
    3. Bias allows shifting the output (without it, only linear)
    4. Batch processing: Z = XW^T + b
    5. Every neural network layer is an affine transformation + activation
    """)
