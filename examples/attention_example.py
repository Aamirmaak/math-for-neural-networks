"""
Attention Mathematics
=====================

Scaled dot-product attention is the core operation in Transformers.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

This example demonstrates:
1. Self-attention computation
2. Shape flow through attention
3. Attention weights interpretation
4. Why scaling by 1/sqrt(d_k) matters
5. Masked attention
"""

import numpy as np

from math_for_neural_networks.neural_networks.attention import (
    attention_weights,
    scaled_dot_product_attention,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_self_attention() -> None:
    """Self-attention on a simple sequence."""
    print_section("Example 1: Self-Attention")

    print("""
    Self-attention: Q, K, V all come from the same sequence.

    Sequence: ["I", "love", "cats"]
    Each word has a 4-dimensional embedding.
    """)

    embeddings = np.array([
        [1.0, 0.0, 1.0, 0.0],   # "I"
        [0.0, 2.0, 0.0, 2.0],   # "love"
        [1.0, 1.0, 1.0, 1.0],   # "cats"
    ])

    words = ["I", "love", "cats"]
    d_k = 4

    print("  Embeddings:")
    for word, emb in zip(words, embeddings):
        print(f"    {word:>6}: {emb}")

    output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)

    print(f"\n  Attention weights (rows sum to 1):")
    print(f"  {'':>8}", end="")
    for w in words:
        print(f" | {w:>8}", end="")
    print()
    print("  " + "-" * 40)
    for i, word in enumerate(words):
        print(f"  {word:>8}", end="")
        for j in range(len(words)):
            print(f" | {weights[i,j]:>8.4f}", end="")
        print()

    print(f"\n  Output shape: {output.shape}")
    print(f"  Each word's output is a weighted sum of all word embeddings.")


def example_2_shape_flow() -> None:
    """Demonstrate shape flow through attention."""
    print_section("Example 2: Shape Flow")

    print("""
    Q: (n_q, d_k) = (3, 4)  - 3 queries, 4 dimensions
    K: (n_k, d_k) = (3, 4)  - 3 keys, 4 dimensions
    V: (n_k, d_v) = (3, 2)  - 3 values, 2 dimensions

    QK^T:     (3,4) @ (4,3) = (3,3)  - attention scores
    Scaled:   (3,3) / sqrt(4)
    Weights:  softmax -> (3,3)  - rows sum to 1
    Output:   (3,3) @ (3,2) = (3,2)  - weighted values
    """)

    n_q, n_k, d_k, d_v = 3, 3, 4, 2
    Q = np.random.default_rng(42).standard_normal((n_q, d_k))
    K = np.random.default_rng(43).standard_normal((n_k, d_k))
    V = np.random.default_rng(44).standard_normal((n_k, d_v))

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights row sums: {np.sum(weights, axis=1)}")


def example_3_scaling_intuition() -> None:
    """Why we scale by 1/sqrt(d_k)."""
    print_section("Example 3: Scaling by 1/sqrt(d_k)")

    print("""
    Without scaling: dot products grow with d_k.
    Large dot products push softmax into saturation.

    Demonstration with different d_k values:
    """)

    rng = np.random.default_rng(42)
    d_k_values = [4, 16, 64, 256]

    for d_k in d_k_values:
        q = rng.standard_normal(d_k)
        k = rng.standard_normal(d_k)
        dot = float(np.dot(q, k))
        scaled_dot = dot / np.sqrt(d_k)
        print(f"  d_k={d_k:>3}: dot={dot:>8.3f}, scaled={scaled_dot:>8.3f}")

    print(f"\n  Scaling keeps variance ~1 regardless of d_k.")
    print(f"  This prevents softmax from saturating.")


def example_4_masked_attention() -> None:
    """Masked attention (causal masking)."""
    print_section("Example 4: Masked Attention")

    print("""
    In language models, we mask future positions:
    - Position 0 can only attend to position 0
    - Position 1 can attend to 0 and 1
    - Position 2 can attend to 0, 1, and 2
    """)

    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    mask = np.array([
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ])

    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("  Causal mask:")
    print("         pos0  pos1  pos2")
    for i, row in enumerate(mask):
        print(f"    pos{i}: {[str(v).ljust(5) for v in row]}")

    print(f"\n  Attention weights:")
    for i in range(3):
        print(f"    pos{i}: {weights[i]}")

    print(f"\n  Output: {output}")


if __name__ == "__main__":
    print("ATTENTION MATHEMATICS")
    print("=" * 60)

    example_1_self_attention()
    example_2_shape_flow()
    example_3_scaling_intuition()
    example_4_masked_attention()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Attention(Q,K,V) = softmax(QK^T/sqrt(d_k)) * V
    2. QK^T computes similarity between queries and keys
    3. Scaling by 1/sqrt(d_k) prevents softmax saturation
    4. Softmax makes weights sum to 1 (probability distribution)
    5. Output is weighted sum of values
    6. Masking prevents attending to future positions
    7. This is the core of Transformers, GPT, BERT
    """)
