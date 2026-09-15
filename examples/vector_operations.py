"""
Vector Operations Example — Math for Neural Networks

This example demonstrates fundamental vector operations and their
connection to neural networks.

Neural network connections:
- Vectors = feature representations, embeddings
- Dot product = weighted sum in a neuron
- Cosine similarity = embedding similarity
"""

import numpy as np

from math_for_neural_networks.linear_algebra import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l1_norm,
    l2_norm,
    scalar_multiply,
    vector_add,
    vector_subtract,
)


def main() -> None:
    print("=" * 60)
    print("VECTOR OPERATIONS — Neural Network Foundations")
    print("=" * 60)

    # --- Basic Vector Operations ---
    print("\n--- Vector Creation ---")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print(f"a = {a}")
    print(f"b = {b}")

    print("\n--- Vector Addition ---")
    # Neural network: residual connection h = f(x) + x
    c = vector_add(a, b)
    print(f"a + b = {c}")
    print("Neural network: Residual connection adds skip vector to main path")

    print("\n--- Vector Subtraction ---")
    d = vector_subtract(a, b)
    print(f"a - b = {d}")

    print("\n--- Scalar Multiplication ---")
    # Neural network: learning rate scales gradient
    scaled = scalar_multiply(2.0, a)
    print(f"2 * a = {scaled}")
    print("Neural network: Gradient update w = w - lr * grad")

    # --- Dot Product ---
    print("\n--- Dot Product ---")
    dp = dot_product(a, b)
    print(f"a · b = {dp}")
    print(f"Manual check: {1*4 + 2*5 + 3*6}")
    print("Neural network: Neuron computes z = w · x + bias")
    print("Neural network: Attention score = q · k")

    # Commutativity
    print(f"\nCommutativity check: a·b = {dot_product(a, b)}, b·a = {dot_product(b, a)}")

    # --- Norms ---
    print("\n--- L1 Norm (Manhattan) ---")
    l1 = l1_norm(a)
    print(f"||a||_1 = {l1}")
    print("Neural network: L1 regularization encourages weight sparsity")

    print("\n--- L2 Norm (Euclidean) ---")
    l2 = l2_norm(a)
    print(f"||a||_2 = {l2}")
    print(f"Verification: sqrt(1^2 + 2^2 + 3^2) = {np.sqrt(1 + 4 + 9)}")
    print("Neural network: L2 regularization (weight decay)")
    print("Neural network: Gradient clipping by norm")

    # --- Distance ---
    print("\n--- Euclidean Distance ---")
    dist = euclidean_distance(a, b)
    print(f"d(a, b) = {dist}")
    print("Neural network: Distance in representation space")

    # --- Cosine Similarity ---
    print("\n--- Cosine Similarity ---")
    sim = cosine_similarity(a, b)
    print(f"cos(a, b) = {sim:.6f}")
    print(f"Angle: {np.arccos(sim) * 180 / np.pi:.1f} degrees")

    # Same direction
    a2 = np.array([2.0, 4.0, 6.0])  # 2 * a
    print(f"\ncos(a, 2a) = {cosine_similarity(a, a2):.6f} (same direction)")

    # Orthogonal
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    print(f"cos(e1, e2) = {cosine_similarity(e1, e2):.6f} (orthogonal)")

    print("\nNeural network: Cosine similarity measures semantic similarity")
    print("  - Word/sentence embeddings")
    print("  - Retrieval: find similar documents")
    print("  - Contrastive learning objectives")

    # --- Neural Network Example ---
    print("\n" + "=" * 60)
    print("NEURAL NETWORK EXAMPLE: Single Neuron")
    print("=" * 60)

    # Input features
    x = np.array([1.0, 2.0, 3.0])
    # Weights
    w = np.array([0.5, -0.3, 0.8])
    # Bias
    bias = 0.1

    # Forward pass: z = w · x + b
    z = dot_product(w, x) + bias
    print(f"\nInput:  x = {x}")
    print(f"Weights: w = {w}")
    print(f"Bias:   b = {bias}")
    print(f"\nNeuron output: z = w · x + b = {z:.4f}")
    print("This single dot product + bias IS a neural network layer!")


if __name__ == "__main__":
    main()
