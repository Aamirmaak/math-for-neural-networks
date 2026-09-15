"""
Matrix Operations Example — Math for Neural Networks

This example demonstrates matrix operations and their neural network
connections: linear layers, transformations, and batch processing.
"""

import numpy as np

from math_for_neural_networks.linear_algebra import (
    identity_matrix,
    matrix_add,
    matrix_multiply,
    matrix_subtract,
    matrix_vector_multiply,
    scalar_multiply_matrix,
    transpose,
)


def main() -> None:
    print("=" * 60)
    print("MATRIX OPERATIONS — Neural Network Layers")
    print("=" * 60)

    # --- Matrix Creation ---
    print("\n--- Matrix as Weight Matrix ---")
    W = np.array([[0.5, -0.3, 0.8],
                  [0.2,  0.7, -0.1]])
    print(f"Weight matrix W (2 neurons, 3 inputs):")
    print(W)
    print(f"Shape: {W.shape}")

    # --- Matrix-Vector Multiplication (Single Neuron Layer) ---
    print("\n--- Matrix-Vector Multiply (Linear Layer) ---")
    x = np.array([1.0, 2.0, 3.0])
    y = matrix_vector_multiply(W, x)
    print(f"Input:  x = {x}")
    print(f"Output: y = Wx = {y}")
    print(f"\nNeural network: y = Wx + b IS a linear layer!")
    print(f"Each row of W is one neuron's weight vector")
    print(f"Neuron 0: dot([0.5, -0.3, 0.8], [1, 2, 3]) = {y[0]}")
    print(f"Neuron 1: dot([0.2, 0.7, -0.1], [1, 2, 3]) = {y[1]}")

    # --- Matrix-Matrix Multiplication (Batch Processing) ---
    print("\n--- Matrix-Matrix Multiply (Batch Processing) ---")
    X_batch = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    Y_batch = matrix_multiply(X_batch, W.T)
    print(f"Batch of 3 inputs (each row is a sample):")
    print(X_batch)
    print(f"\nAll outputs at once: Y = X @ W^T")
    print(Y_batch)
    print("Neural network: Batch processing = parallel matrix multiply!")

    # --- Transpose ---
    print("\n--- Transpose ---")
    print(f"W shape: {W.shape}")
    print(f"W^T shape: {transpose(W).shape}")
    print("Neural network: Backpropagation requires W^T")

    # --- Matrix Addition ---
    print("\n--- Matrix Addition ---")
    W2 = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6]])
    W_sum = matrix_add(W, W2)
    print(f"W + W2 = {W_sum}")
    print("Neural network: Combining two weight matrices")

    # --- Scalar Multiplication ---
    print("\n--- Scalar Multiplication ---")
    W_scaled = scalar_multiply_matrix(0.01, W)
    print(f"0.01 * W = {W_scaled}")
    print("Neural network: Small learning rate scales weight updates")

    # --- Identity Matrix ---
    print("\n--- Identity Matrix ---")
    I = identity_matrix(3)
    print(f"I_3 =")
    print(I)
    result = matrix_multiply(W, I)
    print(f"\nW @ I = W? {np.allclose(result, W)}")
    print("Neural network: Identity preserves information (skip connection base)")

    # --- Non-Commutativity ---
    print("\n--- Matrix Multiplication is NOT Commutative ---")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    AB = matrix_multiply(A, B)
    BA = matrix_multiply(B, A)
    print(f"A @ B =\n{AB}")
    print(f"\nB @ A =\n{BA}")
    print(f"\nAB == BA? {np.allclose(AB, BA)}")
    print("Neural network: Layer order matters! f(g(x)) != g(f(x))")

    # --- Full Neural Network Layer ---
    print("\n" + "=" * 60)
    print("FULL LINEAR LAYER: y = Wx + b")
    print("=" * 60)

    W_full = np.array([[0.5, -0.3],
                        [0.2,  0.7]])
    b = np.array([0.1, -0.2])
    x_full = np.array([1.0, 2.0])

    z = matrix_vector_multiply(W_full, x_full) + b
    print(f"W = {W_full.tolist()}")
    print(f"b = {b}")
    print(f"x = {x_full}")
    print(f"\ny = Wx + b = {z}")
    print("\nThis is the core operation of every neural network dense layer!")


if __name__ == "__main__":
    main()
