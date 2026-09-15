"""
Projection and Linear Transformations Example — Math for Neural Networks

This example demonstrates vector projection and linear transformations,
connecting them to neural network concepts.
"""

import numpy as np

from math_for_neural_networks.linear_algebra import (
    apply_transformation,
    dot_product,
    project_vector,
    orthogonal_component,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    verify_projection,
)


def main() -> None:
    print("=" * 60)
    print("PROJECTION & LINEAR TRANSFORMATIONS")
    print("=" * 60)

    # --- Vector Projection ---
    print("\n--- Vector Projection ---")
    a = np.array([3.0, 4.0])
    b = np.array([1.0, 0.0])

    proj = project_vector(a, b)
    orth = orthogonal_component(a, b)
    verification = verify_projection(a, b)

    print(f"Vector a = {a}")
    print(f"Direction b = {b}")
    print(f"\nProjection of a onto b: {proj}")
    print(f"Orthogonal component:  {orth}")
    print(f"\nReconstruction: proj + orth = {proj + orth}")
    print(f"Original a = {a}")
    print(f"Reconstruction error: {verification['reconstruction_error']:.2e}")
    print(f"Orthogonality check (should be 0): {verification['orthogonality_check']:.2e}")

    print("\nNeural network: Projection decomposes representations")
    print("  - PCA uses projections onto principal components")
    print("  - Understanding how much of one representation lies along another")

    # --- 45-degree projection ---
    print("\n--- 45-degree Projection ---")
    a2 = np.array([1.0, 1.0])
    b2 = np.array([1.0, 0.0])
    proj2 = project_vector(a2, b2)
    orth2 = orthogonal_component(a2, b2)
    print(f"a = {a2}, b = {b2}")
    print(f"proj_b(a) = {proj2}")
    print(f"orth_b(a) = {orth2}")
    print(f"a = proj + orth = {proj2 + orth2}")

    # --- Linear Transformations ---
    print("\n" + "=" * 60)
    print("LINEAR TRANSFORMATIONS")
    print("=" * 60)

    v = np.array([1.0, 0.0])

    # Scaling
    print("\n--- Scaling ---")
    M_scale = scaling_matrix(2.0, 3.0)
    v_scaled = apply_transformation(M_scale, v)
    print(f"Original: {v}")
    print(f"Scaling matrix:\n{M_scale}")
    print(f"Scaled: {v_scaled}")
    print("Neural network: Scaling activations by different amounts per dimension")

    # Rotation
    print("\n--- Rotation (90 degrees) ---")
    M_rot = rotation_matrix(np.pi / 2)
    v_rotated = apply_transformation(M_rot, v)
    print(f"Original: {v}")
    print(f"Rotation matrix:\n{M_rot}")
    print(f"Rotated: {v_rotated}")
    print("Neural network: Rotation preserves norm (length of representation)")

    # Verify rotation preserves norm
    print(f"\nOriginal norm: {np.linalg.norm(v):.4f}")
    print(f"Rotated norm:  {np.linalg.norm(v_rotated):.4f}")

    # Shear
    print("\n--- Shear ---")
    M_shear = shear_matrix(0.5, 0.0)
    v_sheared = apply_transformation(M_shear, np.array([1.0, 1.0]))
    print(f"Original: [1, 1]")
    print(f"Shear matrix:\n{M_shear}")
    print(f"Sheared: {v_sheared}")
    print("Neural network: Non-linear activations warp representation space")

    # --- Neural Network Connection ---
    print("\n" + "=" * 60)
    print("NEURAL NETWORK CONNECTION")
    print("=" * 60)
    print("""
A neural network linear layer IS a linear transformation:
    y = Wx + b

Where:
    W = weight matrix (transformation)
    x = input vector
    b = bias vector (shift)

Without non-linear activations, stacking linear layers
is just one big linear transformation:
    W2 @ (W1 @ x) + b2 = (W2 @ W1) @ x + (b2 + W2 @ b1)

This is why non-linear activations (ReLU, sigmoid, etc.) are essential!
    """)


if __name__ == "__main__":
    main()
