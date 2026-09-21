"""
From Scratch vs PyTorch: Side-by-Side Comparison
==================================================

This educational example demonstrates the same neural network mathematics
implemented both:

1. From scratch (using this toolkit's NumPy implementations)
2. Using PyTorch (the framework that abstracts these operations)

The goal is to understand what PyTorch abstracts away, not to replace
our implementations.

Educational flow:
    1. Define the mathematics
    2. Implement with our toolkit
    3. Implement equivalent PyTorch code
    4. Compare forward outputs, loss, and gradients
    5. Explain what PyTorch abstracts

Requires: pip install torch
"""

import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch not installed. Run: pip install torch")
    raise SystemExit(1)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    """Run the comparison."""

    # ================================================================
    # SETUP: Same input and parameters for both implementations
    # ================================================================
    print_section("SETUP: Defining the Mathematics")

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((4, 3))
    y_np = rng.standard_normal((4, 2))

    # Our parameters (NumPy arrays)
    W1_np = rng.standard_normal((4, 3)) * 0.5
    b1_np = np.zeros(4)
    W2_np = rng.standard_normal((2, 4)) * 0.5
    b2_np = np.zeros(2)

    print("Input shape:", x_np.shape)
    print("Target shape:", y_np.shape)
    print("W1 shape:", W1_np.shape)
    print("W2 shape:", W2_np.shape)

    # ================================================================
    # STEP 1: FORWARD PASS
    # ================================================================
    print_section("STEP 1: Forward Pass")
    print("Mathematics: z1 = x @ W1^T + b1")
    print("             a1 = sigmoid(z1)")
    print("             z2 = a1 @ W2^T + b2")

    # --- From Scratch ---
    print("\n--- From Scratch (NumPy) ---")
    z1_ours = x_np @ W1_np.T + b1_np
    a1_ours = 1.0 / (1.0 + np.exp(-z1_ours))
    z2_ours = a1_ours @ W2_np.T + b2_np
    print(f"  z1 shape: {z1_ours.shape}")
    print(f"  a1 shape: {a1_ours.shape}")
    print(f"  z2 shape: {z2_ours.shape}")

    # --- PyTorch ---
    print("\n--- PyTorch ---")
    W1_t = torch.tensor(W1_np, dtype=torch.float64)
    b1_t = torch.tensor(b1_np, dtype=torch.float64)
    W2_t = torch.tensor(W2_np, dtype=torch.float64)
    b2_t = torch.tensor(b2_np, dtype=torch.float64)
    x_t = torch.tensor(x_np, dtype=torch.float64)

    z1_t = x_t @ W1_t.T + b1_t
    a1_t = torch.sigmoid(z1_t)
    z2_t = a1_t @ W2_t.T + b2_t
    print(f"  z1 shape: {z1_t.shape}")
    print(f"  a1 shape: {a1_t.shape}")
    print(f"  z2 shape: {z2_t.shape}")

    # Compare
    max_err = float(np.max(np.abs(z2_ours - z2_t.numpy())))
    print(f"\n  Forward output max error: {max_err:.2e}")

    # ================================================================
    # STEP 2: LOSS
    # ================================================================
    print_section("STEP 2: Loss Computation")
    print("Mathematics: L = (1/n) * sum((y_true - y_pred)^2)")

    # --- From Scratch ---
    print("\n--- From Scratch ---")
    n = y_np.size
    loss_ours = float(np.mean((y_np - z2_ours) ** 2))
    print(f"  MSE loss: {loss_ours:.6f}")

    # --- PyTorch ---
    print("\n--- PyTorch ---")
    y_t = torch.tensor(y_np, dtype=torch.float64)
    loss_t = torch.mean((z2_t - y_t) ** 2)
    print(f"  MSE loss: {loss_t.item():.6f}")

    # ================================================================
    # STEP 3: BACKWARD PASS (GRADIENTS)
    # ================================================================
    print_section("STEP 3: Backward Pass (Gradients)")
    print("Mathematics: dL/dz2 = (2/n) * (z2 - y)")
    print("             dL/dW2 = dL/dz2^T @ a1")
    print("             dL/db2 = sum(dL/dz2)")
    print("             dL/da1 = dL/dz2 @ W2")
    print("             dL/dz1 = dL/da1 * sigmoid'(z1)")
    print("             dL/dW1 = dL/dz1^T @ x")
    print("             dL/db1 = sum(dL/dz1)")

    # --- From Scratch ---
    print("\n--- From Scratch ---")
    dout = (2.0 / y_np.size) * (z2_ours - y_np)
    dW2_ours = dout.T @ a1_ours
    db2_ours = np.sum(dout, axis=0)
    da1_ours = dout @ W2_np
    dz1_ours = da1_ours * a1_ours * (1.0 - a1_ours)
    dW1_ours = dz1_ours.T @ x_np
    db1_ours = np.sum(dz1_ours, axis=0)

    print(f"  dW2 shape: {dW2_ours.shape}, norm: {np.linalg.norm(dW2_ours):.6f}")
    print(f"  db2 shape: {db2_ours.shape}, norm: {np.linalg.norm(db2_ours):.6f}")
    print(f"  dW1 shape: {dW1_ours.shape}, norm: {np.linalg.norm(dW1_ours):.6f}")
    print(f"  db1 shape: {db1_ours.shape}, norm: {np.linalg.norm(db1_ours):.6f}")

    # --- PyTorch ---
    print("\n--- PyTorch (autograd) ---")
    W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

    z1_t = x_t @ W1_t.T + b1_t
    a1_t = torch.sigmoid(z1_t)
    z2_t = a1_t @ W2_t.T + b2_t
    loss_t = torch.mean((z2_t - y_t) ** 2)
    loss_t.backward()

    print(f"  dW2 shape: {W2_t.grad.shape}, norm: {torch.norm(W2_t.grad).item():.6f}")
    print(f"  db2 shape: {b2_t.grad.shape}, norm: {torch.norm(b2_t.grad).item():.6f}")
    print(f"  dW1 shape: {W1_t.grad.shape}, norm: {torch.norm(W1_t.grad).item():.6f}")
    print(f"  db1 shape: {b1_t.grad.shape}, norm: {torch.norm(b1_t.grad).item():.6f}")

    # Compare gradients
    print("\n--- Gradient Comparison ---")
    for name, ours, pt in [
        ("dW2", dW2_ours, W2_t.grad.numpy()),
        ("db2", db2_ours, b2_t.grad.numpy()),
        ("dW1", dW1_ours, W1_t.grad.numpy()),
        ("db1", db1_ours, b1_t.grad.numpy()),
    ]:
        err = float(np.max(np.abs(ours - pt)))
        print(f"  {name} max error: {err:.2e}")

    # ================================================================
    # STEP 4: PARAMETER UPDATE
    # ================================================================
    print_section("STEP 4: Parameter Update")
    print("Mathematics: theta = theta - lr * grad")

    lr = 0.01
    print(f"  Learning rate: {lr}")

    # --- From Scratch ---
    print("\n--- From Scratch ---")
    W1_new_ours = W1_np - lr * dW1_ours
    b1_new_ours = b1_np - lr * db1_ours
    W2_new_ours = W2_np - lr * dW2_ours
    b2_new_ours = b2_np - lr * db2_ours

    # --- PyTorch ---
    print("\n--- PyTorch ---")
    W1_new_t = (W1_t - lr * W1_t.grad).detach().numpy()
    b1_new_t = (b1_t - lr * b1_t.grad).detach().numpy()
    W2_new_t = (W2_t - lr * W2_t.grad).detach().numpy()
    b2_new_t = (b2_t - lr * b2_t.grad).detach().numpy()

    # Compare updated parameters
    print("\n--- Parameter Update Comparison ---")
    for name, ours, pt in [
        ("W1", W1_new_ours, W1_new_t),
        ("b1", b1_new_ours, b1_new_t),
        ("W2", W2_new_ours, W2_new_t),
        ("b2", b2_new_ours, b2_new_t),
    ]:
        err = float(np.max(np.abs(ours - pt)))
        print(f"  {name} max error: {err:.2e}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print_section("SUMMARY")
    print("""
What we implemented manually:
    - Matrix multiplication: Z = X @ W^T + b
    - Sigmoid: sigma(x) = 1 / (1 + exp(-x))
    - MSE loss: L = mean((y - y_pred)^2)
    - Backward pass: chain rule through each operation
    - Gradient update: theta = theta - lr * grad

What PyTorch abstracts away:
    - Automatic differentiation (autograd)
    - Memory management for computational graphs
    - Gradient accumulation
    - Device placement (CPU/GPU)
    - Mixed precision training
    - Distributed training
    - Optimization algorithms (Adam, SGD, etc.)

The key insight:
    PyTorch's torch.autograd computes the SAME gradients we computed
    manually. It does this by building a computational graph and
    applying the chain rule in reverse (backpropagation).

    The mathematics is identical - PyTorch just automates it.
""")


if __name__ == "__main__":
    main()
