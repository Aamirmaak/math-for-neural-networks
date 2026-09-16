"""
Manual Backpropagation
======================

This example demonstrates manual backpropagation through a simple neural
network, showing every step of the chain rule.

Architecture:
    x -> z = wx + b -> a = sigmoid(z) -> L = -(y*log(a) + (1-y)*log(1-a))

We derive and compute:
    dL/da, da/dz, dz/dw, dz/db
    dL/dw, dL/db

Then verify against numerical gradients.
"""

import math

import numpy as np

from math_for_neural_networks.autograd.value import Value


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_manual_derivation() -> None:
    """Manually derive and compute gradients for a single neuron."""
    print_section("Example 1: Manual Derivation")

    print("""
    Single neuron:
        z = w*x + b
        a = sigmoid(z)
        L = -(y*log(a) + (1-y)*log(1-a))

    Step 1: dL/da
        dL/da = -y/a + (1-y)/(1-a)

    Step 2: da/dz
        da/dz = a * (1 - a)

    Step 3: dz/dw = x, dz/db = 1

    Step 4: Chain rule
        dL/dw = dL/da * da/dz * dz/dw
        dL/db = dL/da * da/dz * dz/db
    """)

    # Forward pass
    w = 0.5
    x = 2.0
    b = 0.1
    y = 1.0  # target

    z = w * x + b
    a = 1.0 / (1.0 + math.exp(-z))
    L = -(y * math.log(a) + (1 - y) * math.log(1 - a))

    print(f"  Forward pass:")
    print(f"    w = {w}, x = {x}, b = {b}, y = {y}")
    print(f"    z = w*x + b = {z}")
    print(f"    a = sigmoid(z) = {a:.6f}")
    print(f"    L = -(y*log(a) + (1-y)*log(1-a)) = {L:.6f}")

    # Backward pass
    dL_da = -y / a + (1 - y) / (1 - a)
    da_dz = a * (1 - a)
    dz_dw = x
    dz_db = 1.0

    dL_dz = dL_da * da_dz
    dL_dw = dL_dz * dz_dw
    dL_db = dL_dz * dz_db

    print(f"\n  Backward pass:")
    print(f"    dL/da = {dL_da:.6f}")
    print(f"    da/dz = {da_dz:.6f}")
    print(f"    dL/dz = dL/da * da/dz = {dL_dz:.6f}")
    print(f"    dL/dw = dL/dz * x = {dL_dw:.6f}")
    print(f"    dL/db = dL/dz * 1 = {dL_db:.6f}")

    # Verify with autograd
    w_v = Value(w)
    b_v = Value(b)
    z_v = w_v * x + b_v
    a_v = z_v.sigmoid()
    L_v = -(Value(y) * a_v.log() + (1 - y) * (1 - a_v).log())

    L_v.backward()

    print(f"\n  Verification with autograd:")
    print(f"    dL/dw (autograd) = {w_v.grad:.6f}")
    print(f"    dL/db (autograd) = {b_v.grad:.6f}")
    print(f"    dL/dw (manual) = {dL_dw:.6f}")
    print(f"    dL/db (manual) = {dL_db:.6f}")


def example_2_numerical_verification() -> None:
    """Verify manual gradients against numerical gradients."""
    print_section("Example 2: Numerical Verification")

    def compute_loss(w: float, b: float) -> float:
        x = 2.0
        y = 1.0
        z = w * x + b
        a = 1.0 / (1.0 + math.exp(-z))
        a = max(a, 1e-15)  # avoid log(0)
        return -(y * math.log(a) + (1 - y) * math.log(1 - a))

    h = 1e-5
    w, b = 0.5, 0.1

    # Numerical gradients
    dL_dw_num = (compute_loss(w + h, b) - compute_loss(w - h, b)) / (2 * h)
    dL_db_num = (compute_loss(w, b + h) - compute_loss(w, b - h)) / (2 * h)

    # Manual gradients (from example 1)
    x, y = 2.0, 1.0
    z = w * x + b
    a = 1.0 / (1.0 + math.exp(-z))
    dL_dz = a - y
    dL_dw = dL_dz * x
    dL_db = dL_dz

    print(f"  w = {w}, b = {b}")
    print(f"\n  Manual gradients:")
    print(f"    dL/dw = {dL_dw:.10f}")
    print(f"    dL/db = {dL_db:.10f}")
    print(f"\n  Numerical gradients:")
    print(f"    dL/dw = {dL_dw_num:.10f}")
    print(f"    dL/db = {dL_db_num:.10f}")
    print(f"\n  Errors:")
    print(f"    dL/dw error = {abs(dL_dw - dL_dw_num):.2e}")
    print(f"    dL/db error = {abs(dL_db - dL_db_num):.2e}")


def example_3_two_layer() -> None:
    """Manual backpropagation through a two-layer network."""
    print_section("Example 3: Two-Layer Network")

    print("""
    Architecture:
        x -> z1 = w1*x + b1 -> a1 = sigmoid(z1)
        -> z2 = w2*a1 + b2 -> a2 = sigmoid(z2)
        -> L = -(y*log(a2) + (1-y)*log(1-a2))
    """)

    # Forward pass
    x = 2.0
    y = 1.0
    w1, b1 = 0.5, 0.1
    w2, b2 = 0.8, -0.2

    z1 = w1 * x + b1
    a1 = 1.0 / (1.0 + math.exp(-z1))
    z2 = w2 * a1 + b2
    a2 = 1.0 / (1.0 + math.exp(-z2))
    L = -(y * math.log(a2) + (1 - y) * math.log(1 - a2))

    print(f"  Forward pass:")
    print(f"    x = {x}, y = {y}")
    print(f"    Layer 1: w1 = {w1}, b1 = {b1}")
    print(f"    Layer 2: w2 = {w2}, b2 = {b2}")
    print(f"    z1 = {z1:.4f}, a1 = {a1:.6f}")
    print(f"    z2 = {z2:.4f}, a2 = {a2:.6f}")
    print(f"    L = {L:.6f}")

    # Backward pass
    dL_da2 = -y / a2 + (1 - y) / (1 - a2)
    da2_dz2 = a2 * (1 - a2)
    dL_dz2 = dL_da2 * da2_dz2

    dL_dw2 = dL_dz2 * a1
    dL_db2 = dL_dz2

    da1_dz1 = a1 * (1 - a1)
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * da1_dz1

    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1

    print(f"\n  Backward pass:")
    print(f"    Layer 2:")
    print(f"      dL/da2 = {dL_da2:.6f}")
    print(f"      da2/dz2 = {da2_dz2:.6f}")
    print(f"      dL/dz2 = {dL_dz2:.6f}")
    print(f"      dL/dw2 = {dL_dw2:.6f}")
    print(f"      dL/db2 = {dL_db2:.6f}")
    print(f"    Layer 1:")
    print(f"      dL/da1 = {dL_da1:.6f}")
    print(f"      da1/dz1 = {da1_dz1:.6f}")
    print(f"      dL/dz1 = {dL_dz1:.6f}")
    print(f"      dL/dw1 = {dL_dw1:.6f}")
    print(f"      dL/db1 = {dL_db1:.6f}")

    # Verify with autograd
    w1_v = Value(w1)
    b1_v = Value(b1)
    w2_v = Value(w2)
    b2_v = Value(b2)

    z1_v = w1_v * x + b1_v
    a1_v = z1_v.sigmoid()
    z2_v = w2_v * a1_v + b2_v
    a2_v = z2_v.sigmoid()
    L_v = -(Value(y) * a2_v.log() + (1 - y) * (1 - a2_v).log())

    L_v.backward()

    print(f"\n  Verification with autograd:")
    print(f"    dL/dw1: manual={dL_dw1:.6f}, autograd={w1_v.grad:.6f}")
    print(f"    dL/db1: manual={dL_db1:.6f}, autograd={b1_v.grad:.6f}")
    print(f"    dL/dw2: manual={dL_dw2:.6f}, autograd={w2_v.grad:.6f}")
    print(f"    dL/db2: manual={dL_db2:.6f}, autograd={b2_v.grad:.6f}")


def example_4_matrix_backprop() -> None:
    """Demonstrate matrix backpropagation for an affine layer."""
    print_section("Example 4: Matrix Backpropagation")

    print("""
    Affine layer: Z = X @ W^T + b

    For a single sample:
        z = W @ x + b
        dL/dW = outer(dL/dz, x)
        dL/db = dL/dz
        dL/dx = W^T @ dL/dz
    """)

    # Forward
    x = np.array([1.0, 2.0, 3.0])
    W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    b = np.array([0.1, 0.2])

    z = W @ x + b
    print(f"  x = {x}")
    print(f"  W =\n{W}")
    print(f"  b = {b}")
    print(f"  z = W @ x + b = {z}")

    # Upstream gradient
    dout = np.array([1.0, 0.5])
    print(f"  dout = {dout}")

    # Backward
    dW = np.outer(dout, x)
    db = dout
    dx = W.T @ dout

    print(f"\n  Backward pass:")
    print(f"  dW = outer(dout, x) =\n{dW}")
    print(f"  db = dout = {db}")
    print(f"  dx = W^T @ dout = {dx}")

    # Verify with numerical gradients
    def compute_loss_w(W_flat: np.ndarray) -> float:
        W_r = W_flat.reshape(2, 3)
        z_r = W_r @ x + b
        return float(np.sum(z_r ** 2))

    h = 1e-5
    dW_num = np.zeros_like(W)
    for i in range(W.size):
        W_plus = W.copy().ravel()
        W_minus = W.copy().ravel()
        W_plus[i] += h
        W_minus[i] -= h
        dW_num.ravel()[i] = (compute_loss_w(W_plus) - compute_loss_w(W_minus)) / (2 * h)

    # Upstream gradient for loss = sum(z^2) is 2*z
    dout_loss = 2 * z
    dW_expected = np.outer(dout_loss, x)

    print(f"\n  Verification:")
    print(f"  dW (backward) =\n{dW}")
    print(f"  dW (expected) =\n{dW_expected}")


if __name__ == "__main__":
    print("MANUAL BACKPROPAGATION")
    print("=" * 60)

    example_1_manual_derivation()
    example_2_numerical_verification()
    example_3_two_layer()
    example_4_matrix_backprop()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Backpropagation applies the chain rule in reverse order
    2. Each operation contributes a local derivative
    3. Global derivative = product of local derivatives along the path
    4. For matrices: transposes appear to match shapes
    5. Numerical verification confirms analytical gradients
    6. Two-layer network: gradients flow backward through both layers
    """)
