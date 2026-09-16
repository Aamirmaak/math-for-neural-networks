"""
Micro Autograd Demo
===================

Demonstrates the educational micro autograd engine that implements
reverse-mode automatic differentiation on scalar values.

Features:
1. Automatic graph construction
2. Forward and backward passes
3. Gradient accumulation
4. Complex expressions
"""

import math

from math_for_neural_networks.autograd.value import Value


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_basic() -> None:
    """Basic autograd operations."""
    print_section("Example 1: Basic Operations")

    a = Value(2.0)
    b = Value(3.0)

    # Build graph
    c = a + b
    d = a * b
    e = c * d

    print(f"  a = {a.data}, b = {b.data}")
    print(f"  c = a + b = {c.data}")
    print(f"  d = a * b = {d.data}")
    print(f"  e = c * d = {e.data}")

    # Backward
    e.backward()

    # e = (a+b) * (a*b) = a^2*b + a*b^2
    # de/da = 2ab + b^2 = 2*2*3 + 9 = 21
    # de/db = a^2 + 2ab = 4 + 12 = 16
    print(f"\n  Gradients:")
    print(f"  de/da = {a.grad} (expected: 2ab + b^2 = {2*a.data*b.data + b.data**2})")
    print(f"  de/db = {b.grad} (expected: a^2 + 2ab = {a.data**2 + 2*a.data*b.data})")


def example_2_neuron() -> None:
    """Simulate a single neuron with autograd."""
    print_section("Example 2: Single Neuron")

    print("""
    Neuron: y = sigmoid(w1*x1 + w2*x2 + b)
    """)

    w1 = Value(0.5)
    w2 = Value(-0.3)
    b = Value(0.1)
    x1 = Value(1.0)
    x2 = Value(2.0)

    # Forward
    z = w1 * x1 + w2 * x2 + b
    y = z.sigmoid()

    print(f"  w1 = {w1.data}, w2 = {w2.data}, b = {b.data}")
    print(f"  x1 = {x1.data}, x2 = {x2.data}")
    print(f"  z = w1*x1 + w2*x2 + b = {z.data}")
    print(f"  y = sigmoid(z) = {y.data:.6f}")

    # Backward
    y.backward()

    # Verify
    s = y.data
    print(f"\n  Gradients:")
    print(f"  dy/dw1 = {w1.grad:.6f} (expected: x1 * s * (1-s) = {x1.data * s * (1-s):.6f})")
    print(f"  dy/dw2 = {w2.grad:.6f} (expected: x2 * s * (1-s) = {x2.data * s * (1-s):.6f})")
    print(f"  dy/db = {b.grad:.6f} (expected: s * (1-s) = {s * (1-s):.6f})")


def example_3_gradient_accumulation() -> None:
    """Demonstrate gradient accumulation with multiple paths."""
    print_section("Example 3: Gradient Accumulation")

    print("""
    f = x * x * x

    Three paths from x to f:
        Path 1: x -> (first mul) -> x^2 -> (second mul) -> x^3
        Path 2: x -> (second mul) -> x^3
        Path 3: x -> (second mul) -> x^3

    df/dx = 3x^2
    """)

    x = Value(2.0)
    f = x * x * x

    print(f"  x = {x.data}")
    print(f"  f = x^3 = {f.data}")

    f.backward()

    expected = 3 * x.data**2
    print(f"\n  Gradient:")
    print(f"  df/dx = {x.grad} (expected: 3x^2 = {expected})")
    print(f"  Gradient accumulated from all three paths!")


def example_4_deep_network() -> None:
    """Simulate a deeper network with autograd."""
    print_section("Example 4: Deep Network Simulation")

    print("""
    Network: x -> (w1*x + b1) -> tanh -> (w2*? + b2) -> sigmoid -> y

    With loss: L = (y - target)^2
    """)

    # Parameters
    w1 = Value(0.5)
    b1 = Value(0.1)
    w2 = Value(-0.8)
    b2 = Value(0.2)
    target = 1.0
    x_val = 1.5

    # Forward
    x = Value(x_val)
    z1 = w1 * x + b1
    a1 = z1.tanh()
    z2 = w2 * a1 + b2
    y = z2.sigmoid()
    loss = (y - Value(target)) ** 2

    print(f"  Parameters: w1={w1.data}, b1={b1.data}, w2={w2.data}, b2={b2.data}")
    print(f"  Input: x = {x_val}, Target: {target}")
    print(f"\n  Forward pass:")
    print(f"    z1 = w1*x + b1 = {z1.data:.4f}")
    print(f"    a1 = tanh(z1) = {a1.data:.4f}")
    print(f"    z2 = w2*a1 + b2 = {z2.data:.4f}")
    print(f"    y = sigmoid(z2) = {y.data:.6f}")
    print(f"    L = (y - target)^2 = {loss.data:.6f}")

    # Backward
    loss.backward()

    print(f"\n  Backward pass (gradients):")
    print(f"    dL/dy = {2 * (y.data - target):.6f}")
    print(f"    dL/dw1 = {w1.grad:.6f}")
    print(f"    dL/db1 = {b1.grad:.6f}")
    print(f"    dL/dw2 = {w2.grad:.6f}")
    print(f"    dL/db2 = {b2.grad:.6f}")

    # Gradient descent step
    lr = 0.1
    w1_new = w1.data - lr * w1.grad
    b1_new = b1.data - lr * b1.grad
    w2_new = w2.data - lr * w2.grad
    b2_new = b2.data - lr * b2.grad

    print(f"\n  After one gradient descent step (lr={lr}):")
    print(f"    w1: {w1.data:.4f} -> {w1_new:.4f}")
    print(f"    b1: {b1.data:.4f} -> {b1_new:.4f}")
    print(f"    w2: {w2.data:.4f} -> {w2_new:.4f}")
    print(f"    b2: {b2.data:.4f} -> {b2_new:.4f}")


def example_5_loss_landscape() -> None:
    """Visualize loss landscape for a simple function."""
    print_section("Example 5: Loss Landscape")

    print("""
    Loss function: L(w) = (sigmoid(w * x + b) - y)^2

    Varying w and computing loss at each point.
    """)

    x_val = 2.0
    b_val = 0.0
    y_val = 1.0

    print(f"  x = {x_val}, b = {b_val}, y = {y_val}")
    print(f"\n  {'w':>6} | {'L(w)':>10} | {'dL/dw':>10}")
    print("  " + "-" * 30)

    for w_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        w = Value(w_val)
        b = Value(b_val)
        z = w * x_val + b
        pred = z.sigmoid()
        loss = (pred - Value(y_val)) ** 2
        loss.backward()

        print(f"  {w_val:>6.1f} | {loss.data:>10.6f} | {w.grad:>10.6f}")

    print(f"\n  The gradient points toward decreasing loss.")
    print(f"  Gradient descent follows the gradient to minimize loss.")


if __name__ == "__main__":
    print("MICRO AUTOGRAD DEMO")
    print("=" * 60)

    example_1_basic()
    example_2_neuron()
    example_3_gradient_accumulation()
    example_4_deep_network()
    example_5_loss_landscape()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Autograd automatically builds computational graphs
    2. Backward pass computes all gradients in one traversal
    3. Gradients accumulate across multiple paths
    4. Supports basic arithmetic and activation functions
    5. Gradient descent uses gradients to update parameters
    6. Loss landscape shows how gradients guide optimization
    """)
