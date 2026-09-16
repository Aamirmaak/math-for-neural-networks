"""
Computational Graph
===================

A computational graph represents mathematical expressions as a directed
acyclic graph (DAG) where:
- Nodes are variables (values)
- Edges are operations

This example demonstrates:
1. Building a computational graph
2. Forward pass (evaluation)
3. Backward pass (gradient computation)
4. How the chain rule applies to graphs
"""

import math

from math_for_neural_networks.autograd.value import Value, get_topo_order


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_simple_graph() -> None:
    """Build and evaluate a simple computational graph."""
    print_section("Example 1: Simple Graph")

    print("""
    Expression: f = (a + b) * c

    Graph:
        a --\
             + --> sum --> multiply --> f
        b --/               ^
                            c
    """)

    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)

    sum_node = a + b
    f = sum_node * c

    print(f"  a = {a.data}")
    print(f"  b = {b.data}")
    print(f"  c = {c.data}")
    print(f"  f = (a + b) * c = {f.data}")

    # Backward pass
    f.backward()

    print(f"\n  After backward():")
    print(f"  df/da = {a.grad} (should be c = {c.data})")
    print(f"  df/db = {b.grad} (should be c = {c.data})")
    print(f"  df/dc = {c.grad} (should be a + b = {a.data + b.data})")


def example_2_chain_rule() -> None:
    """Demonstrate chain rule in a computational graph."""
    print_section("Example 2: Chain Rule")

    print("""
    Expression: f = (a * b)^2

    Chain rule:
        df/da = df/dz * dz/da
        where z = a * b

        df/dz = 2z = 2ab
        dz/db = a

        df/db = 2ab * a = 2a^2 * b
    """)

    a = Value(3.0)
    b = Value(2.0)
    z = a * b
    f = z**2

    print(f"  a = {a.data}")
    print(f"  b = {b.data}")
    print(f"  z = a * b = {z.data}")
    print(f"  f = z^2 = {f.data}")

    f.backward()

    print(f"\n  After backward():")
    print(f"  df/da = {a.grad} (should be 2*a*b^2 = {2 * a.data * b.data**2})")
    print(f"  df/db = {b.grad} (should be 2*a^2*b = {2 * a.data**2 * b.data})")


def example_3_activation() -> None:
    """Demonstrate computational graph with activation function."""
    print_section("Example 3: Activation Function")

    print("""
    Expression: f = sigmoid(a * b + c)

    Graph:
        a --\
             * --> z1 --\
        b --/             + --> z2 --> sigmoid --> f
                         c
    """)

    a = Value(0.5)
    b = Value(1.0)
    c = Value(0.1)

    z1 = a * b
    z2 = z1 + c
    f = z2.sigmoid()

    print(f"  a = {a.data}")
    print(f"  b = {b.data}")
    print(f"  c = {c.data}")
    print(f"  z1 = a * b = {z1.data}")
    print(f"  z2 = z1 + c = {z2.data}")
    print(f"  f = sigmoid(z2) = {f.data:.6f}")

    f.backward()

    # Verify manually
    s = 1.0 / (1.0 + math.exp(-z2.data))
    expected_dz2 = s * (1.0 - s)

    print(f"\n  After backward():")
    print(f"  df/dc = {c.grad:.6f} (should be sigmoid'(z2) = {expected_dz2:.6f})")
    print(f"  df/db = {b.grad:.6f} (should be a * sigmoid'(z2) = {a.data * expected_dz2:.6f})")
    print(f"  df/da = {a.grad:.6f} (should be b * sigmoid'(z2) = {b.data * expected_dz2:.6f})")


def example_4_multiple_paths() -> None:
    """Demonstrate gradient accumulation with multiple paths."""
    print_section("Example 4: Multiple Paths (Gradient Accumulation)")

    print("""
    Expression: f = a * a + a

    Two paths from 'a' to 'f':
        Path 1: a -> multiply -> f (gradient: a)
        Path 2: a -> add -> f (gradient: 1)

    Total: df/da = 2a + 1
    """)

    a = Value(3.0)
    f = a * a + a

    print(f"  a = {a.data}")
    print(f"  f = a*a + a = {f.data}")

    f.backward()

    expected = 2 * a.data + 1
    print(f"\n  After backward():")
    print(f"  df/da = {a.grad} (should be 2a + 1 = {expected})")
    print(f"  Gradient accumulated from both paths!")


def example_5_topological_sort() -> None:
    """Demonstrate topological ordering of the graph."""
    print_section("Example 5: Topological Sort")

    print("""
    Topological sort ensures parents are processed before children.
    Reverse topological sort is used for backpropagation.
    """)

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = c + a
    e = d**2

    topo = get_topo_order(e)

    print("  Forward pass order (topological):")
    for i, node in enumerate(topo):
        parents = ", ".join(
            f"Value({p.data})" for p in node._prev
        ) if node._prev else "none"
        print(f"    {i}: {node._op or 'leaf':>10} -> {node.data:.2f}  (parents: {parents})")

    print(f"\n  Backward pass order (reverse topological):")
    for i, node in enumerate(reversed(topo)):
        print(f"    {i}: {node._op or 'leaf':>10} -> grad={node.grad:.6f}")


if __name__ == "__main__":
    print("COMPUTATIONAL GRAPH")
    print("=" * 60)

    example_1_simple_graph()
    example_2_chain_rule()
    example_3_activation()
    example_4_multiple_paths()
    example_5_topological_sort()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Computational graphs represent expressions as DAGs
    2. Forward pass evaluates the expression
    3. Backward pass computes gradients using the chain rule
    4. Each node stores its local derivative
    5. Gradients accumulate across multiple paths
    6. Topological sort ensures correct processing order
    7. Reverse-mode differentiation (backprop) computes all gradients efficiently
    """)
