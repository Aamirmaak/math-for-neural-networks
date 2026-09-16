"""
Chain Rule — The Engine of Backpropagation
===========================================

The chain rule lets us compute derivatives of composed functions.
It is the mathematical foundation of backpropagation in neural networks.

This example demonstrates:
1. Chain rule for simple compositions
2. Chain rule for multivariable functions
3. How backpropagation applies the chain rule
4. Why the chain rule is essential for deep learning
"""

import numpy as np

from math_for_neural_networks.calculus.chain_rule import (
    chain_rule_multi,
    chain_rule_scalar,
    neural_network_chain_rule_demo,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_basic_chain_rule() -> None:
    """Chain rule for f(g(x))."""
    print_section("Example 1: Basic Chain Rule")

    print("""
    Chain Rule: If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x)

    Example: y = sin(x²)
    - Outer function: f(u) = sin(u), f'(u) = cos(u)
    - Inner function: g(x) = x², g'(x) = 2x
    - Chain rule: dy/dx = cos(x²) · 2x

    At x = 2:
    - g(2) = 4
    - f'(g(2)) = cos(4) ≈ -0.6536
    - g'(2) = 4
    - dy/dx = cos(4) × 4 ≈ -2.6146
    """)

    import math

    def outer(g: float) -> float:
        return math.sin(g)

    def outer_d(g: float) -> float:
        return math.cos(g)

    def inner(x: float) -> float:
        return x**2

    def inner_d(x: float) -> float:
        return 2 * x

    x_val = 2.0
    result = chain_rule_scalar(outer, outer_d, inner, inner_d, x_val)
    expected = math.cos(x_val**2) * 2 * x_val

    print(f"  Chain rule result: {result:.6f}")
    print(f"  Expected (cos(x²)·2x): {expected:.6f}")
    print(f"  Match: {abs(result - expected) < 1e-10}")


def example_2_multivariable_chain_rule() -> None:
    """Chain rule with multiple intermediate variables."""
    print_section("Example 2: Multivariable Chain Rule")

    print("""
    When a function depends on multiple intermediate variables:

    y = f(u₁, u₂), where u₁ = g₁(x), u₂ = g₂(x)

    dy/dx = (∂f/∂u₁)(du₁/dx) + (∂f/∂u₂)(du₂/dx)

    This is the SUM over all paths from y to x.
    """)

    # Example: y = u₁² + u₂², u₁ = 2x, u₂ = x+1
    # dy/dx = 2u₁·2 + 2u₂·1 = 4u₁ + 2u₂
    x_val = 3.0
    u1 = 2 * x_val  # = 6
    u2 = x_val + 1  # = 4

    outer_derivs = np.array([2 * u1, 2 * u2])  # ∂y/∂u₁, ∂y/∂u₂
    inner_derivs = np.array([2.0, 1.0])  # du₁/dx, du₂/dx

    result = chain_rule_multi(outer_derivs, inner_derivs)
    expected = 4 * u1 + 2 * u2  # = 24 + 8 = 32

    print(f"  At x = {x_val}:")
    print(f"  u₁ = {u1}, u₂ = {u2}")
    print(f"  ∂y/∂u₁ = {2*u1}, ∂y/∂u₂ = {2*u2}")
    print(f"  du₁/dx = 2, du₂/dx = 1")
    print(f"\n  Chain rule: ({2*u1})·(2) + ({2*u2})·(1) = {result}")
    print(f"  Expected: 4·{u1} + 2·{u2} = {expected}")
    print(f"  Match: {abs(result - expected) < 1e-10}")


def example_3_backpropagation() -> None:
    """How backpropagation applies the chain rule."""
    print_section("Example 3: Backpropagation = Chain Rule")

    print("""
    A simple neural network:

    x → [w·x + b] → [σ(·)] → [L(·, target)]

    Loss = L(σ(wx + b), target)

    To update w: ∂Loss/∂w = ∂Loss/∂a · ∂a/∂z · ∂z/∂w

    Where:
    - ∂z/∂w = x (input)
    - ∂a/∂z = σ'(z) (activation derivative)
    - ∂Loss/∂a = 2(a - target) (loss derivative)

    The chain rule multiplies these three factors.
    """)

    result = neural_network_chain_rule_demo()

    print(f"  Network: x={result['x']:.1f}, w={result['w']:.1f}, b={result['b']:.1f}")
    print(f"  z = wx + b = {result['z']:.1f}")
    print(f"  a = σ(z) = {result['a']:.4f}")
    print(f"  target = {result['target']:.1f}")
    print(f"  loss = {result['loss']:.4f}")
    print(f"\n  Backpropagation:")
    print(f"    ∂Loss/∂z = 2(a - target) = {result['dloss_dz']:.4f}")
    print(f"    ∂Loss/∂w = ∂Loss/∂z · x = {result['dloss_dw']:.4f}")
    print(f"    ∂Loss/∂b = ∂Loss/∂z · 1 = {result['dloss_db']:.4f}")

    print(f"\n  Gradient descent update (lr=0.1):")
    new_w = result['w'] - 0.1 * result['dloss_dw']
    new_b = result['b'] - 0.1 * result['dloss_db']
    print(f"    w_new = {result['w']:.1f} - 0.1 × ({result['dloss_dw']:.4f}) = {new_w:.4f}")
    print(f"    b_new = {result['b']:.1f} - 0.1 × ({result['dloss_db']:.4f}) = {new_b:.4f}")


def example_4_deep_network() -> None:
    """Chain rule in a deeper network."""
    print_section("Example 4: Chain Rule in Deep Networks")

    print("""
    In a network with L layers, the gradient through ALL layers is:

    ∂Loss/∂w⁽¹⁾ = ∂Loss/∂a⁽ᴸ⁾ · [∏ₖ₌₂ᴸ σ'(z⁽ᵏ⁾) · w⁽ᵏ⁾] · ∂a⁽¹⁾/∂w⁽¹⁾

    This is a PRODUCT of many terms.

    Problems:
    1. If σ'(z) < 1 for many layers → product → 0 (VANISHING GRADIENT)
    2. If σ'(z) > 1 for many layers → product → ∞ (EXPLODING GRADIENT)

    Solution: Use ReLU (σ'(x) = 1 for x > 0, no scaling)
    """)

    # Demonstrate vanishing gradient with sigmoid
    print("  Vanishing gradient through 10 sigmoid layers:\n")
    z = 1.0  # pre-activation
    product = 1.0
    for layer in range(1, 11):
        s = 1.0 / (1.0 + np.exp(-z))
        s_prime = s * (1 - s)
        product *= s_prime
        print(f"    Layer {layer:>2}: σ'(z)={s_prime:.4f}, cumulative product={product:.2e}")
        z = s  # output becomes input to next layer

    print(f"\n  After 10 layers: gradient = {product:.2e} (essentially zero!)")
    print("  This is why deep sigmoid networks are hard to train.")


if __name__ == "__main__":
    print("CHAIN RULE — THE ENGINE OF BACKPROPAGATION")
    print("=" * 55)

    example_1_basic_chain_rule()
    example_2_multivariable_chain_rule()
    example_3_backpropagation()
    example_4_deep_network()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Chain rule: d/dx f(g(x)) = f'(g(x)) · g'(x)
    2. Multivariable: sum over all paths
    3. Backpropagation = chain rule applied layer by layer
    4. Deep networks: product of many derivatives
    5. Vanishing/exploding gradients: product too small/large
    6. ReLU solves vanishing gradient (derivative = 1 for x > 0)
    """)
