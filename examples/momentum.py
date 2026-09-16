"""
Momentum — Accelerating Gradient Descent
==========================================

Momentum adds a velocity term to gradient descent, helping it
accelerate in consistent directions and dampen oscillations.

This example demonstrates:
1. Why plain GD is slow in narrow valleys
2. How momentum helps (ball rolling downhill analogy)
3. Effect of momentum coefficient (beta)
4. Comparison: GD vs Momentum
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.objectives import quadratic


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_why_momentum() -> None:
    """Show the problem that momentum solves."""
    print_section("Example 1: The Problem — Narrow Valley")

    print("""
    Function: f(x,y) = x^2 + 10*y^2

    This function is narrow in the y-direction (10x steeper).
    Plain GD oscillates in y while slowly moving in x.

    Starting at (5, 5) with lr = 0.04:
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    # Plain GD
    params = np.array([5.0, 5.0])
    lr = 0.04

    print(f"  {'Step':>4} | {'(x, y)':>16} | {'f(x,y)':>12} | {'|y| > |x|?':>10}")
    print("  " + "-" * 50)

    for step in range(8):
        print(f"  {step:>4} | ({params[0]:>6.3f}, {params[1]:>6.3f}) | {f(params):>12.4f} | {'YES' if abs(params[1]) > abs(params[0]) else 'no':>10}")
        params = params - lr * grad(params)

    print("\n  Notice: y oscillates while x converges slowly.")


def example_2_momentum_solution() -> None:
    """Show how momentum helps."""
    print_section("Example 2: Momentum Solution")

    print("""
    Momentum: v_t = beta * v_{t-1} + gradient
              x_t = x_{t-1} - lr * v_t

    The velocity accumulates gradients in consistent directions
    and cancels oscillations.
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    # Momentum SGD
    x0 = np.array([5.0, 5.0])
    result = momentum(
        f, grad, x0,
        learning_rate=0.04,
        beta=0.9,
        max_iterations=8,
        param_tol=1e-8,
    )

    print(f"  beta = 0.9, lr = 0.04")
    print(f"\n  Iterations: {result.iterations}")
    print(f"  Final: ({result.parameters[0]:.4f}, {result.parameters[1]:.4f})")
    print(f"  Final loss: {result.final_objective:.6f}")
    print("  Momentum smooths out the oscillation in y.")


def example_3_beta_comparison() -> None:
    """Compare different momentum coefficients."""
    print_section("Example 3: Momentum Coefficient (beta) Comparison")

    print("""
    beta = 0.0: no momentum (plain GD)
    beta = 0.5: moderate momentum
    beta = 0.9: strong momentum (common default)
    beta = 0.99: very strong momentum (may overshoot)
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    betas = [0.0, 0.5, 0.9, 0.99]

    print(f"  {'beta':>6} | {'f(final)':>12} | {'Iterations':>10} | {'Converged':>10}")
    print("  " + "-" * 45)

    for beta in betas:
        x0 = np.array([5.0, 5.0])
        result = momentum(
            f, grad, x0,
            learning_rate=0.04,
            beta=beta,
            max_iterations=20,
            param_tol=1e-8,
        )
        print(
            f"  {beta:>6.2f} | {result.final_objective:>12.6f} | {result.iterations:>10} | {str(result.converged):>10}"
        )


def example_4_gd_vs_momentum() -> None:
    """Head-to-head comparison."""
    print_section("Example 4: GD vs Momentum (Head-to-Head)")

    print("""
    Function: f(x,y) = x^2 + 10*y^2
    Starting at (5, 5), lr = 0.04, 15 iterations
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    # Plain GD
    gd_result = gradient_descent(
        f, grad, np.array([5.0, 5.0]),
        learning_rate=0.04, max_iterations=15, param_tol=1e-8,
    )

    # Momentum
    mom_result = momentum(
        f, grad, np.array([5.0, 5.0]),
        learning_rate=0.04, beta=0.9, max_iterations=15, param_tol=1e-8,
    )

    print(f"  {'Method':>12} | {'f(final)':>12} | {'Iters':>6} | {'Converged':>10}")
    print("  " + "-" * 45)
    print(f"  {'GD':>12} | {gd_result.final_objective:>12.6f} | {gd_result.iterations:>6} | {str(gd_result.converged):>10}")
    print(f"  {'Momentum':>12} | {mom_result.final_objective:>12.6f} | {mom_result.iterations:>6} | {str(mom_result.converged):>10}")


if __name__ == "__main__":
    print("MOMENTUM - ACCELERATING GRADIENT DESCENT")
    print("=" * 55)

    example_1_why_momentum()
    example_2_momentum_solution()
    example_3_beta_comparison()
    example_4_gd_vs_momentum()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Momentum accumulates velocity in consistent gradient directions
    2. It dampens oscillations in narrow valleys
    3. beta = 0.9 is a common default (90% of previous velocity)
    4. beta = 0 reduces to plain gradient descent
    5. Momentum helps SGD converge faster on ill-conditioned problems
    6. Foundation for more advanced optimizers (Nesterov, Adam)
    """)
