"""
Adam — Adaptive Learning Rate Optimization
============================================

Adam combines momentum and RMSProp to give each parameter
its own adaptive learning rate based on gradient history.

This example demonstrates:
1. Why adaptive learning rates matter
2. Adam's two key ideas: momentum + scaling
3. Bias correction mechanism
4. Comparison: GD vs Momentum vs Adam
"""

import numpy as np

from math_for_neural_networks.optimization.adam import adam, adam_step
from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.objectives import quadratic


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_adam_step() -> None:
    """Show one step of Adam by hand."""
    print_section("Example 1: One Step of Adam (By Hand)")

    print("""
    Adam maintains two moving averages:
    - m_t: first moment (mean) of gradients -> momentum
    - v_t: second moment (uncentered variance) of gradients -> scaling

    Update rules:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)     <- bias correction
        v_hat = v_t / (1 - beta2^t)     <- bias correction
        x_t = x_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    """)

    x = np.array([5.0])
    lr = 0.1
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    m = np.zeros_like(x)
    v = np.zeros_like(x)

    print(f"  Starting at x = {x[0]}, f(x) = {f(x):.4f}")
    print(f"  lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}")
    print(f"\n  {'t':>3} | {'x':>8} | {'g':>8} | {'m':>8} | {'v':>8} | {'m_hat':>8} | {'v_hat':>8} | {'f(x)':>10}")
    print("  " + "-" * 75)

    for t in range(1, 6):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        print(
            f"  {t:>3} | {x[0]:>8.4f} | {g[0]:>8.4f} | {m[0]:>8.4f} | {v[0]:>8.4f} | "
            f"{m_hat[0]:>8.4f} | {v_hat[0]:>8.4f} | {f(x):>10.6f}"
        )

    print(f"\n  After 5 steps: x = {x[0]:.6f}, f(x) = {f(x):.8f}")


def example_2_bias_correction() -> None:
    """Show why bias correction matters."""
    print_section("Example 2: Bias Correction")

    print("""
    At t=1 with beta1=0.9:
    - m_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1
    - Without correction: m_hat = 0.1 * g_1 (underestimates!)
    - With correction: m_hat = 0.1 * g_1 / (1 - 0.9) = g_1 (correct!)

    Bias correction compensates for zero initialization.
    """)

    beta1 = 0.9
    g = 1.0  # gradient

    m = 0.1 * g  # after one step
    m_uncorrected = m
    m_corrected = m / (1 - beta1)

    print(f"  After 1 step with g = {g}:")
    print(f"    m_1 = {m:.4f}")
    print(f"    Without correction: {m_uncorrected:.4f} (underestimate)")
    print(f"    With correction:    {m_corrected:.4f} (correct estimate)")

    # Show effect over time
    print(f"\n  Bias correction factor 1/(1-beta1^t):")
    for t in range(1, 11):
        factor = 1.0 / (1.0 - beta1 ** t)
        print(f"    t={t:>2}: {factor:.4f}")


def example_3_adam_step_function() -> None:
    """Use the adam_step function."""
    print_section("Example 3: Using adam_step()")

    print("""
    The adam_step() function performs a single Adam update,
    useful for understanding the internals.
    """)

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    x = np.array([5.0])
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    print(f"  {'Step':>4} | {'x':>8} | {'f(x)':>10}")
    print("  " + "-" * 30)

    for step in range(10):
        g = grad(x)
        x, m, v = adam_step(x, g, m, v, step + 1, learning_rate=0.1)
        print(f"  {step + 1:>4} | {x[0]:>8.5f} | {f(x):>10.8f}")


def example_4_three_optimizers() -> None:
    """Compare GD, Momentum, and Adam."""
    print_section("Example 4: GD vs Momentum vs Adam")

    print("""
    Function: f(x,y) = x^2 + 10*y^2 (anisotropic)
    Starting at (5, 5), 20 iterations
    """)

    def f(params: np.ndarray) -> float:
        return float(params[0] ** 2 + 10 * params[1] ** 2)

    def grad(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], 20.0 * params[1]])

    x0 = np.array([5.0, 5.0])

    gd = gradient_descent(
        f, grad, x0.copy(),
        learning_rate=0.04, max_iterations=20, param_tol=1e-10,
    )
    mom = momentum(
        f, grad, x0.copy(),
        learning_rate=0.04, beta=0.9, max_iterations=20, param_tol=1e-10,
    )
    adam_result = adam(
        f, grad, x0.copy(),
        learning_rate=0.1, max_iterations=20, param_tol=1e-10,
    )

    print(f"  {'Optimizer':>12} | {'f(final)':>12} | {'Iters':>6}")
    print("  " + "-" * 35)
    print(f"  {'GD':>12} | {gd.final_objective:>12.8f} | {gd.iterations:>6}")
    print(f"  {'Momentum':>12} | {mom.final_objective:>12.8f} | {mom.iterations:>6}")
    print(f"  {'Adam':>12} | {adam_result.final_objective:>12.8f} | {adam_result.iterations:>6}")


if __name__ == "__main__":
    print("ADAM - ADAPTIVE LEARNING RATE OPTIMIZATION")
    print("=" * 55)

    example_1_adam_step()
    example_2_bias_correction()
    example_3_adam_step_function()
    example_4_three_optimizers()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Adam combines momentum (m) and adaptive scaling (v)
    2. m_t tracks mean gradient -> accelerates in consistent directions
    3. v_t tracks gradient variance -> scales learning rate per parameter
    4. Bias correction: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)
    5. Default: beta1=0.9, beta2=0.999, lr=0.001
    6. Adam is the most popular optimizer for neural networks
    7. The epsilon (1e-8) prevents division by zero
    """)
