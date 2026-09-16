"""
Learning Rate — The Most Important Hyperparameter
==================================================

The learning rate controls how big a step we take during optimization.
Too small: slow convergence. Too large: divergence. Just right: fast, stable.

This example demonstrates:
1. Learning rate sensitivity analysis
2. Oscillation behavior
3. Adaptive vs fixed learning rates
4. Learning rate scheduling concept
"""

import numpy as np

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import quadratic


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_lr_grid_search() -> None:
    """Test a range of learning rates."""
    print_section("Example 1: Learning Rate Grid Search")

    print("""
    Function: f(x) = x^2 (minimum at x = 0)
    Starting at x = 5.0
    Testing learning rates from 0.01 to 0.99
    """)

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.6, 0.8, 0.99]

    print(f"  {'lr':>6} | {'f(x_0)':>10} | {'f(x_10)':>10} | {'Status':>12}")
    print("  " + "-" * 45)

    for lr in learning_rates:
        x = np.array([5.0])
        for _ in range(10):
            x = x - lr * grad(x)

        val = f(x)
        if val < 0.001:
            status = "CONVERGED"
        elif val > 1000:
            status = "DIVERGED"
        elif val > 10:
            status = "OSCILLATING"
        else:
            status = "SLOW"

        print(f"  {lr:>6.2f} | {f(np.array([5.0])):>10.2f} | {val:>10.4f} | {status:>12}")


def example_2_lr_schedule_concept() -> None:
    """Demonstrate the concept of learning rate scheduling."""
    print_section("Example 2: Learning Rate Schedule Concept")

    print("""
    In practice, we often decrease the learning rate over time:
    - Start large (explore quickly)
    - Decrease gradually (refine the solution)

    This is called a learning rate schedule.
    """)

    def f(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    # Fixed learning rate
    x_fixed = np.array([5.0])
    lr_fixed = 0.1
    values_fixed = [f(x_fixed)]
    for _ in range(15):
        x_fixed = x_fixed - lr_fixed * grad(x_fixed)
        values_fixed.append(f(x_fixed))

    # Decreasing learning rate: lr_t = lr_0 / (1 + 0.1 * t)
    x_decay = np.array([5.0])
    values_decay = [f(x_decay)]
    for t in range(15):
        lr_t = 0.3 / (1 + 0.15 * t)
        x_decay = x_decay - lr_t * grad(x_decay)
        values_decay.append(f(x_decay))

    print(f"  {'Step':>4} | {'Fixed lr=0.1':>14} | {'Decaying lr':>14}")
    print("  " + "-" * 38)

    for t in range(16):
        print(
            f"  {t:>4} | {values_fixed[t]:>14.6f} | {values_decay[t]:>14.6f}"
        )


def example_3_lr_and_loss_landscape() -> None:
    """Show how learning rate interacts with loss landscape curvature."""
    print_section("Example 3: Learning Rate and Curvature")

    print("""
    Function: f(x) = a*x^2 (parabola)
    Gradient: f'(x) = 2*a*x
    Update: x <- x - lr * 2*a*x = x * (1 - 2*a*lr)

    For convergence: |1 - 2*a*lr| < 1
    => lr < 1/a (for positive a)

    Steeper curvature (larger a) requires smaller learning rate!
    """)

    def f_factory(a: float):
        def f(x: np.ndarray) -> float:
            return float(a * np.sum(x ** 2))
        return f

    def grad_factory(a: float):
        def grad(x: np.ndarray) -> np.ndarray:
            return 2.0 * a * x
        return grad

    curvatures = [0.5, 1.0, 2.0, 5.0]
    lr = 0.2

    print(f"  Learning rate: {lr}")
    print(f"  {'Curvature (a)':>14} | {'Max safe lr':>12} | {'f(x_10)':>10} | {'Status':>10}")
    print("  " + "-" * 55)

    for a in curvatures:
        max_lr = 1.0 / a
        f = f_factory(a)
        grad = grad_factory(a)
        x = np.array([5.0])
        for _ in range(10):
            x = x - lr * grad(x)
        val = f(x)
        status = "OK" if val < 1.0 else ("DIVERGED" if val > 1000 else "UNSTABLE")
        print(
            f"  {a:>14.1f} | {max_lr:>12.3f} | {val:>10.4f} | {status:>10}"
        )


if __name__ == "__main__":
    print("LEARNING RATE - THE MOST IMPORTANT HYPERPARAMETER")
    print("=" * 55)

    example_1_lr_grid_search()
    example_2_lr_schedule_concept()
    example_3_lr_and_loss_landscape()

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("""
    1. Learning rate too small: slow convergence
    2. Learning rate too large: oscillation or divergence
    3. For f(x) = a*x^2, max stable lr = 1/a
    4. Learning rate scheduling (decay) helps in practice
    5. Different parameters may need different learning rates
    6. Adaptive methods (Adam) handle this automatically
    """)
