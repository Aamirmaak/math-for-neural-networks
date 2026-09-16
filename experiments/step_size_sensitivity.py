"""
Step-Size Sensitivity Experiment
================================

Demonstrates how the choice of step size h affects the accuracy of
numerical derivatives. This is critical for understanding why gradient
computation in neural networks requires careful numerical considerations.

Key insights:
1. Too large h → truncation error dominates (method is inaccurate)
2. Too small h → roundoff error dominates (floating-point limitations)
3. Optimal h ≈ sqrt(machine_epsilon) for central differences
4. Central differences are more accurate than forward differences
"""

import numpy as np

from math_for_neural_networks.calculus.derivatives import (
    exp_func,
    exp_derivative,
    quadratic,
    quadratic_derivative,
    sin_func,
    sin_derivative,
)
from math_for_neural_networks.calculus.finite_differences import (
    central_difference,
    forward_difference,
    step_size_analysis,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demonstrate_step_size_sensitivity() -> None:
    """Show how h affects derivative accuracy for different functions."""
    print_header("STEP-SIZE SENSITIVITY EXPERIMENT")

    test_cases = [
        (quadratic, quadratic_derivative, 3.0, 6.0, "f(x) = x², f'(3) = 6"),
        (
            sin_func,
            sin_derivative,
            np.pi / 4,
            np.cos(np.pi / 4),
            "f(x) = sin(x), f'(π/4) = cos(π/4)",
        ),
        (exp_func, exp_derivative, 1.0, np.e, "f(x) = eˣ, f'(1) = e"),
    ]

    for func, deriv, x_val, true_val, desc in test_cases:
        print_header(desc)
        result = step_size_analysis(func, x_val, true_val)

        print(f"\n{'h':>12} | {'Forward Error':>15} | {'Central Error':>15}")
        print("-" * 48)
        for i in range(len(result["h_values"])):
            print(
                f"{result['h_values'][i]:>12.1e} | "
                f"{result['forward_errors'][i]:>15.2e} | "
                f"{result['central_errors'][i]:>15.2e}"
            )

        # Find optimal h
        fwd_optimal_idx = np.argmin(result["forward_errors"])
        cent_optimal_idx = np.argmin(result["central_errors"])
        print(f"\nOptimal h (forward):  {result['h_values'][fwd_optimal_idx]:.1e}")
        print(f"Optimal h (central):  {result['h_values'][cent_optimal_idx]:.1e}")


def demonstrate_floating_point_limits() -> None:
    """Show what happens when h becomes too small."""
    print_header("FLOATING-POINT LIMITS")

    def f(x: float) -> float:
        return x**2

    def f_prime(x: float) -> float:
        return 2 * x

    x_val = 3.0
    true_val = f_prime(x_val)

    print(f"\nTrue derivative of f(x) = x² at x = {x_val}: {true_val}")
    print(f"\nAs h approaches machine epsilon (~2.2e-16), errors increase:\n")

    print(f"{'h':>12} | {'Central Diff':>12} | {'Absolute Error':>15}")
    print("-" * 45)

    for h in [1e-5, 1e-8, 1e-10, 1e-12, 1e-14, 1e-15, 1e-16]:
        approx = central_difference(f, x_val, h)
        error = abs(approx - true_val)
        print(f"{h:>12.0e} | {approx:>12.8f} | {error:>15.2e}")

    print(f"\nNote: Error DECREASES as h gets smaller, then INCREASES")
    print(f"due to floating-point subtraction cancelling significant digits.")


def demonstrate_forward_vs_central() -> None:
    """Compare forward vs central difference accuracy."""
    print_header("FORWARD vs CENTRAL DIFFERENCE")

    def f(x: float) -> float:
        return sin_func(x)

    def f_prime(x: float) -> float:
        return sin_derivative(x)

    x_val = np.pi / 3
    true_val = f_prime(x_val)

    print(f"\nTrue derivative of sin(x) at x = π/3: {true_val:.10f}")
    print(f"\n{'h':>12} | {'Forward Error':>15} | {'Central Error':>15} | {'Central/Forward':>15}")
    print("-" * 65)

    for h in [1e-2, 1e-4, 1e-6, 1e-8]:
        fwd = forward_difference(f, x_val, h)
        cent = central_difference(f, x_val, h)
        fwd_err = abs(fwd - true_val)
        cent_err = abs(cent - true_val)
        ratio = cent_err / fwd_err if fwd_err > 0 else float("nan")
        print(f"{h:>12.0e} | {fwd_err:>15.2e} | {cent_err:>15.2e} | {ratio:>15.4f}")

    print(f"\nCentral difference is typically ~100x more accurate for smooth functions.")


def demonstrate_nn_relevance() -> None:
    """Explain why this matters for neural networks."""
    print_header("RELEVANCE TO NEURAL NETWORKS")

    print("""
When computing gradients in neural networks:

1. GRADIENT DESCENT relies on accurate gradient computation.
   Even small errors compound over thousands of iterations.

2. LEARNING RATE interacts with gradient accuracy:
   - If gradient has 10% error, effective learning rate changes
   - Can cause training instability or slow convergence

3. BACKPROPAGATION uses the chain rule:
   ∂Loss/∂w = ∂Loss/∂a · ∂a/∂z · ∂z/∂w
   Each factor must be accurate; errors multiply through the chain.

4. PRACTICAL GUIDELINES for numerical gradients:
   - Use central differences (not forward)
   - h ≈ 1e-5 to 1e-7 for float64
   - Never use h < 1e-8 (roundoff dominates)
   - Verify against analytical gradients when possible

5. AUTOGRAD FRAMEWORKS (PyTorch/JAX) use:
   - Exact analytical derivatives (not numerical)
   - Careful handling of activation derivatives
   - Automatic chain rule application
""")


if __name__ == "__main__":
    demonstrate_step_size_sensitivity()
    demonstrate_floating_point_limits()
    demonstrate_forward_vs_central()
    demonstrate_nn_relevance()
