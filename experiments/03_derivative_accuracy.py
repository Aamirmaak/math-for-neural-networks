"""
Experiment 3: Derivative Approximation
========================================

HYPOTHESIS: Forward and central finite differences approximate analytical
derivatives, with accuracy depending on step size h. There is a tradeoff
between truncation error (large h) and round-off error (small h).

OBJECTIVE: Compare analytical vs numerical derivative approximation across
step sizes.

METHOD: Use f(x) = sin(x), f'(x) = cos(x). Vary h from 1e-1 to 1e-14.
Measure absolute error for forward and central differences.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.calculus.finite_differences import (
    forward_difference,
    central_difference,
    step_size_analysis,
)
from math_for_neural_networks.calculus.derivatives import sin_func, sin_derivative
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the derivative approximation experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Derivative Approximation",
        hypothesis="Central differences are more accurate than forward differences",
        method="Compare analytical vs numerical derivatives across step sizes",
    )

    x = 1.5
    true_deriv = float(sin_derivative(x))
    h_values = np.logspace(-1, -14, 14)

    analysis = step_size_analysis(sin_func, x, true_deriv, h_values)
    forward_errors = analysis["forward_errors"]
    central_errors = analysis["central_errors"]

    rows = []
    for i in range(0, len(h_values), 3):
        h = h_values[i]
        fwd_err = forward_errors[i]
        cen_err = central_errors[i]
        rows.append([f"{h:.1e}", f"{fwd_err:.4e}", f"{cen_err:.4e}"])

    table = print_table(
        ["Step h", "Forward Error", "Central Error"],
        rows,
        title="Derivative Approximation Errors (f(x) = sin(x), x = 1.5)",
    )
    print(table)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].loglog(h_values, forward_errors, "o-", label="Forward O(h)", markersize=4)
    axes[0].loglog(h_values, central_errors, "s-", label="Central O(h^2)", markersize=4)
    axes[0].set_xlabel("Step size h")
    axes[0].set_ylabel("Absolute Error")
    axes[0].set_title("Numerical Derivative Error vs Step Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    best_central_idx = np.argmin(central_errors)
    best_forward_idx = np.argmin(forward_errors)
    axes[1].bar(
        ["Forward (best)", "Central (best)"],
        [forward_errors[best_forward_idx], central_errors[best_central_idx]],
        color=["steelblue", "coral"],
    )
    axes[1].set_ylabel("Best Absolute Error Achieved")
    axes[1].set_title("Best Error: Forward vs Central")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 3: Derivative Approximation Accuracy", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "derivative_accuracy", experiment_name="03_derivative_accuracy")

    result.metrics["true_derivative"] = true_deriv
    result.metrics["best_forward_error"] = float(forward_errors[best_forward_idx])
    result.metrics["best_forward_h"] = float(h_values[best_forward_idx])
    result.metrics["best_central_error"] = float(central_errors[best_central_idx])
    result.metrics["best_central_h"] = float(h_values[best_central_idx])
    result.observations.append(
        f"Forward difference best h: {h_values[best_forward_idx]:.1e}, "
        f"error: {forward_errors[best_forward_idx]:.4e}"
    )
    result.observations.append(
        f"Central difference best h: {h_values[best_central_idx]:.1e}, "
        f"error: {central_errors[best_central_idx]:.4e}"
    )
    result.observations.append("For very small h, round-off error dominates (error increases)")
    result.conclusion = (
        "Central differences are generally more accurate than forward differences. "
        "Both suffer from round-off error at very small step sizes. "
        "There exists an optimal step size that balances truncation and round-off error."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
