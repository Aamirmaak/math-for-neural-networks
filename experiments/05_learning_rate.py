"""
Experiment 5: Learning Rate Effect
===================================

HYPOTHESIS: Learning rate affects convergence speed and stability.
Too small: slow convergence. Too large: divergence. Just right: fast convergence.

OBJECTIVE: Compare gradient descent behavior with different learning rates.

METHOD: Use f(x) = x^2 (quadratic). Start at x = 3.0. Compare learning rates
0.01, 0.1, 0.5, 0.9, 1.1. Track objective value and position.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import quadratic, quadratic_gradient
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the learning rate experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Learning Rate Effect",
        hypothesis="Learning rate affects convergence speed and stability",
        method="Compare gradient descent with different learning rates on f(x) = x^2",
    )

    initial_position = np.array([3.0])
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
    max_iter = 50

    histories = {}
    rows = []

    for lr in learning_rates:
        result_gd = gradient_descent(
            quadratic,
            quadratic_gradient,
            initial_position,
            learning_rate=lr,
            max_iterations=max_iter,
            grad_tol=1e-10,
        )
        histories[lr] = result_gd.objective_history
        status = "converged" if result_gd.converged else "diverged"
        rows.append(
            [f"{lr}", f"{result_gd.final_objective:.6f}", f"{result_gd.iterations}", status]
        )

    table = print_table(
        ["Learning Rate", "Final Objective", "Iterations", "Status"],
        rows,
        title="Learning Rate Effect on f(x) = x^2, x0 = 3.0",
    )
    print(table)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for lr, obj_hist in histories.items():
        label = f"lr={lr}"
        style = "--" if lr >= 1.0 else "-"
        axes[0].plot(obj_hist, style, label=label, alpha=0.8)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective f(x) = x^2")
    axes[0].set_title("Objective vs Iteration")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for lr, obj_hist in histories.items():
        label = f"lr={lr}"
        style = "--" if lr >= 1.0 else "-"
        axes[1].plot(obj_hist, style, label=label, alpha=0.8)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Objective (log scale)")
    axes[1].set_title("Objective vs Iteration (zoomed)")
    axes[1].set_ylim(1e-10, 1e2)
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 5: Learning Rate Effect on Convergence", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "learning_rate_effect", experiment_name="05_learning_rate")

    converged_rates = [lr for lr, h in histories.items() if h[-1] < 1e-6]
    diverged_rates = [lr for lr, h in histories.items() if h[-1] > 1e2 or np.isnan(h[-1])]

    result.metrics["converged_learning_rates"] = converged_rates
    result.metrics["diverged_learning_rates"] = diverged_rates
    result.observations.append(f"Converged: {converged_rates}")
    result.observations.append(f"Failed/diverged: {diverged_rates}")
    result.conclusion = (
        "Learning rate critically affects training. Too small: slow convergence. "
        "Too large: oscillation or divergence. Finding appropriate LR is essential."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
