"""
Experiment 6: Optimizer Comparison
===================================

HYPOTHESIS: GD, Momentum, and Adam have different convergence behaviors
on the same objective due to different update rules.

OBJECTIVE: Compare gradient descent, momentum, and Adam on the same problem.

METHOD: Use Rosenbrock function. Same initial conditions. Track objective,
gradient norm, and convergence.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.adam import adam
from math_for_neural_networks.optimization.objectives import rosenbrock, rosenbrock_gradient
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the optimizer comparison experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Optimizer Comparison",
        hypothesis="Different optimizers have different convergence behaviors",
        method="Compare GD, Momentum, Adam on Rosenbrock function",
    )

    initial = np.array([-1.0, 1.0])
    max_iter = 2000

    gd = gradient_descent(
        rosenbrock,
        rosenbrock_gradient,
        initial,
        learning_rate=0.001,
        max_iterations=max_iter,
        grad_tol=1e-8,
    )
    mom = momentum(
        rosenbrock,
        rosenbrock_gradient,
        initial,
        learning_rate=0.001,
        beta=0.9,
        max_iterations=max_iter,
        grad_tol=1e-8,
    )
    ad = adam(
        rosenbrock,
        rosenbrock_gradient,
        initial,
        learning_rate=0.005,
        max_iterations=max_iter,
        grad_tol=1e-8,
    )

    rows = []
    for name, res in [("GD", gd), ("Momentum", mom), ("Adam", ad)]:
        status = "converged" if res.converged else "max_iter"
        rows.append(
            [
                name,
                f"{res.final_objective:.6f}",
                f"{res.final_gradient_norm:.4e}",
                f"{res.iterations}",
                status,
            ]
        )

    table = print_table(
        ["Optimizer", "Final Objective", "Final Grad Norm", "Iterations", "Status"],
        rows,
        title="Optimizer Comparison on Rosenbrock (x0=[-1, 1])",
    )
    print(table)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    styles = {"GD": "-", "Momentum": "--", "Adam": ":"}
    colors = {"GD": "blue", "Momentum": "orange", "Adam": "red"}

    for (
        name,
        res,
    ) in zip(["GD", "Momentum", "Adam"], [gd, mom, ad]):
        axes[0].plot(
            res.objective_history, linestyle=styles[name], color=colors[name], label=name, alpha=0.8
        )
        axes[1].plot(
            res.gradient_norm_history,
            linestyle=styles[name],
            color=colors[name],
            label=name,
            alpha=0.8,
        )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("Objective vs Iteration")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norm vs Iteration")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 6: Optimizer Comparison on Rosenbrock", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "optimizer_comparison", experiment_name="06_optimizer_comparison")

    result.metrics["gd_final_obj"] = gd.final_objective
    result.metrics["momentum_final_obj"] = mom.final_objective
    result.metrics["adam_final_obj"] = ad.final_objective
    result.metrics["gd_iterations"] = gd.iterations
    result.metrics["momentum_iterations"] = mom.iterations
    result.metrics["adam_iterations"] = ad.iterations
    result.observations.append(
        f"GD: {gd.iterations} iterations, objective={gd.final_objective:.6f}"
    )
    result.observations.append(
        f"Momentum: {mom.iterations} iterations, objective={mom.final_objective:.6f}"
    )
    result.observations.append(
        f"Adam: {ad.iterations} iterations, objective={ad.final_objective:.6f}"
    )
    result.conclusion = (
        "Different optimizers exhibit different convergence patterns. "
        "Momentum and adaptive methods can accelerate convergence on certain problems. "
        "No single optimizer is universally best."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
