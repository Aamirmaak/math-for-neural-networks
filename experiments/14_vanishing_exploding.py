"""
Experiment 14: Vanishing/Exploding Gradients
=============================================

HYPOTHESIS: Repeated application of sigmoid activation causes gradient
magnitudes to shrink exponentially (vanishing gradient).

OBJECTIVE: Demonstrate gradient magnitude change through multiple layers.

METHOD: Propagate gradient through N sigmoid layers, tracking cumulative
gradient magnitude at each step.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.activations import sigmoid
from experiment_utils import ExperimentResult, save_fig, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the vanishing gradient experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Vanishing/Exploding Gradients",
        hypothesis="Repeated sigmoid causes gradient to vanish exponentially",
        method="Propagate gradient through N sigmoid layers",
    )

    n_layers = 15
    x = 0.5

    grad_magnitudes = []
    cumulative_grad = 1.0

    for layer in range(n_layers):
        s = float(sigmoid(x))
        local_grad = s * (1 - s)
        cumulative_grad *= local_grad
        grad_magnitudes.append(cumulative_grad)
        x += 0.1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, n_layers + 1), grad_magnitudes, "o-", linewidth=2)
    axes[0].set_xlabel("Layer Depth")
    axes[0].set_ylabel("Cumulative Gradient Magnitude")
    axes[0].set_title("Gradient Magnitude Through Sigmoid Layers")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    layer_grads = [
        float(sigmoid(0.5 + 0.1 * i) * (1 - sigmoid(0.5 + 0.1 * i))) for i in range(n_layers)
    ]
    axes[1].bar(range(1, n_layers + 1), layer_grads, alpha=0.7)
    axes[1].set_xlabel("Layer Depth")
    axes[1].set_ylabel("Local Gradient sigmoid'(x)")
    axes[1].set_title("Local Gradient Per Layer")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Experiment 14: Vanishing Gradient Intuition", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "vanishing_gradient", experiment_name="14_vanishing_exploding")

    result.metrics["n_layers"] = n_layers
    result.metrics["initial_gradient"] = grad_magnitudes[0]
    result.metrics["final_gradient"] = grad_magnitudes[-1]
    result.metrics["reduction_factor"] = grad_magnitudes[-1] / grad_magnitudes[0]
    result.observations.append(
        f"After {n_layers} sigmoid layers, gradient reduced by factor "
        f"{grad_magnitudes[-1] / grad_magnitudes[0]:.2e}"
    )
    result.observations.append("Each sigmoid layer multiplies gradient by sigmoid'(x) <= 0.25")
    result.conclusion = (
        "Repeated sigmoid activation causes gradient to shrink exponentially. "
        "This is the vanishing gradient problem. Solutions include ReLU activation, "
        "batch normalization, and residual connections."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
