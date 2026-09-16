"""
Experiment 7: Activation Functions
===================================

HYPOTHESIS: Sigmoid, tanh, ReLU, and GELU have different output ranges,
gradient behaviors, and saturation characteristics.

OBJECTIVE: Compare activation functions and their derivatives visually.

METHOD: Plot each activation and its derivative over a range of inputs.
Discuss output range, saturation, gradient flow.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.activations import (
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    relu,
    relu_derivative,
    gelu,
    gelu_derivative,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the activation function experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Activation Functions",
        hypothesis="Different activations have different gradient characteristics",
        method="Plot activations and derivatives, analyze ranges and saturation",
    )

    x = np.linspace(-5, 5, 200)

    activations = [
        ("Sigmoid", sigmoid(x), sigmoid_derivative(x), (0, 1)),
        ("Tanh", tanh(x), tanh_derivative(x), (-1, 1)),
        ("ReLU", relu(x), relu_derivative(x), (0, np.max(x))),
        ("GELU", gelu(x), gelu_derivative(x), (np.min(gelu(x)), np.max(gelu(x)))),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, vals, derivs, output_range) in enumerate(activations):
        axes[0, idx].plot(x, vals, label=name, linewidth=2)
        axes[0, idx].set_title(f"{name} (range: [{output_range[0]:.1f}, {output_range[1]:.1f}])")
        axes[0, idx].set_xlabel("x")
        axes[0, idx].set_ylabel(f"{name}(x)")
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].axhline(y=0, color="k", linewidth=0.5)

        axes[1, idx].plot(x, derivs, label=f"{name}'", linewidth=2, color="orange")
        axes[1, idx].set_title(f"{name}' Derivative")
        axes[1, idx].set_xlabel("x")
        axes[1, idx].set_ylabel(f"{name}'(x)")
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].axhline(y=0, color="k", linewidth=0.5)

        max_deriv = float(np.max(np.abs(derivs)))
        result.metrics[f"{name.lower()}_max_derivative"] = max_deriv

    fig.suptitle("Experiment 7: Activation Functions and Their Derivatives", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "activation_functions", experiment_name="07_activation_functions")

    result.metrics["sigmoid_output_range"] = "[0, 1]"
    result.metrics["tanh_output_range"] = "[-1, 1]"
    result.metrics["relu_output_range"] = "[0, inf)"
    result.metrics["sigmoid_max_derivative"] = 0.25

    result.observations.append("Sigmoid output range (0,1), max derivative = 0.25 at x=0")
    result.observations.append(
        "Tanh output range (-1,1), zero-centered, max derivative = 1.0 at x=0"
    )
    result.observations.append("ReLU: 0 for x<0, identity for x>0, derivative is 0 or 1")
    result.observations.append("GELU: smooth approximation to ReLU, non-zero gradient everywhere")
    result.observations.append(
        "Sigmoid/tanh suffer from vanishing gradients for large |x|. "
        "ReLU avoids this but has dead neuron problem (gradient=0 for x<0)."
    )
    result.conclusion = (
        "Each activation function has distinct characteristics affecting gradient flow. "
        "Sigmoid saturates at extremes. Tanh is zero-centered. ReLU is simple but can die. "
        "GELU provides smooth transition. Choice depends on the specific network architecture."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
