"""
Experiment 13: Gradient Flow
==============================

HYPOTHESIS: Gradient magnitudes differ across layers in a neural network,
reflecting how error signal flows backward through the network.

OBJECTIVE: Track gradient norms through layers during training.

METHOD: Train a small network. Record gradient norms for each layer
at each iteration. Visualize gradient flow.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.training import (
    init_params,
    forward,
    backward,
    sgd_step,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the gradient flow experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Gradient Flow",
        hypothesis="Gradient magnitudes differ across layers",
        method="Track gradient norms per layer during training",
    )

    n_samples = 50
    x = rng.standard_normal((n_samples, 2))
    y = (x[:, 0:1] > 0).astype(float)

    params = init_params(input_dim=2, hidden_dim=8, output_dim=1, seed=seed)

    gw1_norms, gb1_norms, gw2_norms, gb2_norms = [], [], [], []
    n_iters = 200
    lr = 0.5

    for i in range(n_iters):
        output, cache = forward(x, params, activation="sigmoid")
        grads = backward(y, output, params, cache, activation="sigmoid", loss_type="mse")

        gw1_norms.append(float(np.linalg.norm(grads["dW1"])))
        gb1_norms.append(float(np.linalg.norm(grads["db1"])))
        gw2_norms.append(float(np.linalg.norm(grads["dW2"])))
        gb2_norms.append(float(np.linalg.norm(grads["db2"])))

        sgd_step(params, grads, lr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gw1_norms, label="dW1 (layer 1 weights)", alpha=0.8)
    axes[0].plot(gw2_norms, label="dW2 (layer 2 weights)", alpha=0.8)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Gradient Norm")
    axes[0].set_title("Weight Gradient Norms")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gb1_norms, label="db1 (layer 1 bias)", alpha=0.8)
    axes[1].plot(gb2_norms, label="db2 (layer 2 bias)", alpha=0.8)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Bias Gradient Norms")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 13: Gradient Flow Through Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "gradient_flow", experiment_name="13_gradient_flow")

    result.metrics["initial_dW1_norm"] = gw1_norms[0]
    result.metrics["initial_dW2_norm"] = gw2_norms[0]
    result.metrics["final_dW1_norm"] = gw1_norms[-1]
    result.metrics["final_dW2_norm"] = gw2_norms[-1]
    result.metrics["dW1_ratio"] = gw1_norms[-1] / gw1_norms[0] if gw1_norms[0] > 0 else 0
    result.metrics["dW2_ratio"] = gw2_norms[-1] / gw2_norms[0] if gw2_norms[0] > 0 else 0
    result.observations.append(
        f"Layer 1 weight gradient norm: {gw1_norms[0]:.4f} -> {gw1_norms[-1]:.4f}"
    )
    result.observations.append(
        f"Layer 2 weight gradient norm: {gw2_norms[0]:.4f} -> {gw2_norms[-1]:.4f}"
    )
    result.conclusion = (
        "Gradient magnitudes differ between layers and change during training. "
        "Layer 2 (closer to output) generally has larger gradients than Layer 1. "
        "Both decrease as the network converges."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
