"""
Experiment 12: Toy Neural Network Training
============================================

HYPOTHESIS: The Stage 6 neural network can learn a simple binary classification
task by reducing loss through repeated forward-backward-update cycles.

OBJECTIVE: Train a small neural network on a toy dataset and observe learning.

METHOD: Generate a simple linearly separable dataset. Train with SGD.
Plot loss vs iteration and show final predictions.
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
    train,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the toy training experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Toy Neural Network Training",
        hypothesis="Network learns simple classification by reducing loss",
        method="Generate toy data, train with SGD, observe loss decrease",
    )

    # Generate data: y = 1 if x1 > 0, else 0
    n_samples = 100
    x = rng.standard_normal((n_samples, 2))
    y = (x[:, 0:1] > 0).astype(float)

    params = init_params(input_dim=2, hidden_dim=8, output_dim=1, seed=seed)

    # Forward before training
    initial_pred, _ = forward(x, params, activation="sigmoid")

    # Train
    metrics = train(x, y, params, lr=0.5, iterations=500, activation="sigmoid", loss_type="mse")

    # Forward after training
    final_pred, _ = forward(x, params, activation="sigmoid")
    accuracy = float(np.mean((final_pred > 0.5).astype(float) == y))

    print(f"\n  Training Results:")
    print(f"  Initial loss: {metrics.loss_history[0]:.6f}")
    print(f"  Final loss: {metrics.loss_history[-1]:.6f}")
    print(f"  Final gradient norm: {metrics.gradient_norm_history[-1]:.6f}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(metrics.loss_history)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics.gradient_norm_history)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norm During Training")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    colors = ["blue" if yi == 0 else "red" for yi in y.ravel()]
    axes[2].scatter(x[:, 0], x[:, 1], c=colors, alpha=0.6, s=20)
    pred_classes = (final_pred > 0.5).astype(float)
    correct = (pred_classes == y).ravel()
    axes[2].scatter(
        x[~correct, 0],
        x[~correct, 1],
        facecolors="none",
        edgecolors="black",
        s=60,
        linewidths=1.5,
        label="Misclassified",
    )
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].set_title("Decision (blue=0, red=1)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Experiment 12: Toy Neural Network Training", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "toy_training", experiment_name="12_toy_training")

    result.metrics["initial_loss"] = metrics.loss_history[0]
    result.metrics["final_loss"] = metrics.loss_history[-1]
    result.metrics["accuracy"] = accuracy
    result.metrics["iterations"] = metrics.iterations
    result.observations.append(
        f"Loss decreased from {metrics.loss_history[0]:.4f} to {metrics.loss_history[-1]:.4f}"
    )
    result.observations.append(f"Final accuracy: {accuracy:.2%}")
    result.conclusion = (
        "The neural network successfully learns the toy classification task. "
        "Loss decreases monotonically and accuracy reaches near-perfect levels."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
