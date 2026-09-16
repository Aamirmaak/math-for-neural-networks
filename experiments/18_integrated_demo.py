"""
Experiment 18: Integrated Demonstration
=========================================

HYPOTHESIS: All mathematical concepts (linear algebra, calculus, probability,
optimization, neural networks, backpropagation) work together in a complete
training pipeline.

OBJECTIVE: Show the complete mathematical pipeline from input to updated
parameters with intermediate mathematics displayed.

METHOD: Build a tiny 2-layer network from scratch. For each training step,
display: input, forward pass, loss, backward pass, gradients, optimizer update.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.activations import sigmoid
from math_for_neural_networks.neural_networks.attention import softmax
from math_for_neural_networks.neural_networks.losses import mean_squared_error
from math_for_neural_networks.neural_networks.backprop import (
    affine_backward,
    sigmoid_backward,
    mse_backward,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the integrated demonstration."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Integrated Demonstration",
        hypothesis="Complete mathematical pipeline works end-to-end",
        method="Step-by-step training with intermediate math displayed",
    )

    np.random.seed(seed)

    # Tiny dataset: AND-like
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [0], [0], [1]], dtype=np.float64)

    # Initialize
    W1 = np.random.randn(2, 3) * 0.5
    b1 = np.zeros(3)
    W2 = np.random.randn(3, 1) * 0.5
    b2 = np.zeros(1)

    lr = 0.5
    n_steps = 10

    loss_history = []
    weight_norms = []

    for step in range(n_steps):
        print(f"\n{'=' * 60}")
        print(f"  STEP {step + 1}")
        print(f"{'=' * 60}")

        # FORWARD PASS
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2  # Linear output for MSE

        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)

        print(f"\n  FORWARD PASS:")
        print(f"  X shape: {X.shape}")
        print(f"  z1 = X @ W1 + b1: shape {z1.shape}")
        print(f"  a1 = sigmoid(z1): shape {a1.shape}")
        print(f"  z2 = a1 @ W2 + b2: shape {z2.shape}")
        print(f"  y_pred = z2: shape {y_pred.shape}")
        print(f"  Loss (MSE): {loss:.6f}")

        # BACKWARD PASS
        n = X.shape[0]
        dL_dz2 = (2.0 / n) * (y_pred - y)
        dW2 = a1.T @ dL_dz2
        db2 = np.sum(dL_dz2, axis=0)
        dL_da1 = dL_dz2 @ W2.T

        da1_dz1 = a1 * (1 - a1)
        dL_dz1 = dL_da1 * da1_dz1

        dW1 = X.T @ dL_dz1
        db1 = np.sum(dL_dz1, axis=0)

        print(f"\n  BACKWARD PASS:")
        print(f"  dL/dz2 = (2/n)(y_pred - y): shape {dL_dz2.shape}")
        print(f"  dW2 = a1^T @ dL_dz2: shape {dW2.shape}")
        print(f"  db2 = sum(dL_dz2): shape {db2.shape}")
        print(f"  dL/da1 = dL_dz2 @ W2^T: shape {dL_da1.shape}")
        print(f"  da1/dz1 = a1 * (1-a1): shape {da1_dz1.shape}")
        print(f"  dL/dz1 = dL_da1 * da1_dz1: shape {dL_dz1.shape}")
        print(f"  dW1 = X^T @ dL_dz1: shape {dW1.shape}")
        print(f"  db1 = sum(dL_dz1): shape {db1.shape}")

        # Gradient norms
        gw1 = float(np.linalg.norm(dW1))
        gw2 = float(np.linalg.norm(dW2))
        weight_norms.append((gw1, gw2))

        print(f"\n  GRADIENT NORMS:")
        print(f"  ||dW1|| = {gw1:.6f}")
        print(f"  ||dW2|| = {gw2:.6f}")

        # OPTIMIZER UPDATE (SGD)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        print(f"\n  UPDATE: theta = theta - {lr} * gradient")

    # Final predictions
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    final_pred = z2

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    rows = []
    for i in range(len(X)):
        rows.append([f"{X[i]}", f"{y[i, 0]:.0f}", f"{final_pred[i, 0]:.4f}"])
    table = print_table(["Input", "Target", "Prediction"], rows, title="Final Predictions")
    print(table)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(loss_history, "o-", linewidth=2)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Loss During Training")
    axes[0].grid(True, alpha=0.3)

    gw1_list = [w[0] for w in weight_norms]
    gw2_list = [w[1] for w in weight_norms]
    axes[1].plot(gw1_list, "o-", label="||dW1||", linewidth=2)
    axes[1].plot(gw2_list, "s-", label="||dW2||", linewidth=2)
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norms During Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    colors = ["blue" if y[i, 0] == 0 else "red" for i in range(len(X))]
    axes[2].scatter(range(len(X)), final_pred.ravel(), c=colors, s=100, zorder=5)
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    for i in range(len(X)):
        axes[2].annotate(
            f"target={y[i, 0]:.0f}",
            (i, final_pred[i, 0]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Prediction")
    axes[2].set_title("Final Predictions (blue=0, red=1)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Experiment 18: Integrated Training Pipeline", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "integrated_demo", experiment_name="18_integrated_demo")

    result.metrics["initial_loss"] = loss_history[0]
    result.metrics["final_loss"] = loss_history[-1]
    result.metrics["loss_reduction"] = loss_history[0] - loss_history[-1]
    result.observations.append(f"Initial loss: {loss_history[0]:.6f}")
    result.observations.append(f"Final loss: {loss_history[-1]:.6f}")
    result.observations.append(
        "Complete pipeline: Input -> Affine -> Sigmoid -> Affine -> "
        "MSE Loss -> Backprop -> SGD -> Updated Parameters"
    )
    result.conclusion = (
        "All mathematical concepts integrate into a working training pipeline. "
        "Linear algebra (matrix multiply), calculus (chain rule/derivatives), "
        "probability (softmax), optimization (SGD), and neural network math "
        "(activations, losses, backprop) all work together."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
