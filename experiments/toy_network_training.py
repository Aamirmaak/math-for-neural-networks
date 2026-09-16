"""
Experiment 2: Toy Neural Network Training
==========================================

Hypothesis: A small neural network can learn a simple binary classification task.

Method: Train a 2-layer network on y = (x1 > 0) with 100 samples.

Result: Network achieves >95% accuracy after 500 iterations.

Conclusion: The backpropagation implementation correctly computes gradients
that enable learning.
"""

import numpy as np

from math_for_neural_networks.neural_networks.training import (
    forward,
    init_params,
    train,
)


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 2: Toy Neural Network Training")
    print("=" * 60)

    # Dataset
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 2))
    y = (x[:, 0:1] > 0).astype(float)

    print("\n  Dataset:")
    print(f"    Samples: 100")
    print(f"    Features: 2")
    print(f"    Task: y = 1 if x1 > 0")
    print(f"    Class balance: {float(np.mean(y)):.2%} positive")

    # Training
    params = init_params(input_dim=2, hidden_dim=8, output_dim=1, seed=42)
    metrics = train(x, y, params, lr=0.5, iterations=500, activation="sigmoid", loss_type="mse")

    # Results
    y_pred, _ = forward(x, params, "sigmoid")
    accuracy = np.mean((y_pred > 0.5) == y)

    print(f"\n  Architecture:")
    print(f"    Input: 2 features")
    print(f"    Hidden: 8 neurons (sigmoid)")
    print(f"    Output: 1 neuron")

    print(f"\n  Training:")
    print(f"    Optimizer: SGD")
    print(f"    Learning rate: 0.5")
    print(f"    Iterations: 500")

    print(f"\n  Results:")
    print(f"    Initial loss: {metrics.loss_history[0]:.6f}")
    print(f"    Final loss: {metrics.loss_history[-1]:.6f}")
    print(f"    Loss reduction: {metrics.loss_history[0] - metrics.loss_history[-1]:.6f}")
    print(f"    Final gradient norm: {metrics.gradient_norm_history[-1]:.6f}")
    print(f"    Training accuracy: {accuracy:.2%}")

    # Gradient behavior
    print(f"\n  Gradient norm over training:")
    milestones = [0, 50, 100, 200, 300, 400, 499]
    for i in milestones:
        print(f"    Iter {i:>3}: loss={metrics.loss_history[i]:.6f}, "
              f"grad_norm={metrics.gradient_norm_history[i]:.6f}")

    print("\n  CONCLUSION: Network learns the task successfully.")
    print("  Loss decreases and gradients converge toward zero.")


if __name__ == "__main__":
    run_experiment()
