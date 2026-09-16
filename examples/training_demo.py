"""
Training Demo
=============

Demonstrates training a simple neural network on a toy dataset.

Features:
1. Forward pass through the network
2. Loss computation
3. Backward pass (backpropagation)
4. Parameter updates (SGD)
5. Loss reduction over time
"""

import numpy as np

from math_for_neural_networks.neural_networks.training import (
    forward,
    backward,
    sgd_step,
    init_params,
    train,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def example_1_forward_backward() -> None:
    """Demonstrate forward and backward pass on a small batch."""
    print_section("Example 1: Forward and Backward Pass")

    params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[1.0], [0.0]])

    print("  Network: 2 inputs -> 3 hidden (sigmoid) -> 1 output")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    # Forward
    y_pred, cache = forward(x, params, "sigmoid")
    print(f"\n  Forward pass:")
    print(f"    z1 = W1 @ x + b1 shape: {cache['z1'].shape}")
    print(f"    a1 = sigmoid(z1) shape: {cache['a1'].shape}")
    print(f"    z2 = W2 @ a1 + b2 shape: {y_pred.shape}")
    print(f"    Predictions: {y_pred.flatten()}")

    # Backward
    grads = backward(y, y_pred, params, cache, "sigmoid", "mse")
    print(f"\n  Backward pass:")
    for name, grad in grads.items():
        print(f"    {name} shape: {grad.shape}, norm: {np.linalg.norm(grad):.6f}")


def example_2_training_loop() -> None:
    """Train a network and observe loss reduction."""
    print_section("Example 2: Training Loop")

    # Generate simple dataset: y = (x1 > 0)
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((50, 2))
    y_train = (x_train[:, 0:1] > 0).astype(float)

    print("  Dataset: 50 samples, 2 features")
    print("  Task: Binary classification (y = 1 if x1 > 0)")

    params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)

    print(f"\n  Initial parameters:")
    print(f"    W1 shape: {params.W1.shape}")
    print(f"    W2 shape: {params.W2.shape}")

    # Training
    metrics = train(
        x_train, y_train, params,
        lr=0.5, iterations=200,
        activation="sigmoid", loss_type="mse"
    )

    print(f"\n  Training results:")
    print(f"    Initial loss: {metrics.loss_history[0]:.6f}")
    print(f"    Final loss: {metrics.loss_history[-1]:.6f}")
    print(f"    Loss reduction: {metrics.loss_history[0] - metrics.loss_history[-1]:.6f}")
    print(f"    Final gradient norm: {metrics.gradient_norm_history[-1]:.6f}")

    # Predictions
    y_pred, _ = forward(x_train, params, "sigmoid")
    accuracy = np.mean((y_pred > 0.5) == y_train)
    print(f"\n    Training accuracy: {accuracy:.2%}")

    print(f"\n  Loss history (first 10 iterations):")
    for i in range(min(10, len(metrics.loss_history))):
        print(f"    Iter {i:>3}: loss = {metrics.loss_history[i]:.6f}, "
              f"grad_norm = {metrics.gradient_norm_history[i]:.6f}")

    print(f"\n  Loss history (last 10 iterations):")
    for i in range(max(0, len(metrics.loss_history) - 10), len(metrics.loss_history)):
        print(f"    Iter {i:>3}: loss = {metrics.loss_history[i]:.6f}, "
              f"grad_norm = {metrics.gradient_norm_history[i]:.6f}")


def example_3_activation_comparison() -> None:
    """Compare different activation functions."""
    print_section("Example 3: Activation Comparison")

    rng = np.random.default_rng(42)
    x = rng.standard_normal((30, 2))
    y = (x[:, 0:1] * x[:, 1:2] > 0).astype(float)

    print("  Dataset: 30 samples, 2 features")
    print("  Task: y = 1 if x1 * x2 > 0")

    for act in ["sigmoid", "tanh", "relu"]:
        params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
        metrics = train(x, y, params, lr=0.1, iterations=200, activation=act, loss_type="mse")

        y_pred, _ = forward(x, params, act)
        accuracy = np.mean((y_pred > 0.5) == y)

        print(f"\n  {act:>8}: initial_loss={metrics.loss_history[0]:.4f}, "
              f"final_loss={metrics.loss_history[-1]:.4f}, "
              f"accuracy={accuracy:.2%}")


def example_4_learning_rate_effect() -> None:
    """Demonstrate the effect of learning rate."""
    print_section("Example 4: Learning Rate Effect")

    rng = np.random.default_rng(42)
    x = rng.standard_normal((30, 2))
    y = (x[:, 0:1] > 0).astype(float)

    print("  Dataset: 30 samples, 2 features")
    print("  Task: Binary classification")

    for lr in [0.01, 0.1, 0.5, 1.0]:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        metrics = train(x, y, params, lr=lr, iterations=100, activation="sigmoid", loss_type="mse")

        print(f"\n  lr={lr:>4}: initial={metrics.loss_history[0]:.4f}, "
              f"final={metrics.loss_history[-1]:.4f}, "
              f"grad_norm={metrics.gradient_norm_history[-1]:.6f}")


def example_5_gradient_norm() -> None:
    """Track gradient norm during training."""
    print_section("Example 5: Gradient Norm During Training")

    rng = np.random.default_rng(42)
    x = rng.standard_normal((40, 2))
    y = (x[:, 0:1] > 0).astype(float)

    params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
    metrics = train(x, y, params, lr=0.5, iterations=100, activation="sigmoid", loss_type="mse")

    print("  Gradient norm over training:")
    print(f"  {'Iter':>5} | {'Loss':>10} | {'Grad Norm':>10}")
    print("  " + "-" * 30)

    for i in range(0, len(metrics.loss_history), 10):
        print(f"  {i:>5} | {metrics.loss_history[i]:>10.6f} | "
              f"{metrics.gradient_norm_history[i]:>10.6f}")

    # Final
    i = len(metrics.loss_history) - 1
    print(f"  {i:>5} | {metrics.loss_history[i]:>10.6f} | "
          f"{metrics.gradient_norm_history[i]:>10.6f}")


if __name__ == "__main__":
    print("TRAINING DEMO")
    print("=" * 60)

    example_1_forward_backward()
    example_2_training_loop()
    example_3_activation_comparison()
    example_4_learning_rate_effect()
    example_5_gradient_norm()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Forward pass: input -> layers -> output -> loss
    2. Backward pass: loss -> gradients -> parameters
    3. Training loop: forward -> loss -> backward -> update
    4. Loss decreases as parameters are updated
    5. Gradient norm decreases as network converges
    6. Learning rate affects convergence speed and stability
    7. Different activations have different training characteristics
    """)
