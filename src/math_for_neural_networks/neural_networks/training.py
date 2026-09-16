"""
Training Loop Mathematics
=========================

This module provides:
1. Simple neural network forward/backward pass
2. Training loop with optimizer integration
3. Gradient computation and parameter updates

The neural network is intentionally small and transparent:
    Input -> Linear -> Activation -> Linear -> Output

All operations use NumPy. No autograd is used for the neural network
(it uses manual backpropagation to show the mathematics clearly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.neural_networks.activations import relu, sigmoid
from math_for_neural_networks.neural_networks.losses import (
    binary_cross_entropy,
    mean_squared_error,
)


@dataclass
class TrainingMetrics:
    """Metrics recorded during training."""

    loss_history: list[float] = field(default_factory=list)
    gradient_norm_history: list[float] = field(default_factory=list)
    iterations: int = 0


@dataclass
class NetworkParams:
    """Parameters for a small two-layer neural network.

    Architecture:
        Input (input_dim) -> Linear -> Activation -> Linear -> Output (output_dim)

    Shapes:
        W1: (hidden_dim, input_dim)
        b1: (hidden_dim,)
        W2: (output_dim, hidden_dim)
        b2: (output_dim,)
    """

    W1: NDArray
    b1: NDArray
    W2: NDArray
    b2: NDArray

    def zero_grad(self) -> None:
        """Reset all gradients to zero. Not used in manual backprop
        but included for completeness."""
        pass

    def get_all_params(self) -> list[tuple[str, NDArray]]:
        """Return all parameters as (name, array) pairs."""
        return [
            ("W1", self.W1),
            ("b1", self.b1),
            ("W2", self.W2),
            ("b2", self.b2),
        ]


def init_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int | None = None,
) -> NetworkParams:
    """Initialize network parameters with simple random initialization.

    Uses small random values centered around 0.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((hidden_dim, input_dim)) * 0.5
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((output_dim, hidden_dim)) * 0.5
    b2 = np.zeros(output_dim)
    return NetworkParams(W1=W1, b1=b1, W2=W2, b2=b2)


def forward(
    x: NDArray,
    params: NetworkParams,
    activation: str = "sigmoid",
) -> tuple[NDArray, dict[str, NDArray]]:
    """Forward pass through a two-layer neural network.

    Architecture:
        z1 = W1 @ x + b1
        a1 = activation(z1)
        z2 = W2 @ a1 + b2
        output = z2 (regression) or softmax(z2) (classification)

    Args:
        x: Input, shape (input_dim,) or (batch_size, input_dim).
        params: Network parameters.
        activation: Activation function name ("sigmoid", "tanh", "relu").

    Returns:
        Tuple of (output, cache) where cache contains intermediate values
        needed for backpropagation.
    """
    x = np.asarray(x, dtype=np.float64)

    # Layer 1: affine + activation
    z1 = x @ params.W1.T + params.b1

    if activation == "sigmoid":
        a1 = sigmoid(z1)
    elif activation == "tanh":
        a1 = np.tanh(z1)
    elif activation == "relu":
        a1 = relu(z1)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Layer 2: affine
    z2 = a1 @ params.W2.T + params.b2

    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
    return z2, cache


def backward(
    y_true: NDArray,
    y_pred: NDArray,
    params: NetworkParams,
    cache: dict[str, NDArray],
    activation: str = "sigmoid",
    loss_type: str = "mse",
) -> dict[str, NDArray]:
    """Backward pass: compute gradients for all parameters.

    For regression (loss_type="mse"):
        Output is z2 (raw), loss is MSE.

    For classification (loss_type="bce"):
        Output is sigmoid(z2), loss is BCE.

    Gradient equations (batch-first, transposed convention):

        For Z2 = A1 @ W2^T + b2:
            dL/dW2 = dL/dZ2^T @ A1
            dL/db2 = sum(dL/dZ2, axis=0)
            dL/dA1 = dL/dZ2 @ W2

        For Z1 = X @ W1^T + b1:
            dL/dZ1 = dL/dA1 * activation'(Z1)
            dL/dW1 = dL/dZ1^T @ X
            dL/db1 = sum(dL/dZ1, axis=0)
            dL/dX = dL/dZ1 @ W1  (for input gradient, rarely needed)

    Args:
        y_true: True values.
        y_pred: Predicted values (output of forward pass).
        params: Network parameters.
        cache: Intermediate values from forward pass.
        activation: Activation function name.
        loss_type: Loss function type ("mse" or "bce").

    Returns:
        Dictionary of gradients: {"dW1", "db1", "dW2", "db2"}.
    """
    x = cache["x"]
    z1 = cache["z1"]
    a1 = cache["a1"]

    # Output layer gradient
    if loss_type == "mse":
        # L = (1/n) * sum((y_true - y_pred)^2)
        # dL/d(y_pred) = (2/n) * (y_pred - y_true)
        dout = (2.0 / y_true.size) * (y_pred - y_true)
    elif loss_type == "bce":
        # For sigmoid output: dL/dz2 = sigmoid(z2) - y_true
        probs = 1.0 / (1.0 + np.exp(-y_pred))
        dout = probs - y_true
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Layer 2 gradients: Z2 = A1 @ W2^T + b2
    dW2 = dout.T @ a1
    db2 = np.sum(dout, axis=0) if dout.ndim == 2 else dout.copy()
    dA1 = dout @ params.W2

    # Activation gradient
    if activation == "sigmoid":
        dz1 = dA1 * a1 * (1.0 - a1)
    elif activation == "tanh":
        dz1 = dA1 * (1.0 - a1**2)
    elif activation == "relu":
        dz1 = dA1 * (z1 > 0).astype(np.float64)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Layer 1 gradients: Z1 = X @ W1^T + b1
    dW1 = dz1.T @ x
    db1 = np.sum(dz1, axis=0) if dz1.ndim == 2 else dz1.copy()

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def sgd_step(
    params: NetworkParams,
    grads: dict[str, NDArray],
    lr: float,
) -> None:
    """Update parameters using stochastic gradient descent.

        theta = theta - lr * grad

    Args:
        params: Network parameters (modified in-place).
        grads: Gradients for each parameter.
        lr: Learning rate.
    """
    params.W1 -= lr * grads["dW1"]
    params.b1 -= lr * grads["db1"]
    params.W2 -= lr * grads["dW2"]
    params.b2 -= lr * grads["db2"]


def train(
    x: NDArray,
    y: NDArray,
    params: NetworkParams,
    lr: float = 0.01,
    iterations: int = 100,
    activation: str = "sigmoid",
    loss_type: str = "mse",
    record_metrics: bool = True,
) -> TrainingMetrics:
    """Train a two-layer neural network.

    Performs forward pass -> loss -> backward pass -> parameter update
    for the specified number of iterations.

    Args:
        x: Input data, shape (batch_size, input_dim).
        y: Target values, shape (batch_size, output_dim).
        params: Network parameters (modified in-place).
        lr: Learning rate.
        iterations: Number of training iterations.
        activation: Activation function name.
        loss_type: Loss function type.
        record_metrics: Whether to record loss and gradient norms.

    Returns:
        TrainingMetrics with loss history and gradient norms.
    """
    metrics = TrainingMetrics()

    for i in range(iterations):
        # Forward pass
        y_pred, cache = forward(x, params, activation)

        # Compute loss
        if loss_type == "mse":
            loss = mean_squared_error(y, y_pred)
        elif loss_type == "bce":
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            loss = binary_cross_entropy(y, probs)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Backward pass
        grads = backward(y, y_pred, params, cache, activation, loss_type)

        # Compute gradient norm
        grad_norm = float(
            np.sqrt(
                np.sum(grads["dW1"] ** 2)
                + np.sum(grads["db1"] ** 2)
                + np.sum(grads["dW2"] ** 2)
                + np.sum(grads["db2"] ** 2)
            )
        )

        # Parameter update
        sgd_step(params, grads, lr)

        # Record metrics
        if record_metrics:
            metrics.loss_history.append(loss)
            metrics.gradient_norm_history.append(grad_norm)

        metrics.iterations = i + 1

    return metrics


def compute_numerical_gradients(
    x: NDArray,
    y: NDArray,
    params: NetworkParams,
    activation: str = "sigmoid",
    loss_type: str = "mse",
    h: float = 1e-5,
) -> dict[str, NDArray]:
    """Compute numerical gradients for all network parameters.

    Uses central differences: dL/dtheta ≈ (L(theta+h) - L(theta-h)) / (2h)

    This is used for gradient checking against analytical gradients.

    Args:
        x: Input data.
        y: Target values.
        params: Network parameters.
        activation: Activation function name.
        loss_type: Loss function type.
        h: Finite-difference step size.

    Returns:
        Dictionary of numerical gradients matching the format of backward().
    """

    def compute_loss(p: NetworkParams) -> float:
        y_pred, _ = forward(x, p, activation)
        if loss_type == "mse":
            return mean_squared_error(y, y_pred)
        elif loss_type == "bce":
            probs = 1.0 / (1.0 + np.exp(-y_pred))
            return binary_cross_entropy(y, probs)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    num_grads: dict[str, NDArray] = {}

    for name, param in [("W1", params.W1), ("b1", params.b1), ("W2", params.W2), ("b2", params.b2)]:
        num_grad = np.zeros_like(param)
        flat_param = param.ravel()
        flat_num = num_grad.ravel()

        for i in range(flat_param.size):
            original = flat_param[i]

            flat_param[i] = original + h
            loss_plus = compute_loss(params)

            flat_param[i] = original - h
            loss_minus = compute_loss(params)

            flat_num[i] = (loss_plus - loss_minus) / (2.0 * h)

            flat_param[i] = original

        num_grads[name] = num_grad

    return num_grads
