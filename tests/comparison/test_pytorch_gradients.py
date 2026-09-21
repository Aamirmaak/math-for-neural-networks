"""PyTorch comparison: gradient computations.

This is one of the most important comparison tests.
Verifies our analytical gradients against PyTorch autograd as a reference.

Tests:
- Scalar function gradients
- Activation derivative gradients
- Affine layer gradients (single + batch)
- MSE gradient
- Softmax + cross-entropy gradient identity
- Sigmoid + BCE gradient identity
- Small neural network gradient (end-to-end)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestAffineGradients:
    """Compare affine layer gradients."""

    def test_single_sample(self) -> None:
        """Affine gradient for single sample."""
        from math_for_neural_networks.neural_networks.backprop import affine_backward

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        W_np = rng.standard_normal((2, 3))
        b_np = rng.standard_normal((2,))
        dout_np = rng.standard_normal((2,))

        our_dx, our_dW, our_db = affine_backward(x_np, W_np, b_np, dout_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        W_t = torch.tensor(W_np, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)

        out_t = x_t @ W_t.T + b_t
        loss = torch.sum(out_t * torch.tensor(dout_np, dtype=torch.float64))
        loss.backward()

        assert_close(our_dx, x_t.grad.numpy(), operation="affine dx single")
        assert_close(our_dW, W_t.grad.numpy(), operation="affine dW single")
        assert_close(our_db, b_t.grad.numpy(), operation="affine db single")

    def test_batch(self) -> None:
        """Affine gradient for batched input."""
        from math_for_neural_networks.neural_networks.backprop import affine_backward

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 3))
        W_np = rng.standard_normal((2, 3))
        b_np = rng.standard_normal((2,))
        dout_np = rng.standard_normal((4, 2))

        our_dx, our_dW, our_db = affine_backward(x_np, W_np, b_np, dout_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        W_t = torch.tensor(W_np, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)

        out_t = x_t @ W_t.T + b_t
        loss = torch.sum(out_t * torch.tensor(dout_np, dtype=torch.float64))
        loss.backward()

        assert_close(our_dx, x_t.grad.numpy(), operation="affine dx batch")
        assert_close(our_dW, W_t.grad.numpy(), operation="affine dW batch")
        assert_close(our_db, b_t.grad.numpy(), operation="affine db batch")


@requires_torch
class TestActivationGradients:
    """Compare activation function backward gradients."""

    def test_sigmoid_backward(self) -> None:
        """Sigmoid backward gradient."""
        from math_for_neural_networks.neural_networks.backprop import sigmoid_backward

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((5,))
        dout_np = rng.standard_normal((5,))

        our_dx = sigmoid_backward(x_np, dout_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.sigmoid(x_t)
        loss = torch.sum(y_t * torch.tensor(dout_np, dtype=torch.float64))
        loss.backward()

        assert_close(our_dx, x_t.grad.numpy(), operation="sigmoid backward")

    def test_relu_backward(self) -> None:
        """ReLU backward gradient."""
        from math_for_neural_networks.neural_networks.backprop import relu_backward

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((5,))
        dout_np = rng.standard_normal((5,))

        our_dx = relu_backward(x_np, dout_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.relu(x_t)
        loss = torch.sum(y_t * torch.tensor(dout_np, dtype=torch.float64))
        loss.backward()

        assert_close(our_dx, x_t.grad.numpy(), operation="relu backward")

    def test_tanh_backward(self) -> None:
        """Tanh backward gradient."""
        from math_for_neural_networks.neural_networks.backprop import tanh_backward

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((5,))
        dout_np = rng.standard_normal((5,))

        our_dx = tanh_backward(x_np, dout_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.tanh(x_t)
        loss = torch.sum(y_t * torch.tensor(dout_np, dtype=torch.float64))
        loss.backward()

        assert_close(our_dx, x_t.grad.numpy(), operation="tanh backward")


@requires_torch
class TestSoftmaxCEGradient:
    """Compare softmax + cross-entropy gradient identity."""

    def test_gradient_is_p_minus_y(self) -> None:
        """dL/dz = softmax(z) - y for cross-entropy loss."""
        from math_for_neural_networks.neural_networks.backprop import softmax_backward

        rng = np.random.default_rng(42)
        logits_np = rng.standard_normal((4, 3))
        targets_np = np.eye(3)[rng.choice(3, size=(4,))]

        our_grad = softmax_backward(logits_np, targets_np)

        logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(
            logits_t,
            torch.tensor(np.argmax(targets_np, axis=1), dtype=torch.long),
            reduction="mean",
        )
        loss.backward()
        pt_grad = logits_t.grad.numpy()

        # Our softmax_backward returns (probs - targets) without 1/n factor.
        # PyTorch cross_entropy(reduction="mean") includes 1/n in its gradient.
        n = logits_np.shape[0]
        assert_close(our_grad, pt_grad * n, operation="softmax+CE gradient (scaled)")

    def test_gradient_sums_to_zero(self) -> None:
        """Gradient of CE w.r.t. logits sums to ~0 per sample (for one-hot targets)."""
        from math_for_neural_networks.neural_networks.backprop import softmax_backward

        rng = np.random.default_rng(42)
        logits_np = rng.standard_normal((4, 3))
        targets_np = np.eye(3)[rng.choice(3, size=(4,))]

        grad = softmax_backward(logits_np, targets_np)

        # Each row should sum to approximately 0 (softmax sums to 1, one-hot sums to 1)
        row_sums = np.sum(grad, axis=-1)
        assert_close(row_sums, np.zeros(4), atol=1e-10, operation="softmax+CE sum-to-zero")


@requires_torch
class TestSigmoidBCEGradient:
    """Compare sigmoid + BCE gradient identity."""

    def test_gradient_identity(self) -> None:
        """dL/dz = (sigma(z) - y) / n for sigmoid + BCE."""
        from math_for_neural_networks.neural_networks.backprop import sigmoid_bce_backward

        rng = np.random.default_rng(42)
        logits_np = rng.standard_normal((8,))
        targets_np = rng.choice([0.0, 1.0], size=(8,))

        our_grad = sigmoid_bce_backward(logits_np, targets_np)

        logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_t,
            torch.tensor(targets_np, dtype=torch.float64),
            reduction="mean",
        )
        loss.backward()
        pt_grad = logits_t.grad.numpy()

        assert_close(our_grad, pt_grad, operation="sigmoid+BCE gradient")


@requires_torch
class TestNeuralNetworkGradients:
    """Compare gradients of a small neural network end-to-end."""

    def test_two_layer_network(self) -> None:
        """Compare gradients of our 2-layer network against PyTorch autograd."""
        from math_for_neural_networks.neural_networks.training import (
            backward,
            forward,
            init_params,
        )

        rng = np.random.default_rng(42)
        params = init_params(input_dim=3, hidden_dim=4, output_dim=2, seed=42)
        x_np = rng.standard_normal((4, 3))
        y_np = rng.standard_normal((4, 2))

        # Our forward + backward
        y_pred, cache = forward(x_np, params, activation="sigmoid")
        our_grads = backward(y_np, y_pred, params, cache, activation="sigmoid", loss_type="mse")

        # PyTorch equivalent
        W1_t = torch.tensor(params.W1, dtype=torch.float64, requires_grad=True)
        b1_t = torch.tensor(params.b1, dtype=torch.float64, requires_grad=True)
        W2_t = torch.tensor(params.W2, dtype=torch.float64, requires_grad=True)
        b2_t = torch.tensor(params.b2, dtype=torch.float64, requires_grad=True)

        x_t = torch.tensor(x_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)

        z1_t = x_t @ W1_t.T + b1_t
        a1_t = torch.sigmoid(z1_t)
        z2_t = a1_t @ W2_t.T + b2_t
        loss_t = torch.mean((z2_t - y_t) ** 2)
        loss_t.backward()

        # Compare gradients
        assert_close(our_grads["dW1"], W1_t.grad.numpy(), rtol=1e-5, operation="network dW1")
        assert_close(our_grads["db1"], b1_t.grad.numpy(), rtol=1e-5, operation="network db1")
        assert_close(our_grads["dW2"], W2_t.grad.numpy(), rtol=1e-5, operation="network dW2")
        assert_close(our_grads["db2"], b2_t.grad.numpy(), rtol=1e-5, operation="network db2")
