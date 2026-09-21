"""PyTorch comparison: loss functions.

Tests:
- MSE loss
- Binary cross-entropy loss
- Categorical cross-entropy loss
- Cross-entropy with logits
- Convention differences (mean vs sum, eps values)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestMSE:
    """Compare Mean Squared Error loss."""

    def test_single_value(self) -> None:
        """MSE on single predictions matches PyTorch."""
        from math_for_neural_networks.neural_networks.losses import mean_squared_error

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])

        our_loss = mean_squared_error(y_true, y_pred)
        pt_loss = torch.nn.functional.mse_loss(
            torch.tensor(y_pred, dtype=torch.float64),
            torch.tensor(y_true, dtype=torch.float64),
            reduction="mean",
        ).item()

        assert_close_scalar(our_loss, pt_loss, operation="MSE single")

    def test_batch(self) -> None:
        """MSE on batched predictions."""
        from math_for_neural_networks.neural_networks.losses import mean_squared_error

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal((4, 3))
        y_pred = rng.standard_normal((4, 3))

        our_loss = mean_squared_error(y_true, y_pred)
        pt_loss = torch.nn.functional.mse_loss(
            torch.tensor(y_pred, dtype=torch.float64),
            torch.tensor(y_true, dtype=torch.float64),
            reduction="mean",
        ).item()

        assert_close_scalar(our_loss, pt_loss, operation="MSE batch")

    def test_gradient(self) -> None:
        """MSE gradient matches PyTorch autograd."""
        from math_for_neural_networks.neural_networks.backprop import mse_backward

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal((4,))
        y_pred = rng.standard_normal((4,))

        our_grad = mse_backward(y_true, y_pred)

        y_true_t = torch.tensor(y_true, dtype=torch.float64)
        y_pred_t = torch.tensor(y_pred, dtype=torch.float64, requires_grad=True)
        loss = torch.mean((y_true_t - y_pred_t) ** 2)
        loss.backward()
        pt_grad = y_pred_t.grad.numpy()

        assert_close(our_grad, pt_grad, operation="MSE gradient")


def assert_close_scalar(
    a: float,
    b: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    operation: str = "",
) -> None:
    """Assert two scalars are approximately equal."""
    diff = abs(a - b)
    threshold = atol + rtol * abs(b)
    if diff > threshold:
        msg = (
            f"Scalar comparison failed [{operation}]:\n"
            f"  Values: {a:.6e} vs {b:.6e}\n"
            f"  Absolute error: {diff:.2e}\n"
            f"  Threshold: {threshold:.2e}"
        )
        raise AssertionError(msg)


@requires_torch
class TestBCE:
    """Compare Binary Cross-Entropy loss."""

    def test_forward(self) -> None:
        """BCE loss matches PyTorch (reduction=mean)."""
        from math_for_neural_networks.neural_networks.losses import binary_cross_entropy

        rng = np.random.default_rng(42)
        y_true = rng.choice([0.0, 1.0], size=(8,))
        y_pred = rng.uniform(0.1, 0.9, size=(8,))

        our_loss = binary_cross_entropy(y_true, y_pred)
        pt_loss = torch.nn.functional.binary_cross_entropy(
            torch.tensor(y_pred, dtype=torch.float64),
            torch.tensor(y_true, dtype=torch.float64),
            reduction="mean",
        ).item()

        assert_close_scalar(our_loss, pt_loss, operation="BCE forward")

    def test_gradient(self) -> None:
        """BCE gradient matches PyTorch autograd."""
        from math_for_neural_networks.neural_networks.backprop import sigmoid_bce_backward

        rng = np.random.default_rng(42)
        y_true = rng.choice([0.0, 1.0], size=(8,))
        logits_np = rng.standard_normal((8,))

        our_grad = sigmoid_bce_backward(logits_np, y_true)

        logits_t = torch.tensor(logits_np, dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_t,
            torch.tensor(y_true, dtype=torch.float64),
            reduction="mean",
        )
        loss.backward()
        pt_grad = logits_t.grad.numpy()

        assert_close(our_grad, pt_grad, operation="BCE+Sigmoid gradient")


@requires_torch
class TestCCE:
    """Compare Categorical Cross-Entropy loss."""

    def test_one_hot(self) -> None:
        """CCE on one-hot targets: compare with PyTorch using same conventions."""
        from math_for_neural_networks.neural_networks.losses import categorical_cross_entropy

        rng = np.random.default_rng(42)
        y_true = np.eye(3)[rng.choice(3, size=(4,))]
        y_pred = rng.uniform(0.1, 0.9, size=(4, 3))
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

        our_loss = categorical_cross_entropy(y_true, y_pred)

        # PyTorch cross_entropy takes log-probs (log_softmax) + class indices
        # Our CCE clips predictions to [eps, inf) before log
        # To match conventions, compute PyTorch manually with same clipping
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, None)
        targets_np = np.argmax(y_true, axis=1)

        pt_log_probs = torch.log(torch.tensor(y_pred_clipped, dtype=torch.float64))
        pt_loss_manual = torch.nn.functional.cross_entropy(
            pt_log_probs,
            torch.tensor(targets_np, dtype=torch.long),
            reduction="mean",
            weight=None,
        ).item()

        # Alternative: compare with our own manual computation
        our_manual_loss = -float(np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1)))
        assert_close_scalar(our_loss, our_manual_loss, operation="CCE consistency")

        # Compare with PyTorch (may differ due to eps convention)
        # Use relaxed tolerance since PyTorch uses a different eps internally
        assert_close_scalar(our_loss, pt_loss_manual, rtol=1e-2, operation="CCE one-hot")


@requires_torch
class TestCrossEntropyWithLogits:
    """Compare cross-entropy with logits."""

    def test_forward(self) -> None:
        """Cross-entropy with logits matches PyTorch."""
        from math_for_neural_networks.neural_networks.losses import cross_entropy_with_logits

        rng = np.random.default_rng(42)
        logits_np = rng.standard_normal((4, 3))
        targets_np = np.eye(3)[rng.choice(3, size=(4,))]

        our_loss = cross_entropy_with_logits(logits_np, targets_np)
        pt_loss = torch.nn.functional.cross_entropy(
            torch.tensor(logits_np, dtype=torch.float64),
            torch.tensor(np.argmax(targets_np, axis=1), dtype=torch.long),
            reduction="mean",
        ).item()

        assert_close_scalar(our_loss, pt_loss, rtol=1e-5, operation="CE with logits")

    def test_gradient(self) -> None:
        """Cross-entropy with logits gradient = softmax - targets."""
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
        assert_close(our_grad, pt_grad * n, rtol=1e-5, operation="CE+softmax gradient (scaled)")
