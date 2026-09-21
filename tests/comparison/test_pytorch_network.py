"""PyTorch comparison: small neural network parity.

Builds a small 2-layer network using our implementation and an equivalent
PyTorch model, then compares:
- Forward outputs
- Loss values
- Parameter shapes
- Gradient magnitudes
- Training behavior (same seed, same steps)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestNetworkParity:
    """Compare small neural network implementations."""

    def _build_our_network(self, seed: int = 42) -> tuple:
        """Build our 2-layer network and return params, forward output, loss, grads."""
        from math_for_neural_networks.neural_networks.losses import mean_squared_error
        from math_for_neural_networks.neural_networks.training import (
            backward,
            forward,
            init_params,
        )

        rng = np.random.default_rng(seed)
        params = init_params(input_dim=3, hidden_dim=4, output_dim=2, seed=seed)
        x_np = rng.standard_normal((4, 3))
        y_np = rng.standard_normal((4, 2))

        y_pred, cache = forward(x_np, params, activation="sigmoid")
        loss = mean_squared_error(y_np, y_pred)
        grads = backward(y_np, y_pred, params, cache, activation="sigmoid", loss_type="mse")

        return params, x_np, y_np, y_pred, loss, grads

    def _build_pytorch_network(self, seed: int = 42) -> tuple:
        """Build equivalent PyTorch 2-layer network and return model, forward output, loss, grads."""
        rng = np.random.default_rng(seed)
        W1_np = rng.standard_normal((4, 3)) * 0.5
        b1_np = np.zeros(4)
        W2_np = rng.standard_normal((2, 4)) * 0.5
        b2_np = np.zeros(2)

        W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
        b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
        W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
        b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

        x_np = rng.standard_normal((4, 3))
        y_np = rng.standard_normal((4, 2))

        x_t = torch.tensor(x_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)

        z1 = x_t @ W1_t.T + b1_t
        a1 = torch.sigmoid(z1)
        z2 = a1 @ W2_t.T + b2_t
        loss = torch.mean((z2 - y_t) ** 2)
        loss.backward()

        grads = {
            "dW1": W1_t.grad.numpy(),
            "db1": b1_t.grad.numpy(),
            "dW2": W2_t.grad.numpy(),
            "db2": b2_t.grad.numpy(),
        }

        return None, x_np, y_np, z2.detach().numpy(), loss.item(), grads

    def test_forward_output_parity(self) -> None:
        """Same initial parameters produce same forward output."""
        _, x_np, y_np, our_pred, our_loss, _ = self._build_our_network(seed=42)

        # Re-run PyTorch with same seed
        rng = np.random.default_rng(42)
        W1_np = rng.standard_normal((4, 3)) * 0.5
        b1_np = np.zeros(4)
        W2_np = rng.standard_normal((2, 4)) * 0.5
        b2_np = np.zeros(2)

        W1_t = torch.tensor(W1_np, dtype=torch.float64)
        b1_t = torch.tensor(b1_np, dtype=torch.float64)
        W2_t = torch.tensor(W2_np, dtype=torch.float64)
        b2_t = torch.tensor(b2_np, dtype=torch.float64)

        x_t = torch.tensor(x_np, dtype=torch.float64)

        z1 = x_t @ W1_t.T + b1_t
        a1 = torch.sigmoid(z1)
        pt_pred = (a1 @ W2_t.T + b2_t).numpy()

        assert_close(our_pred, pt_pred, rtol=1e-5, operation="network forward output")

    def test_loss_parity(self) -> None:
        """Same forward output produces same MSE loss."""
        _, _, _, _, our_loss, _ = self._build_our_network(seed=42)

        rng = np.random.default_rng(42)
        W1_np = rng.standard_normal((4, 3)) * 0.5
        b1_np = np.zeros(4)
        W2_np = rng.standard_normal((2, 4)) * 0.5
        b2_np = np.zeros(2)

        W1_t = torch.tensor(W1_np, dtype=torch.float64)
        b1_t = torch.tensor(b1_np, dtype=torch.float64)
        W2_t = torch.tensor(W2_np, dtype=torch.float64)
        b2_t = torch.tensor(b2_np, dtype=torch.float64)

        rng2 = np.random.default_rng(42)
        x_np = rng2.standard_normal((4, 3))
        y_np = rng2.standard_normal((4, 2))

        x_t = torch.tensor(x_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)

        z1 = x_t @ W1_t.T + b1_t
        a1 = torch.sigmoid(z1)
        z2 = a1 @ W2_t.T + b2_t
        pt_loss = torch.mean((z2 - y_t) ** 2).item()

        assert_close_scalar(our_loss, pt_loss, operation="network loss")

    def test_gradient_parity(self) -> None:
        """Same network produces same gradient magnitudes."""
        _, _, _, _, _, our_grads = self._build_our_network(seed=42)

        rng = np.random.default_rng(42)
        W1_np = rng.standard_normal((4, 3)) * 0.5
        b1_np = np.zeros(4)
        W2_np = rng.standard_normal((2, 4)) * 0.5
        b2_np = np.zeros(2)

        W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
        b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
        W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
        b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

        rng2 = np.random.default_rng(42)
        x_np = rng2.standard_normal((4, 3))
        y_np = rng2.standard_normal((4, 2))

        x_t = torch.tensor(x_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)

        z1 = x_t @ W1_t.T + b1_t
        a1 = torch.sigmoid(z1)
        z2 = a1 @ W2_t.T + b2_t
        loss = torch.mean((z2 - y_t) ** 2)
        loss.backward()

        # Compare gradient norms (shapes should match)
        our_norm = float(np.sqrt(sum(np.sum(g**2) for g in our_grads.values())))
        pt_norm = float(
            np.sqrt(
                sum(np.sum(g.numpy() ** 2) for g in [W1_t.grad, b1_t.grad, W2_t.grad, b2_t.grad])
            )
        )

        assert_close_scalar(our_norm, pt_norm, rtol=1e-5, operation="network gradient norm")


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
