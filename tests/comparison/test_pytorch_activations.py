"""PyTorch comparison: activation functions.

Compares our sigmoid, tanh, ReLU, GELU implementations against PyTorch equivalents.

Tests:
- Forward outputs
- Derivatives
- Numerical stability (large positive/negative inputs)
- Gradient of composed functions
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestSigmoid:
    """Compare sigmoid implementation."""

    def test_forward(self) -> None:
        """Sigmoid forward pass matches PyTorch."""
        from math_for_neural_networks.neural_networks.activations import sigmoid

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_out = sigmoid(x_np)
        pt_out = torch.sigmoid(torch.tensor(x_np, dtype=torch.float64)).numpy()

        assert_close(our_out, pt_out, operation="sigmoid forward")

    def test_large_positive(self) -> None:
        """Sigmoid handles large positive inputs without overflow."""
        from math_for_neural_networks.neural_networks.activations import sigmoid

        x_np = np.array([5.0, 10.0, 50.0, 100.0])
        our_out = sigmoid(x_np)
        pt_out = torch.sigmoid(torch.tensor(x_np, dtype=torch.float64)).numpy()

        assert_close(our_out, pt_out, rtol=1e-6, operation="sigmoid large positive")

    def test_large_negative(self) -> None:
        """Sigmoid handles large negative inputs without underflow."""
        from math_for_neural_networks.neural_networks.activations import sigmoid

        x_np = np.array([-5.0, -10.0, -50.0, -100.0])
        our_out = sigmoid(x_np)
        pt_out = torch.sigmoid(torch.tensor(x_np, dtype=torch.float64)).numpy()

        assert_close(our_out, pt_out, rtol=1e-6, operation="sigmoid large negative")

    def test_derivative(self) -> None:
        """Sigmoid derivative matches analytical formula."""
        from math_for_neural_networks.neural_networks.activations import sigmoid_derivative

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_deriv = sigmoid_derivative(x_np)

        # PyTorch autograd reference
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.sigmoid(x_t)
        y_t.sum().backward()
        pt_deriv = x_t.grad.numpy()

        assert_close(our_deriv, pt_deriv, operation="sigmoid derivative")


@requires_torch
class TestTanh:
    """Compare tanh implementation."""

    def test_forward(self) -> None:
        """Tanh forward pass matches PyTorch."""
        from math_for_neural_networks.neural_networks.activations import tanh

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_out = tanh(x_np)
        pt_out = torch.tanh(torch.tensor(x_np, dtype=torch.float64)).numpy()

        assert_close(our_out, pt_out, operation="tanh forward")

    def test_derivative(self) -> None:
        """Tanh derivative matches PyTorch autograd."""
        from math_for_neural_networks.neural_networks.activations import tanh_derivative

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_deriv = tanh_derivative(x_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.tanh(x_t)
        y_t.sum().backward()
        pt_deriv = x_t.grad.numpy()

        assert_close(our_deriv, pt_deriv, operation="tanh derivative")


@requires_torch
class TestReLU:
    """Compare ReLU implementation."""

    def test_forward(self) -> None:
        """ReLU forward pass matches PyTorch."""
        from math_for_neural_networks.neural_networks.activations import relu

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_out = relu(x_np)
        pt_out = torch.relu(torch.tensor(x_np, dtype=torch.float64)).numpy()

        assert_close(our_out, pt_out, operation="relu forward")

    def test_derivative(self) -> None:
        """ReLU derivative matches PyTorch autograd."""
        from math_for_neural_networks.neural_networks.activations import relu_derivative

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_deriv = relu_derivative(x_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.relu(x_t)
        y_t.sum().backward()
        pt_deriv = x_t.grad.numpy()

        assert_close(our_deriv, pt_deriv, operation="relu derivative")


@requires_torch
class TestGELU:
    """Compare GELU implementation."""

    def test_forward(self) -> None:
        """GELU forward pass matches PyTorch (approximate mode)."""
        from math_for_neural_networks.neural_networks.activations import gelu

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_out = gelu(x_np)

        # PyTorch GELU approximate=True uses the same tanh approximation
        pt_out = torch.nn.functional.gelu(
            torch.tensor(x_np, dtype=torch.float64), approximate="tanh"
        ).numpy()

        assert_close(our_out, pt_out, rtol=1e-5, atol=1e-7, operation="GELU forward")

    def test_derivative(self) -> None:
        """GELU derivative matches PyTorch autograd."""
        from math_for_neural_networks.neural_networks.activations import gelu_derivative

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((10,))

        our_deriv = gelu_derivative(x_np)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        y_t = torch.nn.functional.gelu(x_t, approximate="tanh")
        y_t.sum().backward()
        pt_deriv = x_t.grad.numpy()

        assert_close(our_deriv, pt_deriv, rtol=1e-5, atol=1e-6, operation="GELU derivative")
