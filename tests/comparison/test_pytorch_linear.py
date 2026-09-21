"""PyTorch comparison: linear/affine transformations.

Compares our affine_transform implementation against PyTorch's nn.Linear
and explicit matrix operations.

Tests:
- Single sample forward pass
- Batched forward pass
- Weight and bias shapes
- Numerical parity (float64)
- Numerical parity (float32)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestAffineSingleSample:
    """Compare affine transform on single samples (1D input)."""

    def test_no_bias(self) -> None:
        """z = W @ x (no bias)."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        W_np = rng.standard_normal((2, 3))

        x_t = torch.tensor(x_np, dtype=torch.float64)
        W_t = torch.tensor(W_np, dtype=torch.float64)

        our_out = affine_transform(x_np, W_np, b=None)
        pt_out = (W_t @ x_t).numpy()

        assert_close(our_out, pt_out, operation="affine single no-bias")

    def test_with_bias(self) -> None:
        """z = W @ x + b."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        W_np = rng.standard_normal((2, 3))
        b_np = rng.standard_normal((2,))

        x_t = torch.tensor(x_np, dtype=torch.float64)
        W_t = torch.tensor(W_np, dtype=torch.float64)
        b_t = torch.tensor(b_np, dtype=torch.float64)

        our_out = affine_transform(x_np, W_np, b=b_np)
        pt_out = (W_t @ x_t + b_t).numpy()

        assert_close(our_out, pt_out, operation="affine single with-bias")


@requires_torch
class TestAffineBatch:
    """Compare affine transform on batched inputs (2D input)."""

    def test_no_bias(self) -> None:
        """Z = X @ W^T (no bias, batch)."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((4, 3))
        W_np = rng.standard_normal((2, 3))

        X_t = torch.tensor(X_np, dtype=torch.float64)
        W_t = torch.tensor(W_np, dtype=torch.float64)

        our_out = affine_transform(X_np, W_np, b=None)
        pt_out = (X_t @ W_t.T).numpy()

        assert_close(our_out, pt_out, operation="affine batch no-bias")

    def test_with_bias(self) -> None:
        """Z = X @ W^T + b (batch)."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        X_np = rng.standard_normal((4, 3))
        W_np = rng.standard_normal((2, 3))
        b_np = rng.standard_normal((2,))

        X_t = torch.tensor(X_np, dtype=torch.float64)
        W_t = torch.tensor(W_np, dtype=torch.float64)
        b_t = torch.tensor(b_np, dtype=torch.float64)

        our_out = affine_transform(X_np, W_np, b=b_np)
        pt_out = (X_t @ W_t.T + b_t).numpy()

        assert_close(our_out, pt_out, operation="affine batch with-bias")

    def test_nn_linear_equivalence(self) -> None:
        """Our affine_transform produces same output as nn.Linear."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        in_features, out_features = 5, 3
        W_np = rng.standard_normal((out_features, in_features))
        b_np = rng.standard_normal((out_features,))
        X_np = rng.standard_normal((8, in_features))

        linear = torch.nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(W_np, dtype=torch.float32))
            linear.bias.copy_(torch.tensor(b_np, dtype=torch.float32))

        our_out = affine_transform(X_np, W_np, b=b_np)

        X_t = torch.tensor(X_np, dtype=torch.float32)
        pt_out = linear(X_t).detach().numpy()

        assert_close(our_out, pt_out, rtol=1e-4, atol=1e-5, operation="nn.Linear equivalence")


@requires_torch
class TestAffineFloat32:
    """Test numerical behavior in float32."""

    def test_float32_output(self) -> None:
        """Our implementation always outputs float64; compare against PyTorch float32."""
        from math_for_neural_networks.neural_networks.layers import affine_transform

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        W_np = rng.standard_normal((2, 3))
        b_np = rng.standard_normal((2,))

        x_t = torch.tensor(x_np, dtype=torch.float32)
        W_t = torch.tensor(W_np, dtype=torch.float32)
        b_t = torch.tensor(b_np, dtype=torch.float32)

        our_out = affine_transform(x_np, W_np, b=b_np)
        pt_out = (W_t @ x_t + b_t).double().numpy()

        assert_close(our_out, pt_out, rtol=1e-5, atol=1e-6, operation="affine float32")
