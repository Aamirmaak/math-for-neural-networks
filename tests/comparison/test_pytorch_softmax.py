"""PyTorch comparison: softmax and log-softmax.

Tests:
- Standard softmax
- Large positive logits (stability)
- Large negative logits
- Batched inputs
- Sum of probabilities equals 1
- Log-softmax numerical stability
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestSoftmax:
    """Compare softmax implementation."""

    def test_standard(self) -> None:
        """Softmax on standard inputs matches PyTorch."""
        from math_for_neural_networks.neural_networks.attention import softmax

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((5,))

        our_out = softmax(x_np)
        pt_out = torch.softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_out, pt_out, operation="softmax standard")

    def test_large_positive(self) -> None:
        """Softmax handles large positive logits."""
        from math_for_neural_networks.neural_networks.attention import softmax

        x_np = np.array([100.0, 101.0, 102.0, 103.0])
        our_out = softmax(x_np)
        pt_out = torch.softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_out, pt_out, rtol=1e-6, operation="softmax large positive")

    def test_large_negative(self) -> None:
        """Softmax handles large negative logits."""
        from math_for_neural_networks.neural_networks.attention import softmax

        x_np = np.array([-100.0, -101.0, -102.0, -103.0])
        our_out = softmax(x_np)
        pt_out = torch.softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_out, pt_out, rtol=1e-6, operation="softmax large negative")

    def test_batched(self) -> None:
        """Softmax on batched inputs."""
        from math_for_neural_networks.neural_networks.attention import softmax

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 5))

        our_out = softmax(x_np)
        pt_out = torch.softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_out, pt_out, operation="softmax batched")

    def test_sums_to_one(self) -> None:
        """Softmax probabilities sum to 1 along the correct axis."""
        from math_for_neural_networks.neural_networks.attention import softmax

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 5))

        our_out = softmax(x_np)

        assert our_out.ndim == 2
        sums = np.sum(our_out, axis=-1)
        assert_close(sums, np.ones(4), rtol=1e-6, operation="softmax sum=1")

    def test_batch_axis1(self) -> None:
        """Softmax along axis=1 (different axis)."""
        from math_for_neural_networks.neural_networks.attention import softmax

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4))

        our_out = softmax(x_np, axis=1)
        pt_out = torch.softmax(torch.tensor(x_np, dtype=torch.float64), dim=1).numpy()

        assert_close(our_out, pt_out, operation="softmax axis=1")


@requires_torch
class TestLogSoftmax:
    """Compare log-softmax behavior."""

    def test_log_softmax_numerical_stability(self) -> None:
        """log(softmax(x)) matches log_softmax(x) for extreme values."""
        x_np = np.array([1000.0, 1001.0, 1002.0])

        # Our implementation: log(softmax)
        from math_for_neural_networks.neural_networks.attention import softmax

        our_log_probs = np.log(softmax(x_np))

        pt_log_probs = torch.log_softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_log_probs, pt_log_probs, rtol=1e-5, operation="log_softmax stability")

    def test_log_softmax_negative_stability(self) -> None:
        """log(softmax(x)) for large negative values."""
        x_np = np.array([-1000.0, -1001.0, -1002.0])

        from math_for_neural_networks.neural_networks.attention import softmax

        our_log_probs = np.log(softmax(x_np))

        pt_log_probs = torch.log_softmax(torch.tensor(x_np, dtype=torch.float64), dim=-1).numpy()

        assert_close(our_log_probs, pt_log_probs, rtol=1e-5, operation="log_softmax neg")
