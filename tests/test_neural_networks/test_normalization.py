"""Tests for layer normalization."""

from __future__ import annotations

import numpy as np
import pytest

from math_for_neural_networks.neural_networks.normalization import layer_norm, layer_norm_stats


class TestLayerNorm:
    """Tests for layer normalization."""

    def test_output_shape_1d(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = layer_norm(x)
        assert y.shape == x.shape

    def test_output_shape_2d(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = layer_norm(x)
        assert y.shape == x.shape

    def test_zero_mean(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = layer_norm(x)
        assert np.mean(y) == pytest.approx(0.0, abs=1e-10)

    def test_unit_variance(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = layer_norm(x)
        # layer_norm uses population variance (divides by n), not sample variance (n-1)
        assert np.var(y, ddof=0) == pytest.approx(1.0, abs=1e-4)

    def test_batch_zero_mean(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        y = layer_norm(x)
        for i in range(2):
            assert np.mean(y[i]) == pytest.approx(0.0, abs=1e-10)

    def test_with_gamma_beta(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])
        y = layer_norm(x, gamma=gamma, beta=beta)
        assert np.mean(y) == pytest.approx(1.0, abs=1e-10)

    def test_constant_input(self) -> None:
        x = np.array([5.0, 5.0, 5.0])
        y = layer_norm(x)
        np.testing.assert_array_almost_equal(y, [0.0, 0.0, 0.0])

    def test_x_not_1d_or_2d(self) -> None:
        x = np.array([[[1.0, 2.0]]])
        with pytest.raises(ValueError, match="x must be 1D or 2D"):
            layer_norm(x)


class TestLayerNormStats:
    """Tests for layer norm statistics."""

    def test_basic(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = layer_norm_stats(x)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["variance"] == pytest.approx(2.0)

    def test_normalized_zero_mean(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = layer_norm_stats(x)
        assert np.mean(stats["x_hat"]) == pytest.approx(0.0, abs=1e-10)

    def test_normalized_unit_variance(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = layer_norm_stats(x)
        assert np.var(stats["x_hat"], ddof=0) == pytest.approx(1.0, abs=1e-4)

    def test_x_not_1d(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="x must be 1D"):
            layer_norm_stats(x)
