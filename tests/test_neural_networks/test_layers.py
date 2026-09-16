"""Tests for affine transformation and linear layer mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from math_for_neural_networks.neural_networks.layers import affine_transform


class TestAffineTransformSingleSample:
    """Tests for single-sample (1D) affine transformation."""

    def test_basic(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.0, 0.0])
        z = affine_transform(x, W, b)
        np.testing.assert_array_almost_equal(z, [1.0, 2.0])

    def test_with_bias(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([10.0, 20.0])
        z = affine_transform(x, W, b)
        np.testing.assert_array_almost_equal(z, [11.0, 22.0])

    def test_no_bias(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = affine_transform(x, W)
        np.testing.assert_array_almost_equal(z, [1.0, 2.0])

    def test_non_identity(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([0.0, 0.0])
        z = affine_transform(x, W, b)
        np.testing.assert_array_almost_equal(z, [5.0, 11.0])

    def test_matches_numpy(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = np.array([1.0, 2.0, 3.0])
        z_ours = affine_transform(x, W, b)
        z_numpy = W @ x + b
        np.testing.assert_array_almost_equal(z_ours, z_numpy)

    def test_output_shape(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([0.0, 0.0])
        z = affine_transform(x, W, b)
        assert z.shape == (2,)


class TestAffineTransformBatch:
    """Tests for batched (2D) affine transformation."""

    def test_basic_batch(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.0, 0.0])
        Z = affine_transform(X, W, b)
        np.testing.assert_array_almost_equal(Z, [[1.0, 2.0], [3.0, 4.0]])

    def test_batch_with_bias(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([10.0, 20.0])
        Z = affine_transform(X, W, b)
        np.testing.assert_array_almost_equal(Z, [[11.0, 22.0], [13.0, 24.0]])

    def test_batch_shape(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([0.0, 0.0, 0.0])
        Z = affine_transform(X, W, b)
        assert Z.shape == (3, 3)

    def test_batch_matches_numpy(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])
        Z_ours = affine_transform(X, W, b)
        Z_numpy = X @ W.T + b
        np.testing.assert_array_almost_equal(Z_ours, Z_numpy)


class TestAffineTransformValidation:
    """Tests for input validation."""

    def test_W_not_2d(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="W must be 2D"):
            affine_transform(x, W)

    def test_input_features_mismatch(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="Input features mismatch"):
            affine_transform(x, W)

    def test_bias_size_mismatch(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Bias size mismatch"):
            affine_transform(x, W, b)

    def test_x_not_1d_or_2d(self) -> None:
        x = np.array([[[1.0, 2.0]]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="x must be 1D or 2D"):
            affine_transform(x, W)

    def test_bias_not_1d(self) -> None:
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="b must be 1D"):
            affine_transform(x, W, b)
