"""Numerical verification tests for neural network mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from math_for_neural_networks.calculus.finite_differences import central_difference
from math_for_neural_networks.neural_networks.activations import (
    gelu,
    gelu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)
from math_for_neural_networks.neural_networks.attention import (
    scaled_dot_product_attention,
    softmax,
)
from math_for_neural_networks.neural_networks.layers import affine_transform
from math_for_neural_networks.neural_networks.losses import (
    cross_entropy_with_logits,
    mean_squared_error,
)


class TestAffineNumericalVerification:
    """Verify affine transformation against direct NumPy computation."""

    def test_random_single(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        W = rng.standard_normal((3, 5))
        b = rng.standard_normal(3)
        z_ours = affine_transform(x, W, b)
        z_numpy = W @ x + b
        np.testing.assert_array_almost_equal(z_ours, z_numpy)

    def test_random_batch(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 8))
        W = rng.standard_normal((4, 8))
        b = rng.standard_normal(4)
        Z_ours = affine_transform(X, W, b)
        Z_numpy = X @ W.T + b
        np.testing.assert_array_almost_equal(Z_ours, Z_numpy)


class TestActivationNumericalVerification:
    """Verify activation derivatives against numerical differentiation."""

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_sigmoid_derivative(self, x_val: float) -> None:
        analytical = sigmoid_derivative(x_val)
        numerical = central_difference(sigmoid, x_val, h=1e-5)
        assert analytical == pytest.approx(numerical, abs=1e-5)

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_tanh_derivative(self, x_val: float) -> None:
        analytical = tanh_derivative(x_val)
        numerical = central_difference(tanh, x_val, h=1e-5)
        assert analytical == pytest.approx(numerical, abs=1e-5)

    @pytest.mark.parametrize("x_val", [-5.0, -1.0, 1.0, 5.0])
    def test_relu_derivative_away_from_zero(self, x_val: float) -> None:
        analytical = relu_derivative(x_val)
        numerical = central_difference(relu, x_val, h=1e-5)
        assert analytical == pytest.approx(numerical, abs=1e-5)

    @pytest.mark.parametrize("x_val", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_gelu_derivative(self, x_val: float) -> None:
        analytical = gelu_derivative(x_val)
        numerical = central_difference(gelu, x_val, h=1e-5)
        assert analytical == pytest.approx(numerical, abs=1e-4)


class TestSoftmaxNumericalVerification:
    """Verify softmax properties numerically."""

    def test_random_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal(10)
            s = softmax(x)
            assert np.sum(s) == pytest.approx(1.0)

    def test_random_translation_invariant(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal(5)
            c = rng.standard_normal()
            np.testing.assert_array_almost_equal(softmax(x), softmax(x + c))


class TestLossNumericalVerification:
    """Verify loss function properties numerically."""

    def test_mse_gradient_direction(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        loss = mean_squared_error(y_true, y_pred)
        assert loss > 0.0

    def test_cross_entropy_with_logits_matches_softmax_ce(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            logits = rng.standard_normal(5)
            targets = np.zeros(5)
            targets[rng.integers(0, 5)] = 1.0
            loss_logits = cross_entropy_with_logits(logits, targets)
            probs = softmax(logits)
            loss_manual = -float(np.sum(targets * np.log(probs)))
            assert loss_logits == pytest.approx(loss_manual, abs=1e-8)


class TestAttentionNumericalVerification:
    """Verify attention properties numerically."""

    def test_weights_rows_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            Q = rng.standard_normal((3, 4))
            K = rng.standard_normal((5, 4))
            V = rng.standard_normal((5, 2))
            _, weights = scaled_dot_product_attention(Q, K, V)
            for row in weights:
                assert np.sum(row) == pytest.approx(1.0)

    def test_output_is_weighted_average(self) -> None:
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[10.0, 20.0], [30.0, 40.0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        expected = weights[0, 0] * V[0] + weights[0, 1] * V[1]
        np.testing.assert_array_almost_equal(output[0], expected)
