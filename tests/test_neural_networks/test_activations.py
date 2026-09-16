"""Tests for activation functions and their derivatives."""

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


class TestSigmoid:
    """Tests for sigmoid activation function."""

    def test_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_positive(self) -> None:
        assert sigmoid(2.0) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))

    def test_negative(self) -> None:
        assert sigmoid(-2.0) == pytest.approx(1.0 / (1.0 + np.exp(2.0)))

    def test_range(self) -> None:
        for x in np.linspace(-10, 10, 100):
            s = sigmoid(x)
            assert 0.0 < s < 1.0

    def test_symmetry(self) -> None:
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)

    def test_large_positive(self) -> None:
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self) -> None:
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_array(self) -> None:
        x = np.array([-2.0, 0.0, 2.0])
        s = sigmoid(x)
        assert s.shape == (3,)
        assert s[1] == pytest.approx(0.5)

    def test_derivative_at_zero(self) -> None:
        assert sigmoid_derivative(0.0) == pytest.approx(0.25)

    def test_derivative_known_values(self) -> None:
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            s = sigmoid(x)
            expected = s * (1.0 - s)
            assert sigmoid_derivative(x) == pytest.approx(expected)

    def test_derivative_non_negative(self) -> None:
        for x in np.linspace(-10, 10, 50):
            assert sigmoid_derivative(x) >= 0.0

    def test_derivative_matches_numerical(self) -> None:
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            analytical = sigmoid_derivative(x)
            numerical = central_difference(sigmoid, x, h=1e-5)
            assert analytical == pytest.approx(numerical, abs=1e-5)


class TestTanh:
    """Tests for tanh activation function."""

    def test_zero(self) -> None:
        assert tanh(0.0) == pytest.approx(0.0)

    def test_positive(self) -> None:
        assert tanh(1.0) == pytest.approx(np.tanh(1.0))

    def test_negative(self) -> None:
        assert tanh(-1.0) == pytest.approx(np.tanh(-1.0))

    def test_range(self) -> None:
        for x in np.linspace(-10, 10, 100):
            t = tanh(x)
            assert -1.0 < t < 1.0

    def test_odd_symmetry(self) -> None:
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert tanh(x) == pytest.approx(-tanh(-x))

    def test_large_positive(self) -> None:
        assert tanh(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self) -> None:
        assert tanh(-100.0) == pytest.approx(-1.0, abs=1e-10)

    def test_array(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        t = tanh(x)
        assert t.shape == (3,)
        assert t[1] == pytest.approx(0.0)

    def test_derivative_at_zero(self) -> None:
        assert tanh_derivative(0.0) == pytest.approx(1.0)

    def test_derivative_known_values(self) -> None:
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            t = tanh(x)
            expected = 1.0 - t**2
            assert tanh_derivative(x) == pytest.approx(expected)

    def test_derivative_non_negative(self) -> None:
        for x in np.linspace(-10, 10, 50):
            assert tanh_derivative(x) >= 0.0

    def test_derivative_matches_numerical(self) -> None:
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            analytical = tanh_derivative(x)
            numerical = central_difference(tanh, x, h=1e-5)
            assert analytical == pytest.approx(numerical, abs=1e-5)


class TestReLU:
    """Tests for ReLU activation function."""

    def test_positive(self) -> None:
        assert relu(5.0) == 5.0

    def test_negative(self) -> None:
        assert relu(-5.0) == 0.0

    def test_zero(self) -> None:
        assert relu(0.0) == 0.0

    def test_array(self) -> None:
        x = np.array([-3.0, 0.0, 3.0])
        r = relu(x)
        np.testing.assert_array_equal(r, [0.0, 0.0, 3.0])

    def test_derivative_positive(self) -> None:
        assert relu_derivative(5.0) == 1.0

    def test_derivative_negative(self) -> None:
        assert relu_derivative(-5.0) == 0.0

    def test_derivative_zero_convention(self) -> None:
        assert relu_derivative(0.0) == 0.0

    def test_derivative_array(self) -> None:
        x = np.array([-3.0, 0.0, 3.0])
        d = relu_derivative(x)
        np.testing.assert_array_equal(d, [0.0, 0.0, 1.0])

    def test_derivative_matches_numerical_away_from_zero(self) -> None:
        for x in [-5.0, -1.0, 1.0, 5.0]:
            analytical = relu_derivative(x)
            numerical = central_difference(relu, x, h=1e-5)
            assert analytical == pytest.approx(numerical, abs=1e-5)


class TestGELU:
    """Tests for GELU activation function."""

    def test_zero(self) -> None:
        assert gelu(0.0) == pytest.approx(0.0)

    def test_positive(self) -> None:
        assert gelu(1.0) > 0.0

    def test_negative(self) -> None:
        assert gelu(-1.0) < 0.0

    def test_approaches_relu_for_large_positive(self) -> None:
        assert gelu(100.0) == pytest.approx(100.0, abs=1e-5)

    def test_approaches_zero_for_large_negative(self) -> None:
        assert gelu(-100.0) == pytest.approx(0.0, abs=1e-5)

    def test_array(self) -> None:
        x = np.array([-2.0, 0.0, 2.0])
        g = gelu(x)
        assert g.shape == (3,)
        assert g[1] == pytest.approx(0.0)

    def test_derivative_continuous(self) -> None:
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            d = gelu_derivative(x)
            assert np.isfinite(d)

    def test_derivative_matches_numerical(self) -> None:
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            analytical = gelu_derivative(x)
            numerical = central_difference(gelu, x, h=1e-5)
            assert analytical == pytest.approx(numerical, abs=1e-4)
