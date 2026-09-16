"""Tests for analytical derivatives."""

import numpy as np
import pytest

from math_for_neural_networks.calculus.derivatives import (
    cos_derivative,
    cos_func,
    cubic,
    cubic_derivative,
    exp_derivative,
    exp_func,
    log_derivative,
    log_func,
    polynomial,
    polynomial_derivative,
    quadratic,
    quadratic_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    sin_derivative,
    sin_func,
    tanh_derivative,
    tanh_func,
)


class TestQuadratic:
    """Test f(x) = x² and f'(x) = 2x."""

    def test_function(self) -> None:
        assert quadratic(3.0) == pytest.approx(9.0)
        assert quadratic(-2.0) == pytest.approx(4.0)
        assert quadratic(0.0) == pytest.approx(0.0)

    def test_derivative(self) -> None:
        assert quadratic_derivative(3.0) == pytest.approx(6.0)
        assert quadratic_derivative(-2.0) == pytest.approx(-4.0)
        assert quadratic_derivative(0.0) == pytest.approx(0.0)

    def test_derivative_array(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(quadratic_derivative(x), [2.0, 4.0, 6.0])


class TestCubic:
    """Test f(x) = x³ and f'(x) = 3x²."""

    def test_function(self) -> None:
        assert cubic(2.0) == pytest.approx(8.0)
        assert cubic(-1.0) == pytest.approx(-1.0)

    def test_derivative(self) -> None:
        assert cubic_derivative(2.0) == pytest.approx(12.0)
        assert cubic_derivative(-1.0) == pytest.approx(3.0)
        assert cubic_derivative(0.0) == pytest.approx(0.0)


class TestPolynomial:
    """Test f(x) = x⁴ - 3x² + 2x - 1."""

    def test_function(self) -> None:
        assert polynomial(1.0) == pytest.approx(1 - 3 + 2 - 1)

    def test_derivative(self) -> None:
        assert polynomial_derivative(1.0) == pytest.approx(4 - 6 + 2)


class TestSin:
    """Test f(x) = sin(x) and f'(x) = cos(x)."""

    def test_function(self) -> None:
        assert sin_func(0.0) == pytest.approx(0.0)
        assert sin_func(np.pi / 2) == pytest.approx(1.0)

    def test_derivative(self) -> None:
        assert sin_derivative(0.0) == pytest.approx(1.0)
        assert sin_derivative(np.pi / 2) == pytest.approx(0.0, abs=1e-10)


class TestCos:
    """Test f(x) = cos(x) and f'(x) = -sin(x)."""

    def test_function(self) -> None:
        assert cos_func(0.0) == pytest.approx(1.0)
        assert cos_func(np.pi) == pytest.approx(-1.0)

    def test_derivative(self) -> None:
        assert cos_derivative(0.0) == pytest.approx(0.0, abs=1e-10)
        assert cos_derivative(np.pi / 2) == pytest.approx(-1.0)


class TestExp:
    """Test f(x) = e^x and f'(x) = e^x."""

    def test_function(self) -> None:
        assert exp_func(0.0) == pytest.approx(1.0)
        assert exp_func(1.0) == pytest.approx(np.e)

    def test_derivative(self) -> None:
        assert exp_derivative(0.0) == pytest.approx(1.0)
        assert exp_derivative(1.0) == pytest.approx(np.e)

    def test_derivative_equals_function(self) -> None:
        x_vals = [0.0, 1.0, -1.0, 2.0]
        for x in x_vals:
            assert exp_derivative(x) == pytest.approx(exp_func(x))


class TestLog:
    """Test f(x) = ln(x) and f'(x) = 1/x."""

    def test_function(self) -> None:
        assert log_func(1.0) == pytest.approx(0.0)
        assert log_func(np.e) == pytest.approx(1.0)

    def test_derivative(self) -> None:
        assert log_derivative(1.0) == pytest.approx(1.0)
        assert log_derivative(2.0) == pytest.approx(0.5)

    def test_domain(self) -> None:
        with pytest.raises(ValueError):
            log_func(0.0)
        with pytest.raises(ValueError):
            log_func(-1.0)


class TestSigmoid:
    """Test sigmoid and its derivative."""

    def test_function(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_derivative_at_zero(self) -> None:
        assert sigmoid_derivative(0.0) == pytest.approx(0.25)

    def test_derivative_formula(self) -> None:
        x = 1.5
        s = sigmoid(x)
        assert sigmoid_derivative(x) == pytest.approx(s * (1 - s))

    def test_range(self) -> None:
        x_vals = np.linspace(-10, 10, 100)
        for x in x_vals:
            assert 0 < sigmoid(x) < 1


class TestTanh:
    """Test tanh and its derivative."""

    def test_function(self) -> None:
        assert tanh_func(0.0) == pytest.approx(0.0)
        assert tanh_func(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_derivative_at_zero(self) -> None:
        assert tanh_derivative(0.0) == pytest.approx(1.0)

    def test_derivative_formula(self) -> None:
        x = 1.5
        t = tanh_func(x)
        assert tanh_derivative(x) == pytest.approx(1 - t**2)


class TestRelu:
    """Test ReLU and its derivative."""

    def test_function(self) -> None:
        assert relu(5.0) == pytest.approx(5.0)
        assert relu(-5.0) == pytest.approx(0.0)
        assert relu(0.0) == pytest.approx(0.0)

    def test_derivative(self) -> None:
        assert relu_derivative(5.0) == pytest.approx(1.0)
        assert relu_derivative(-5.0) == pytest.approx(0.0)
        assert relu_derivative(0.0) == pytest.approx(0.0)
