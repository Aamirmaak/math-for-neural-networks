"""Tests for objective functions."""

import numpy as np
import pytest

from math_for_neural_networks.optimization.objectives import (
    ackley,
    ackley_gradient,
    beale,
    beale_gradient,
    linear_regression_gradient,
    linear_regression_loss,
    logistic_gradient,
    logistic_loss,
    quadratic,
    quadratic_gradient,
    quartic,
    quartic_gradient,
    rosenbrock,
    rosenbrock_gradient,
    sphere,
    sphere_gradient,
)


class TestQuadratic:
    def test_at_zero(self):
        assert quadratic(0.0) == 0.0

    def test_at_one(self):
        assert quadratic(1.0) == 1.0

    def test_at_negative(self):
        assert quadratic(-3.0) == 9.0

    def test_gradient_at_zero(self):
        assert quadratic_gradient(0.0) == 0.0

    def test_gradient_at_one(self):
        assert quadratic_gradient(1.0) == 2.0

    def test_gradient_at_negative(self):
        assert quadratic_gradient(-3.0) == -6.0

    def test_gradient_matches_numerical(self):
        x = 2.5
        h = 1e-7
        numerical = (quadratic(x + h) - quadratic(x - h)) / (2 * h)
        assert abs(quadratic_gradient(x) - numerical) < 1e-5


class TestQuartic:
    def test_at_zero(self):
        assert quartic(0.0) == 0.0

    def test_at_one(self):
        assert quartic(1.0) == -1.0

    def test_at_minus_one(self):
        assert quartic(-1.0) == -1.0

    def test_gradient_at_zero(self):
        assert quartic_gradient(0.0) == 0.0

    def test_gradient_at_one(self):
        assert quartic_gradient(1.0) == 0.0

    def test_gradient_at_minus_one(self):
        assert quartic_gradient(-1.0) == 0.0


class TestSphere:
    def test_at_origin(self):
        assert sphere(np.array([0.0, 0.0])) == 0.0

    def test_at_one_one(self):
        assert sphere(np.array([1.0, 1.0])) == 2.0

    def test_gradient_at_origin(self):
        g = sphere_gradient(np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(g, [0.0, 0.0])

    def test_gradient_at_one_one(self):
        g = sphere_gradient(np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(g, [2.0, 2.0])

    def test_gradient_matches_numerical(self):
        x = np.array([1.5, -2.0])
        h = 1e-7
        numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            numerical[i] = (sphere(x_plus) - sphere(x_minus)) / (2 * h)
        np.testing.assert_array_almost_equal(sphere_gradient(x), numerical, decimal=5)


class TestRosenbrock:
    def test_at_minimum(self):
        assert rosenbrock(np.array([1.0, 1.0])) == 0.0

    def test_at_origin(self):
        assert rosenbrock(np.array([0.0, 0.0])) == 1.0 + 0.0

    def test_gradient_at_minimum(self):
        g = rosenbrock_gradient(np.array([1.0, 1.0]))
        np.testing.assert_array_almost_equal(g, [0.0, 0.0], decimal=10)

    def test_gradient_matches_numerical(self):
        x = np.array([0.5, 0.5])
        h = 1e-7
        numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            numerical[i] = (rosenbrock(x_plus) - rosenbrock(x_minus)) / (2 * h)
        np.testing.assert_array_almost_equal(rosenbrock_gradient(x), numerical, decimal=4)


class TestBeale:
    def test_at_minimum(self):
        assert beale(np.array([3.0, 0.5])) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_minimum(self):
        g = beale_gradient(np.array([3.0, 0.5]))
        np.testing.assert_array_almost_equal(g, [0.0, 0.0], decimal=3)


class TestAckley:
    def test_at_origin(self):
        assert ackley(np.array([0.0, 0.0])) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_origin(self):
        g = ackley_gradient(np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(g, [0.0, 0.0], decimal=3)


class TestLinearRegression:
    def test_zero_loss_perfect_fit(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])  # y = 2x
        params = np.array([2.0, 0.0])
        assert linear_regression_loss(params, X, y) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_perfect_fit(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        params = np.array([2.0, 0.0])
        g = linear_regression_gradient(params, X, y)
        np.testing.assert_array_almost_equal(g, [0.0, 0.0], decimal=10)

    def test_gradient_matches_numerical(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        params = np.array([1.0, 0.5])
        h = 1e-7
        numerical = np.zeros(2)
        for i in range(2):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            numerical[i] = (
                linear_regression_loss(p_plus, X, y) - linear_regression_loss(p_minus, X, y)
            ) / (2 * h)
        np.testing.assert_array_almost_equal(
            linear_regression_gradient(params, X, y), numerical, decimal=5
        )


class TestLogistic:
    def test_zero_loss_perfect_prediction(self):
        X = np.array([1.0, -1.0, 2.0])
        y = np.array([1.0, 0.0, 1.0])
        # Large positive w for class 1, large negative w for class 0
        params = np.array([10.0, 5.0])
        loss = logistic_loss(params, X, y)
        assert loss < 0.01  # Should be very small

    def test_gradient_matches_numerical(self):
        X = np.array([1.0, 2.0, -1.0])
        y = np.array([1.0, 1.0, 0.0])
        params = np.array([0.5, 0.1])
        h = 1e-7
        numerical = np.zeros(2)
        for i in range(2):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            numerical[i] = (logistic_loss(p_plus, X, y) - logistic_loss(p_minus, X, y)) / (2 * h)
        np.testing.assert_array_almost_equal(logistic_gradient(params, X, y), numerical, decimal=5)
