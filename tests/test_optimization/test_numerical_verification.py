"""Numerical verification tests for optimization algorithms."""

import numpy as np
import pytest

from math_for_neural_networks.optimization.adam import adam
from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.objectives import (
    linear_regression_gradient,
    linear_regression_loss,
    quadratic,
    quadratic_gradient,
    sphere,
    sphere_gradient,
)


class TestGradientVerification:
    """Verify gradients against numerical approximation."""

    def test_quadratic_gradient_numerical(self):
        x = 3.0
        h = 1e-7
        numerical = (quadratic(x + h) - quadratic(x - h)) / (2 * h)
        assert abs(quadratic_gradient(x) - numerical) < 1e-5

    def test_sphere_gradient_numerical(self):
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

    def test_linear_regression_gradient_numerical(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
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


class TestConvergenceVerification:
    """Verify that optimizers converge to known solutions."""

    def test_gd_quadratic_minimum_at_zero(self):
        result = gradient_descent(
            quadratic,
            quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        assert abs(result.parameters[0]) < 1e-4

    def test_momentum_quadratic_minimum_at_zero(self):
        result = momentum(
            quadratic,
            quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.1,
            beta=0.9,
            max_iterations=1000,
        )
        assert abs(result.parameters[0]) < 1e-4

    def test_adam_quadratic_minimum_at_zero(self):
        result = adam(
            quadratic,
            quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        assert abs(result.parameters[0]) < 1e-4

    def test_all_optimizers_converge_on_sphere(self):
        init = np.array([5.0, -3.0])
        for opt_fn in [gradient_descent, momentum, adam]:
            kwargs = dict(
                objective=sphere,
                gradient=sphere_gradient,
                initial_position=init.copy(),
                learning_rate=0.1,
                max_iterations=1000,
            )
            if opt_fn is momentum:
                kwargs["beta"] = 0.9
            result = opt_fn(**kwargs)
            np.testing.assert_array_almost_equal(
                result.parameters,
                [0.0, 0.0],
                decimal=2,
                err_msg=f"{opt_fn.__name__} failed to converge on sphere",
            )


class TestObjectiveDecrease:
    """Verify that objective value decreases over iterations."""

    @pytest.mark.parametrize(
        "opt_fn,kwargs",
        [
            (gradient_descent, {"learning_rate": 0.1}),
            (momentum, {"learning_rate": 0.1, "beta": 0.9}),
            (adam, {"learning_rate": 0.1}),
        ],
    )
    def test_objective_decreases(self, opt_fn, kwargs):
        result = opt_fn(
            quadratic,
            quadratic_gradient,
            initial_position=np.array([5.0]),
            max_iterations=100,
            **kwargs,
        )
        # First objective should be greater than last
        assert result.objective_history[0] > result.objective_history[-1]


class TestLinearRegressionFitting:
    """Verify optimizers can fit a simple linear model."""

    def test_fit_linear_relationship(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * X + 1.0  # y = 2x + 1

        result = gradient_descent(
            objective=lambda p: linear_regression_loss(p, X, y),
            gradient=lambda p: linear_regression_gradient(p, X, y),
            initial_position=np.array([0.0, 0.0]),
            learning_rate=0.01,
            max_iterations=5000,
        )
        # Should find w ~= 2.0, b ~= 1.0
        assert abs(result.parameters[0] - 2.0) < 0.1
        assert abs(result.parameters[1] - 1.0) < 0.1

    def test_adam_fit_linear(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * X + 1.0

        result = adam(
            objective=lambda p: linear_regression_loss(p, X, y),
            gradient=lambda p: linear_regression_gradient(p, X, y),
            initial_position=np.array([0.0, 0.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        assert abs(result.parameters[0] - 2.0) < 0.1
        assert abs(result.parameters[1] - 1.0) < 0.1
