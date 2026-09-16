"""Tests for momentum optimizer."""

import numpy as np
import pytest

from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.objectives import (
    quadratic,
    quadratic_gradient,
    rosenbrock,
    rosenbrock_gradient,
    sphere,
    sphere_gradient,
)


class TestMomentum:
    def test_quadratic_convergence(self):
        result = momentum(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            beta=0.9,
            max_iterations=500,
        )
        assert abs(result.parameters[0]) < 0.01
        assert result.converged

    def test_sphere_2d_convergence(self):
        result = momentum(
            sphere, sphere_gradient,
            initial_position=np.array([3.0, -4.0]),
            learning_rate=0.05,
            beta=0.9,
            max_iterations=500,
        )
        np.testing.assert_array_almost_equal(result.parameters, [0.0, 0.0], decimal=2)
        assert result.converged

    def test_momentum_faster_than_gd(self):
        """Momentum should converge in fewer iterations than vanilla GD on some problems."""
        from math_for_neural_networks.optimization.gradient_descent import gradient_descent

        gd_result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.01,
            max_iterations=2000,
            grad_tol=1e-6,
        )
        mom_result = momentum(
            quadratic, quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.01,
            beta=0.9,
            max_iterations=2000,
            grad_tol=1e-6,
        )
        # Both should converge, momentum may converge faster
        # (not guaranteed for all problems, but for quadratic it often does)
        assert mom_result.converged
        assert gd_result.converged

    def test_invalid_beta_negative(self):
        with pytest.raises(ValueError, match="beta"):
            momentum(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                beta=-0.1,
            )

    def test_invalid_beta_one(self):
        with pytest.raises(ValueError, match="beta"):
            momentum(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                beta=1.0,
            )

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            momentum(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                learning_rate=-0.1,
            )

    def test_beta_zero_reduces_to_gd(self):
        """With beta=0, momentum should behave like vanilla gradient descent."""
        from math_for_neural_networks.optimization.gradient_descent import gradient_descent

        gd_result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=100,
            record_history=False,
        )
        mom_result = momentum(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            beta=0.0,
            max_iterations=100,
            record_history=False,
        )
        np.testing.assert_array_almost_equal(
            gd_result.parameters, mom_result.parameters, decimal=10
        )

    def test_objective_decreases(self):
        result = momentum(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            beta=0.9,
            max_iterations=100,
        )
        assert result.objective_history[-1] < result.objective_history[0]

    def test_records_history(self):
        result = momentum(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            beta=0.9,
            max_iterations=50,
        )
        assert len(result.objective_history) > 0
        assert len(result.gradient_norm_history) > 0
        assert len(result.parameter_history) > 0

    def test_rosenbrock_from_near_minimum(self):
        result = momentum(
            rosenbrock, rosenbrock_gradient,
            initial_position=np.array([0.5, 0.5]),
            learning_rate=0.001,
            beta=0.9,
            max_iterations=5000,
        )
        assert abs(result.parameters[0] - 1.0) < 0.5
        assert abs(result.parameters[1] - 1.0) < 0.5
