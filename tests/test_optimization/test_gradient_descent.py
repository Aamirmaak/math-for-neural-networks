"""Tests for gradient descent optimizer."""

import numpy as np
import pytest

from math_for_neural_networks.optimization.diagnostics import (
    OptResult,
    check_convergence,
    has_finite_values,
)
from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.objectives import (
    quadratic,
    quadratic_gradient,
    rosenbrock,
    rosenbrock_gradient,
    sphere,
    sphere_gradient,
)


class TestDiagnostics:
    def test_optresult_properties_empty(self):
        r = OptResult(parameters=np.array([0.0]))
        assert r.final_objective != r.final_objective  # NaN
        assert r.final_gradient_norm != r.final_gradient_norm  # NaN
        assert r.objective_improvement == 0.0

    def test_optresult_properties_with_history(self):
        r = OptResult(
            parameters=np.array([0.0]),
            objective_history=[10.0, 5.0, 1.0],
            gradient_norm_history=[3.0, 1.0, 0.1],
        )
        assert r.final_objective == 1.0
        assert r.final_gradient_norm == 0.1
        assert r.objective_improvement == pytest.approx(9.0)

    def test_check_convergence_gradient(self):
        converged, reason = check_convergence(
            gradient_norm=1e-8, objective_change=1.0,
            parameter_change=1.0, grad_tol=1e-6,
            obj_tol=1e-12, param_tol=1e-8,
        )
        assert converged
        assert "gradient_norm" in reason

    def test_check_convergence_objective(self):
        converged, reason = check_convergence(
            gradient_norm=1.0, objective_change=1e-14,
            parameter_change=1.0, grad_tol=1e-6,
            obj_tol=1e-12, param_tol=1e-8,
        )
        assert converged
        assert "objective_change" in reason

    def test_check_convergence_parameter(self):
        converged, reason = check_convergence(
            gradient_norm=1.0, objective_change=1.0,
            parameter_change=1e-10, grad_tol=1e-6,
            obj_tol=1e-12, param_tol=1e-8,
        )
        assert converged
        assert "parameter_change" in reason

    def test_check_no_convergence(self):
        converged, _ = check_convergence(
            gradient_norm=1.0, objective_change=1.0,
            parameter_change=1.0, grad_tol=1e-6,
            obj_tol=1e-12, param_tol=1e-8,
        )
        assert not converged

    def test_has_finite_values(self):
        assert has_finite_values(np.array([1.0, 2.0, 3.0]))
        assert not has_finite_values(np.array([1.0, np.nan, 3.0]))
        assert not has_finite_values(np.array([1.0, np.inf, 3.0]))


class TestGradientDescent:
    def test_quadratic_convergence(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=500,
        )
        assert abs(result.parameters[0]) < 0.01
        assert result.converged

    def test_quadratic_reaches_minimum(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        assert abs(result.parameters[0]) < 1e-4
        assert result.final_objective < 1e-6

    def test_sphere_2d_convergence(self):
        result = gradient_descent(
            sphere, sphere_gradient,
            initial_position=np.array([3.0, -4.0]),
            learning_rate=0.1,
            max_iterations=500,
        )
        np.testing.assert_array_almost_equal(result.parameters, [0.0, 0.0], decimal=3)
        assert result.converged

    def test_objective_decreases(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=100,
        )
        # Objectives should generally decrease (not strictly monotonic for all methods,
        # but for simple quadratic with reasonable LR, they should)
        assert result.objective_history[-1] < result.objective_history[0]

    def test_gradient_norm_decreases(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=500,
        )
        assert result.gradient_norm_history[-1] < result.gradient_norm_history[0]

    def test_records_history(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=50,
            record_history=True,
        )
        assert len(result.objective_history) > 0
        assert len(result.gradient_norm_history) > 0
        assert len(result.parameter_history) > 0

    def test_no_history(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=50,
            record_history=False,
        )
        assert len(result.objective_history) == 0

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            gradient_descent(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                learning_rate=-0.1,
            )

    def test_zero_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            gradient_descent(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                learning_rate=0.0,
            )

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            gradient_descent(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                max_iterations=0,
            )

    def test_small_lr_slow_convergence(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.001,
            max_iterations=100,
        )
        # With very small LR, should not converge in 100 iterations
        assert result.iterations == 100
        assert abs(result.parameters[0]) > 0.1

    def test_large_lr_oscillation(self):
        # Learning rate too large for x^2 should cause oscillation
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([1.0]),
            learning_rate=1.5,
            max_iterations=100,
        )
        # May or may not converge, but parameters should not be at minimum
        # unless it happens to land there
        assert result.iterations == 100 or abs(result.parameters[0]) > 0.01

    def test_rosenbrock_from_near_minimum(self):
        result = gradient_descent(
            rosenbrock, rosenbrock_gradient,
            initial_position=np.array([0.5, 0.5]),
            learning_rate=0.001,
            max_iterations=5000,
        )
        # Should get reasonably close to (1, 1)
        assert abs(result.parameters[0] - 1.0) < 0.5
        assert abs(result.parameters[1] - 1.0) < 0.5

    def test_parameter_history_correct(self):
        result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=5,
            record_history=True,
        )
        # First parameter in history should be initial position
        assert result.parameter_history[0][0] == pytest.approx(5.0)
        # Second should be 5.0 - 0.1 * 10.0 = 4.0
        assert result.parameter_history[1][0] == pytest.approx(4.0)
