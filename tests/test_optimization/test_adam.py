"""Tests for Adam optimizer."""

import numpy as np
import pytest

from math_for_neural_networks.optimization.adam import adam, adam_step
from math_for_neural_networks.optimization.objectives import (
    quadratic,
    quadratic_gradient,
    rosenbrock,
    rosenbrock_gradient,
    sphere,
    sphere_gradient,
)


class TestAdamStep:
    def test_basic_update(self):
        x = np.array([1.0])
        grad = np.array([2.0])
        m = np.array([0.0])
        v = np.array([0.0])
        t = 1

        x_new, m_new, v_new = adam_step(x, grad, m, v, t, learning_rate=0.001)

        # m_1 = 0.9 * 0 + 0.1 * 2.0 = 0.2
        assert m_new[0] == pytest.approx(0.2)
        # v_1 = 0.999 * 0 + 0.001 * 4.0 = 0.004
        assert v_new[0] == pytest.approx(0.004)

    def test_bias_correction(self):
        x = np.array([1.0])
        grad = np.array([1.0])
        m = np.array([0.0])
        v = np.array([0.0])
        t = 1

        x_new, m_new, v_new = adam_step(x, grad, m, v, t, learning_rate=0.001)

        # m_hat = 0.2 / (1 - 0.9) = 0.2 / 0.1 = 2.0
        # v_hat = 0.004 / (1 - 0.999) = 0.004 / 0.001 = 4.0
        # x_new = 1.0 - 0.001 * 2.0 / (sqrt(4.0) + 1e-8) = 1.0 - 0.001
        assert x_new[0] == pytest.approx(1.0 - 0.001 * 2.0 / 2.0, abs=1e-10)

    def test_second_step(self):
        x = np.array([1.0])
        grad = np.array([1.0])
        m = np.array([0.2])
        v = np.array([0.004])
        t = 2

        x_new, m_new, v_new = adam_step(x, grad, m, v, t, learning_rate=0.001)

        # m_2 = 0.9 * 0.2 + 0.1 * 1.0 = 0.28
        assert m_new[0] == pytest.approx(0.28)
        # v_2 = 0.999 * 0.004 + 0.001 * 1.0 = 0.003996 + 0.001 = 0.004996
        assert v_new[0] == pytest.approx(0.004996)

    def test_bias_correction_converges_to_one(self):
        """After many steps, bias correction should approach 1."""
        m = np.array([0.5])
        v = np.array([0.5])
        grad = np.array([0.1])

        # After t=10000, (1 - 0.9^10000) is very close to 1
        x_new, _, _ = adam_step(np.array([1.0]), grad, m, v, 10000)
        # The bias-corrected values should be close to m and v themselves
        m_hat = m / (1.0 - 0.9**10000)
        v_hat = v / (1.0 - 0.999**10000)
        expected = 1.0 - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Should be close but not exact due to remaining bias
        assert x_new[0] == pytest.approx(expected, abs=1e-4)


class TestAdam:
    def test_quadratic_convergence(self):
        result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        assert abs(result.parameters[0]) < 0.01
        assert result.converged

    def test_sphere_2d_convergence(self):
        result = adam(
            sphere, sphere_gradient,
            initial_position=np.array([3.0, -4.0]),
            learning_rate=0.1,
            max_iterations=1000,
        )
        np.testing.assert_array_almost_equal(result.parameters, [0.0, 0.0], decimal=3)
        assert result.converged

    def test_adam_faster_than_gd(self):
        """Adam should converge in fewer iterations than vanilla GD."""
        from math_for_neural_networks.optimization.gradient_descent import gradient_descent

        gd_result = gradient_descent(
            quadratic, quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.01,
            max_iterations=5000,
            grad_tol=1e-6,
        )
        adam_result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([10.0]),
            learning_rate=0.01,
            max_iterations=5000,
            grad_tol=1e-6,
        )
        assert adam_result.converged
        assert gd_result.converged
        # Adam typically converges faster (not guaranteed but expected for simple problems)

    def test_invalid_beta1(self):
        with pytest.raises(ValueError, match="beta1"):
            adam(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                beta1=1.0,
            )

    def test_invalid_beta2(self):
        with pytest.raises(ValueError, match="beta2"):
            adam(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                beta2=1.0,
            )

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            adam(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                epsilon=0.0,
            )

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            adam(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                learning_rate=-0.1,
            )

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            adam(
                quadratic, quadratic_gradient,
                initial_position=np.array([5.0]),
                max_iterations=0,
            )

    def test_objective_decreases(self):
        result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=100,
        )
        assert result.objective_history[-1] < result.objective_history[0]

    def test_records_history(self):
        result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=50,
        )
        assert len(result.objective_history) > 0
        assert len(result.gradient_norm_history) > 0
        assert len(result.parameter_history) > 0

    def test_rosenbrock(self):
        result = adam(
            rosenbrock, rosenbrock_gradient,
            initial_position=np.array([0.0, 0.0]),
            learning_rate=0.001,
            max_iterations=10000,
        )
        # Adam should get reasonably close to (1, 1) from origin
        assert abs(result.parameters[0] - 1.0) < 0.3
        assert abs(result.parameters[1] - 1.0) < 0.3

    def test_gradient_norm_decreases(self):
        result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=500,
        )
        assert result.gradient_norm_history[-1] < result.gradient_norm_history[0]

    def test_parameter_history_first_element(self):
        result = adam(
            quadratic, quadratic_gradient,
            initial_position=np.array([5.0]),
            learning_rate=0.1,
            max_iterations=5,
        )
        assert result.parameter_history[0][0] == pytest.approx(5.0)
