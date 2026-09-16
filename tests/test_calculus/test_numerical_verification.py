"""Numerical verification tests for calculus module.

Verifies analytical derivatives against numerical approximations.
Uses central differences as the reference numerical method.

Tolerance rationale:
- Central difference has O(h²) error, optimal around h ≈ 1e-5
- For smooth functions, error should be < 1e-6 at optimal h
- For sigmoid/tanh, derivatives depend on output, slightly larger tolerance
"""

import numpy as np
import pytest

from math_for_neural_networks.calculus.derivatives import (
    cubic,
    cubic_derivative,
    exp_func,
    exp_derivative,
    log_func,
    log_derivative,
    polynomial,
    polynomial_derivative,
    quadratic,
    quadratic_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    sin_func,
    sin_derivative,
    tanh_func,
    tanh_derivative,
)
from math_for_neural_networks.calculus.finite_differences import central_difference
from math_for_neural_networks.calculus.gradients import numerical_gradient


@pytest.mark.numerical
class TestAnalyticalVsNumerical:
    """Verify analytical derivatives match numerical approximations."""

    @pytest.mark.parametrize(
        "func, deriv, x_val",
        [
            (quadratic, quadratic_derivative, 0.0),
            (quadratic, quadratic_derivative, 1.0),
            (quadratic, quadratic_derivative, -2.0),
            (quadratic, quadratic_derivative, 5.0),
            (cubic, cubic_derivative, 0.0),
            (cubic, cubic_derivative, 1.0),
            (cubic, cubic_derivative, -1.0),
            (cubic, cubic_derivative, 2.0),
            (sin_func, sin_derivative, 0.0),
            (sin_func, sin_derivative, np.pi / 4),
            (sin_func, sin_derivative, np.pi / 2),
            (exp_func, exp_derivative, 0.0),
            (exp_func, exp_derivative, 1.0),
            (exp_func, exp_derivative, -1.0),
            (log_func, log_derivative, 1.0),
            (log_func, log_derivative, 2.0),
            (log_func, log_derivative, 0.5),
            (polynomial, polynomial_derivative, 0.0),
            (polynomial, polynomial_derivative, 1.0),
            (polynomial, polynomial_derivative, -1.0),
        ],
    )
    def test_derivative_matches_numerical(self, func, deriv, x_val) -> None:
        """Analytical derivative should match central difference."""
        numerical = central_difference(func, x_val, h=1e-5)
        analytical = deriv(x_val)
        if abs(analytical) > 1e-10:
            # Relative error for non-zero derivatives
            rel_err = abs(numerical - analytical) / abs(analytical)
            assert rel_err < 1e-4, (
                f"x={x_val}: analytical={analytical}, numerical={numerical}, rel_err={rel_err}"
            )
        else:
            # Absolute error for zero derivatives
            assert abs(numerical - analytical) < 1e-6


@pytest.mark.numerical
class TestActivationDerivatives:
    """Verify activation function derivatives match numerical approximations."""

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_sigmoid_derivative(self, x_val: float) -> None:
        numerical = central_difference(sigmoid, x_val, h=1e-7)
        analytical = sigmoid_derivative(x_val)
        assert abs(numerical - analytical) < 1e-5, (
            f"x={x_val}: analytical={analytical}, numerical={numerical}"
        )

    @pytest.mark.parametrize("x_val", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_tanh_derivative(self, x_val: float) -> None:
        numerical = central_difference(tanh_func, x_val, h=1e-7)
        analytical = tanh_derivative(x_val)
        assert abs(numerical - analytical) < 1e-5, (
            f"x={x_val}: analytical={analytical}, numerical={numerical}"
        )

    @pytest.mark.parametrize("x_val", [-2.0, -0.5, 0.5, 2.0])
    def test_relu_derivative(self, x_val: float) -> None:
        # ReLU derivative is discontinuous at 0, so we skip near 0
        numerical = central_difference(relu, x_val, h=1e-7)
        analytical = relu_derivative(x_val)
        assert abs(numerical - analytical) < 1e-5, (
            f"x={x_val}: analytical={analytical}, numerical={numerical}"
        )


@pytest.mark.numerical
class TestGradientVerification:
    """Verify numerical gradient against analytical gradient."""

    def test_sphere_gradient(self) -> None:
        """f(x,y) = x² + y², ∇f = [2x, 2y]"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        points = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([-1.0, 0.5]),
        ]
        for point in points:
            grad = numerical_gradient(f, point, h=1e-5)
            expected = 2 * point
            np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_mixed_gradient(self) -> None:
        """f(x,y) = x² + 3xy + y², ∇f = [2x+3y, 3x+2y]"""

        def f(point: np.ndarray) -> float:
            x, y = point[0], point[1]
            return x**2 + 3 * x * y + y**2

        point = np.array([1.0, 2.0])
        grad = numerical_gradient(f, point, h=1e-5)
        expected = np.array([8.0, 7.0])
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_three_dim_gradient(self) -> None:
        """f(x,y,z) = x²y + z³, ∇f = [2xy, x², 3z²]"""

        def f(point: np.ndarray) -> float:
            x, y, z = point[0], point[1], point[2]
            return x**2 * y + z**3

        point = np.array([1.0, 2.0, 3.0])
        grad = numerical_gradient(f, point, h=1e-5)
        expected = np.array([4.0, 1.0, 27.0])
        np.testing.assert_allclose(grad, expected, rtol=1e-4)
