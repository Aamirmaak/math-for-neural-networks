"""Tests for partial derivatives."""

import numpy as np
import pytest

from math_for_neural_networks.calculus.partials import (
    partial_derivative,
    partial_derivative_forward,
)


class TestPartialDerivative:
    """Test partial derivative computation."""

    def test_quadratic_xy(self) -> None:
        """f(x,y) = x² + y², ∂f/∂x = 2x"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        # At (3, 4): ∂f/∂x = 2*3 = 6
        assert partial_derivative(f, np.array([3.0, 4.0]), 0) == pytest.approx(6.0, rel=1e-4)

    def test_quadratic_y(self) -> None:
        """f(x,y) = x² + y², ∂f/∂y = 2y"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        # At (3, 4): ∂f/∂y = 2*4 = 8
        assert partial_derivative(f, np.array([3.0, 4.0]), 1) == pytest.approx(8.0, rel=1e-4)

    def test_mixed_terms(self) -> None:
        """f(x,y) = x² + 3xy + y², ∂f/∂x = 2x + 3y"""

        def f(point: np.ndarray) -> float:
            x, y = point[0], point[1]
            return x**2 + 3 * x * y + y**2

        # At (1, 2): ∂f/∂x = 2*1 + 3*2 = 8
        assert partial_derivative(f, np.array([1.0, 2.0]), 0) == pytest.approx(8.0, rel=1e-4)

    def test_three_variables(self) -> None:
        """f(x,y,z) = x*y*z, ∂f/∂x = y*z"""

        def f(point: np.ndarray) -> float:
            return point[0] * point[1] * point[2]

        # At (2, 3, 4): ∂f/∂x = 3*4 = 12
        assert partial_derivative(f, np.array([2.0, 3.0, 4.0]), 0) == pytest.approx(12.0, rel=1e-4)

    def test_negative_h_raises(self) -> None:
        def f(point: np.ndarray) -> float:
            return point[0] ** 2

        with pytest.raises(ValueError, match="positive"):
            partial_derivative(f, np.array([1.0]), 0, h=-0.1)

    def test_out_of_bounds_raises(self) -> None:
        def f(point: np.ndarray) -> float:
            return point[0] ** 2

        with pytest.raises(ValueError, match="out of bounds"):
            partial_derivative(f, np.array([1.0]), 5)


class TestPartialDerivativeForward:
    """Test forward difference partial derivative."""

    def test_matches_central(self) -> None:
        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        point = np.array([3.0, 4.0])
        cd = partial_derivative(f, point, 0)
        fd = partial_derivative_forward(f, point, 0)
        # Both should approximate 6.0
        assert fd == pytest.approx(6.0, rel=1e-4)
        # Forward should be close to central
        assert fd == pytest.approx(cd, rel=1e-3)
