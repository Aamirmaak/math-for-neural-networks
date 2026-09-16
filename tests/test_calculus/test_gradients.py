"""Tests for gradient computation."""

import numpy as np
import pytest

from math_for_neural_networks.calculus.gradients import (
    gradient_direction,
    gradient_magnitude,
    numerical_gradient,
)


class TestNumericalGradient:
    """Test numerical gradient computation."""

    def test_sphere(self) -> None:
        """f(x,y) = x² + y², ∇f = [2x, 2y]"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        grad = numerical_gradient(f, np.array([3.0, 4.0]))
        np.testing.assert_allclose(grad, [6.0, 8.0], rtol=1e-4)

    def test_origin(self) -> None:
        """f(x,y) = x² + y², ∇f(0,0) = [0, 0]"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2

        grad = numerical_gradient(f, np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_three_dim(self) -> None:
        """f(x,y,z) = x² + y² + z², ∇f = [2x, 2y, 2z]"""

        def f(point: np.ndarray) -> float:
            return point[0] ** 2 + point[1] ** 2 + point[2] ** 2

        grad = numerical_gradient(f, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(grad, [2.0, 4.0, 6.0], rtol=1e-4)

    def test_mixed_terms(self) -> None:
        """f(x,y) = x² + 3xy + y², ∇f = [2x+3y, 3x+2y]"""

        def f(point: np.ndarray) -> float:
            x, y = point[0], point[1]
            return x**2 + 3 * x * y + y**2

        # At (1, 2): ∇f = [2+6, 3+4] = [8, 7]
        grad = numerical_gradient(f, np.array([1.0, 2.0]))
        np.testing.assert_allclose(grad, [8.0, 7.0], rtol=1e-4)

    def test_zero_h_raises(self) -> None:
        def f(point: np.ndarray) -> float:
            return point[0] ** 2

        with pytest.raises(ValueError):
            numerical_gradient(f, np.array([1.0]), h=0.0)


class TestGradientMagnitude:
    """Test gradient magnitude computation."""

    def test_basic(self) -> None:
        assert gradient_magnitude(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_zero_gradient(self) -> None:
        assert gradient_magnitude(np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_single_component(self) -> None:
        assert gradient_magnitude(np.array([5.0])) == pytest.approx(5.0)


class TestGradientDirection:
    """Test gradient direction computation."""

    def test_basic(self) -> None:
        direction = gradient_direction(np.array([3.0, 4.0]))
        np.testing.assert_allclose(direction, [0.6, 0.8], atol=1e-10)

    def test_unit_length(self) -> None:
        direction = gradient_direction(np.array([1.0, 2.0, 3.0]))
        assert np.linalg.norm(direction) == pytest.approx(1.0)

    def test_zero_gradient_raises(self) -> None:
        with pytest.raises(ValueError, match="zero gradient"):
            gradient_direction(np.array([0.0, 0.0]))
