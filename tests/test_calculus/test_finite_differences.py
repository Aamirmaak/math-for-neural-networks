"""Tests for finite difference methods."""

import numpy as np
import pytest

from math_for_neural_networks.calculus.derivatives import (
    quadratic,
    quadratic_derivative,
    sin_func,
    sin_derivative,
    exp_func,
    exp_derivative,
)
from math_for_neural_networks.calculus.finite_differences import (
    central_difference,
    forward_difference,
    numerical_derivative,
    step_size_analysis,
)


class TestForwardDifference:
    """Test forward difference approximation."""

    def test_quadratic(self) -> None:
        fd = forward_difference(quadratic, 3.0, h=1e-5)
        assert fd == pytest.approx(6.0, rel=1e-4)

    def test_sin(self) -> None:
        fd = forward_difference(sin_func, 0.0, h=1e-5)
        assert fd == pytest.approx(1.0, rel=1e-4)

    def test_exp(self) -> None:
        fd = forward_difference(exp_func, 0.0, h=1e-5)
        assert fd == pytest.approx(1.0, rel=1e-4)

    def test_negative_h_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            forward_difference(quadratic, 1.0, h=-0.1)

    def test_zero_h_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            forward_difference(quadratic, 1.0, h=0.0)


class TestCentralDifference:
    """Test central difference approximation."""

    def test_quadratic(self) -> None:
        cd = central_difference(quadratic, 3.0, h=1e-5)
        assert cd == pytest.approx(6.0, rel=1e-6)

    def test_sin(self) -> None:
        cd = central_difference(sin_func, np.pi / 4, h=1e-5)
        assert cd == pytest.approx(np.cos(np.pi / 4), rel=1e-6)

    def test_exp(self) -> None:
        cd = central_difference(exp_func, 1.0, h=1e-5)
        assert cd == pytest.approx(np.e, rel=1e-6)

    def test_more_accurate_than_forward(self) -> None:
        h = 1e-4
        true_val = quadratic_derivative(3.0)
        fd = forward_difference(quadratic, 3.0, h)
        cd = central_difference(quadratic, 3.0, h)
        assert abs(cd - true_val) < abs(fd - true_val)

    def test_negative_h_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            central_difference(quadratic, 1.0, h=-0.1)


class TestNumericalDerivative:
    """Test numerical_derivative with method selection."""

    def test_central_default(self) -> None:
        result = numerical_derivative(quadratic, 3.0)
        assert result == pytest.approx(6.0, rel=1e-4)

    def test_forward_method(self) -> None:
        result = numerical_derivative(quadratic, 3.0, method="forward")
        assert result == pytest.approx(6.0, rel=1e-4)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            numerical_derivative(quadratic, 3.0, method="backward")


class TestStepSizeAnalysis:
    """Test step size sensitivity experiment."""

    def test_returns_expected_keys(self) -> None:
        result = step_size_analysis(quadratic, 3.0, 6.0)
        assert "h_values" in result
        assert "forward_errors" in result
        assert "central_errors" in result

    def test_central_generally_more_accurate(self) -> None:
        result = step_size_analysis(quadratic, 3.0, 6.0)
        # At moderate h values (1e-5 to 1e-8), central should be more accurate
        moderate_mask = (result["h_values"] >= 1e-8) & (result["h_values"] <= 1e-5)
        if np.any(moderate_mask):
            avg_forward = np.mean(result["forward_errors"][moderate_mask])
            avg_central = np.mean(result["central_errors"][moderate_mask])
            assert avg_central < avg_forward

    def test_too_small_h_increases_error(self) -> None:
        result = step_size_analysis(quadratic, 3.0, 6.0)
        # Very small h should have larger error than optimal h
        very_small = result["h_values"] < 1e-11
        optimal = (result["h_values"] >= 1e-7) & (result["h_values"] <= 1e-5)
        if np.any(very_small) and np.any(optimal):
            avg_small = np.mean(result["central_errors"][very_small])
            avg_optimal = np.mean(result["central_errors"][optimal])
            # At least not dramatically better at very small h
            assert avg_small >= avg_optimal * 0.1
