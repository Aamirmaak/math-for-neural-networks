"""Tests for chain rule."""

import numpy as np
import pytest

from math_for_neural_networks.calculus.chain_rule import (
    chain_rule_multi,
    chain_rule_scalar,
    demonstrate_chain_rule,
    neural_network_chain_rule_demo,
)


class TestChainRuleScalar:
    """Test scalar chain rule."""

    def test_sin_x_squared(self) -> None:
        """y = sin(x²), dy/dx = cos(x²) * 2x"""
        import math

        def outer(g: float) -> float:
            return math.sin(g)

        def outer_d(g: float) -> float:
            return math.cos(g)

        def inner(x: float) -> float:
            return x**2

        def inner_d(x: float) -> float:
            return 2 * x

        result = chain_rule_scalar(outer, outer_d, inner, inner_d, 2.0)
        # cos(4) * 4
        expected = math.cos(4) * 4
        assert result == pytest.approx(expected)

    def test_exp_linear(self) -> None:
        """y = e^(2x+1), dy/dx = e^(2x+1) * 2"""
        import math

        def outer(g: float) -> float:
            return math.exp(g)

        def outer_d(g: float) -> float:
            return math.exp(g)

        def inner(x: float) -> float:
            return 2 * x + 1

        def inner_d(x: float) -> float:
            return 2.0

        result = chain_rule_scalar(outer, outer_d, inner, inner_d, 1.0)
        expected = math.exp(3) * 2
        assert result == pytest.approx(expected)


class TestChainRuleMulti:
    """Test multivariable chain rule."""

    def test_basic(self) -> None:
        """Two paths contributing to total derivative."""
        # dy/dx = (∂f/∂g₁)(dg₁/dx) + (∂f/∂g₂)(dg₂/dx)
        outer_derivs = np.array([2.0, 3.0])
        inner_derivs = np.array([4.0, 5.0])
        result = chain_rule_multi(outer_derivs, inner_derivs)
        assert result == pytest.approx(2 * 4 + 3 * 5)


class TestDemonstrateChainRule:
    """Test the sin(x²) demonstration."""

    def test_chain_rule_matches_numerical(self) -> None:
        result = demonstrate_chain_rule()
        assert result["error"] < 1e-6


class TestNeuralNetworkChainRule:
    """Test neural network chain rule demo."""

    def test_values(self) -> None:
        result = neural_network_chain_rule_demo()
        # z = 2*3 + 1 = 7
        assert result["z"] == pytest.approx(7.0)
        # loss = (7-10)^2 = 9
        assert result["loss"] == pytest.approx(9.0)
        # dloss_dz = 2*(7-10) = -6
        assert result["dloss_dz"] == pytest.approx(-6.0)
        # dloss_dw = -6 * 3 = -18
        assert result["dloss_dw"] == pytest.approx(-18.0)
        # dloss_db = -6 * 1 = -6
        assert result["dloss_db"] == pytest.approx(-6.0)
