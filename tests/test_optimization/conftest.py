"""Test configuration for optimization tests."""

import numpy as np
import pytest


@pytest.fixture
def quadratic_1d():
    """Quadratic f(x) = x^2 with gradient f'(x) = 2x."""
    from math_for_neural_networks.optimization.objectives import quadratic, quadratic_gradient

    return quadratic, quadratic_gradient


@pytest.fixture
def sphere_2d():
    """Sphere f(x,y) = x^2 + y^2 with gradient [2x, 2y]."""
    from math_for_neural_networks.optimization.objectives import sphere, sphere_gradient

    return sphere, sphere_gradient
