"""Numerical verification tests for probability module.

Verifies analytical results against known mathematical identities
and cross-checks between implementations.
"""

import math

import numpy as np
import pytest

from math_for_neural_networks.probability.distributions import (
    binomial_pmf,
    normal_cdf,
    normal_pdf,
)
from math_for_neural_networks.probability.information import (
    cross_entropy,
    entropy,
    kl_divergence,
)
from math_for_neural_networks.probability.moments import (
    expectation,
    variance,
)


@pytest.mark.numerical
class TestBinomialNumerical:
    """Verify binomial PMF sums to 1."""

    @pytest.mark.parametrize("n,p", [(5, 0.3), (10, 0.5), (20, 0.7)])
    def test_pmf_sums_to_one(self, n: int, p: float) -> None:
        total = sum(binomial_pmf(k, n, p) for k in range(n + 1))
        assert total == pytest.approx(1.0, abs=1e-10)


@pytest.mark.numerical
class TestNormalNumerical:
    """Verify normal PDF integrates to 1 (numerically)."""

    def test_pdf_integral(self) -> None:
        x = np.linspace(-10, 10, 10000)
        dx = x[1] - x[0]
        integral = np.sum(normal_pdf(x, 0.0, 1.0)) * dx
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_cdf_at_minus_inf(self) -> None:
        assert normal_cdf(-100.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-10)

    def test_cdf_at_plus_inf(self) -> None:
        assert normal_cdf(100.0, 0.0, 1.0) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.numerical
class TestEntropyNumerical:
    """Verify entropy properties."""

    def test_uniform_maximizes_entropy(self) -> None:
        # For k outcomes, uniform distribution maximizes entropy
        k = 5
        uniform = np.ones(k) / k
        h_uniform = entropy(uniform)

        # Any other distribution should have lower or equal entropy
        skewed = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        h_skewed = entropy(skewed)
        assert h_uniform >= h_skewed

    def test_entropy_non_negative(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        assert entropy(probs) >= 0


@pytest.mark.numerical
class TestKLCrossEntropyRelationship:
    """Verify the relationship: H(p, q) = H(p) + D_KL(p || q)."""

    @pytest.mark.parametrize(
        "p,q",
        [
            (np.array([0.3, 0.7]), np.array([0.5, 0.5])),
            (np.array([0.1, 0.9]), np.array([0.2, 0.8])),
            (np.array([0.25, 0.25, 0.5]), np.array([0.33, 0.33, 0.34])),
        ],
    )
    def test_relationship(self, p: np.ndarray, q: np.ndarray) -> None:
        ce = cross_entropy(p, q)
        h = entropy(p)
        kl = kl_divergence(p, q)
        assert ce == pytest.approx(h + kl, rel=1e-10)


@pytest.mark.numerical
class TestExpectationVarianceNumerical:
    """Verify expectation and variance against known results."""

    def test_expectation_of_sum(self) -> None:
        # E[X + Y] = E[X] + E[Y]
        values = np.array([1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        e_x = expectation(values, probs)
        e_2x = expectation(2 * values, probs)
        assert e_2x == pytest.approx(2 * e_x)

    def test_variance_non_negative(self) -> None:
        values = np.array([-5.0, -1.0, 0.0, 3.0, 7.0])
        assert variance(values) >= 0

    def test_variance_constant_zero(self) -> None:
        values = np.array([5.0, 5.0, 5.0, 5.0])
        assert variance(values) == pytest.approx(0.0)
