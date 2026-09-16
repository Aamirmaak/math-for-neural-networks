"""Tests for probability distributions."""

import math

import numpy as np
import pytest

from math_for_neural_networks.probability.distributions import (
    bernoulli_expectation,
    bernoulli_pmf,
    bernoulli_variance,
    binomial_expectation,
    binomial_pmf,
    binomial_variance,
    categorical_expectation,
    categorical_pmf,
    categorical_variance,
    normal_cdf,
    normal_expectation,
    normal_pdf,
    normal_variance,
    uniform_expectation,
    uniform_pdf,
    uniform_variance,
)


class TestBernoulli:
    def test_pmf_one(self) -> None:
        assert bernoulli_pmf(1, 0.7) == pytest.approx(0.7)

    def test_pmf_zero(self) -> None:
        assert bernoulli_pmf(0, 0.7) == pytest.approx(0.3)

    def test_pmf_one_p_zero(self) -> None:
        assert bernoulli_pmf(1, 0.0) == pytest.approx(0.0)

    def test_pmf_zero_p_one(self) -> None:
        assert bernoulli_pmf(0, 1.0) == pytest.approx(0.0)

    def test_invalid_x(self) -> None:
        with pytest.raises(ValueError):
            bernoulli_pmf(2, 0.5)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError):
            bernoulli_pmf(1, 1.5)

    def test_expectation(self) -> None:
        assert bernoulli_expectation(0.7) == pytest.approx(0.7)

    def test_variance(self) -> None:
        assert bernoulli_variance(0.7) == pytest.approx(0.21)

    def test_variance_extreme(self) -> None:
        assert bernoulli_variance(0.0) == pytest.approx(0.0)
        assert bernoulli_variance(1.0) == pytest.approx(0.0)


class TestBinomial:
    def test_pmf_k_zero(self) -> None:
        assert binomial_pmf(0, 5, 0.5) == pytest.approx(0.5**5)

    def test_pmf_k_n(self) -> None:
        assert binomial_pmf(5, 5, 0.5) == pytest.approx(0.5**5)

    def test_pmf_sum_to_one(self) -> None:
        n, p = 10, 0.3
        total = sum(binomial_pmf(k, n, p) for k in range(n + 1))
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            binomial_pmf(6, 5, 0.5)

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            binomial_pmf(1, -1, 0.5)

    def test_expectation(self) -> None:
        assert binomial_expectation(10, 0.3) == pytest.approx(3.0)

    def test_variance(self) -> None:
        assert binomial_variance(10, 0.3) == pytest.approx(2.1)


class TestCategorical:
    def test_pmf(self) -> None:
        probs = np.array([0.2, 0.5, 0.3])
        assert categorical_pmf(0, probs) == pytest.approx(0.2)
        assert categorical_pmf(1, probs) == pytest.approx(0.5)
        assert categorical_pmf(2, probs) == pytest.approx(0.3)

    def test_pmf_out_of_bounds(self) -> None:
        with pytest.raises(ValueError):
            categorical_pmf(3, np.array([0.5, 0.5]))

    def test_pmf_not_normalized(self) -> None:
        with pytest.raises(ValueError):
            categorical_pmf(0, np.array([0.3, 0.3]))

    def test_expectation(self) -> None:
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert categorical_expectation(probs, values) == pytest.approx(2.5)

    def test_variance(self) -> None:
        probs = np.array([0.5, 0.5])
        values = np.array([0.0, 1.0])
        assert categorical_variance(probs, values) == pytest.approx(0.25)


class TestUniform:
    def test_pdf_inside(self) -> None:
        assert uniform_pdf(0.5, 0.0, 1.0) == pytest.approx(1.0)

    def test_pdf_outside(self) -> None:
        assert uniform_pdf(2.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_pdf_boundary(self) -> None:
        assert uniform_pdf(0.0, 0.0, 1.0) == pytest.approx(1.0)
        assert uniform_pdf(1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            uniform_pdf(0.5, 1.0, 0.0)

    def test_expectation(self) -> None:
        assert uniform_expectation(0.0, 10.0) == pytest.approx(5.0)

    def test_variance(self) -> None:
        assert uniform_variance(0.0, 12.0) == pytest.approx(12.0)


class TestNormal:
    def test_pdf_at_mean(self) -> None:
        assert normal_pdf(0.0, 0.0, 1.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi))

    def test_pdf_symmetry(self) -> None:
        assert normal_pdf(-1.0, 0.0, 1.0) == pytest.approx(normal_pdf(1.0, 0.0, 1.0))

    def test_pdf_negative_sigma(self) -> None:
        with pytest.raises(ValueError):
            normal_pdf(0.0, 0.0, -1.0)

    def test_pdf_array(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        result = normal_pdf(x, 0.0, 1.0)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(result[2])  # symmetric around 0
        assert result[1] > result[0]  # peak at mean

    def test_expectation(self) -> None:
        assert normal_expectation(5.0, 2.0) == pytest.approx(5.0)

    def test_variance(self) -> None:
        assert normal_variance(5.0, 2.0) == pytest.approx(4.0)

    def test_cdf_at_mean(self) -> None:
        assert normal_cdf(0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_cdf_symmetry(self) -> None:
        cdf_neg = normal_cdf(-1.0, 0.0, 1.0)
        cdf_pos = normal_cdf(1.0, 0.0, 1.0)
        assert cdf_neg + cdf_pos == pytest.approx(1.0)
