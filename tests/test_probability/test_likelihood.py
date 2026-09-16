"""Tests for likelihood, log-likelihood, and MLE."""

import math

import numpy as np
import pytest

from math_for_neural_networks.probability.likelihood import (
    bernoulli_log_likelihood,
    bernoulli_likelihood,
    bernoulli_mle,
    categorical_log_likelihood,
    gaussian_log_likelihood,
    gaussian_mle,
)


class TestBernoulliLikelihood:
    def test_all_successes(self) -> None:
        data = np.array([1, 1, 1])
        assert bernoulli_likelihood(data, 0.9) == pytest.approx(0.9**3)

    def test_all_failures(self) -> None:
        data = np.array([0, 0, 0])
        assert bernoulli_likelihood(data, 0.3) == pytest.approx(0.7**3)

    def test_mixed(self) -> None:
        data = np.array([1, 0, 1])
        assert bernoulli_likelihood(data, 0.6) == pytest.approx(0.6 * 0.4 * 0.6)

    def test_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            bernoulli_likelihood(np.array([0, 2, 1]), 0.5)


class TestBernoulliLogLikelihood:
    def test_matches_log(self) -> None:
        data = np.array([1, 0, 1, 0, 1])
        p = 0.7
        ll = bernoulli_log_likelihood(data, p)
        lik = bernoulli_likelihood(data, p)
        assert ll == pytest.approx(math.log(lik), rel=1e-6)

    def test_all_successes(self) -> None:
        data = np.array([1, 1, 1])
        assert bernoulli_log_likelihood(data, 0.9) == pytest.approx(3 * math.log(0.9))


class TestBernoulliMLE:
    def test_all_ones(self) -> None:
        assert bernoulli_mle(np.array([1, 1, 1])) == pytest.approx(1.0)

    def test_all_zeros(self) -> None:
        assert bernoulli_mle(np.array([0, 0, 0])) == pytest.approx(0.0)

    def test_half(self) -> None:
        assert bernoulli_mle(np.array([1, 0, 1, 0])) == pytest.approx(0.5)

    def test_mle_maximizes_likelihood(self) -> None:
        data = np.array([1, 1, 0, 1, 0, 1])
        mle = bernoulli_mle(data)
        # MLE should give higher likelihood than any other p
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert bernoulli_likelihood(data, mle) >= bernoulli_likelihood(data, p) - 1e-10

    def test_empty_data(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            bernoulli_mle(np.array([]))


class TestGaussianLogLikelihood:
    def test_at_true_params(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, sigma = 3.0, 1.0
        ll = gaussian_log_likelihood(data, mu, sigma)
        # Should be finite and reasonable
        assert np.isfinite(ll)

    def test_increases_with_closer_params(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        ll_good = gaussian_log_likelihood(data, 2.0, 1.0)
        ll_bad = gaussian_log_likelihood(data, 10.0, 1.0)
        assert ll_good > ll_bad


class TestGaussianMLE:
    def test_sample_mean(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu_mle, sigma_mle = gaussian_mle(data)
        assert mu_mle == pytest.approx(3.0)
        assert sigma_mle == pytest.approx(np.std(data, ddof=0))

    def test_mle_maximizes_log_likelihood(self) -> None:
        data = np.array([2.0, 3.0, 3.0, 4.0, 5.0])
        mu_mle, sigma_mle = gaussian_mle(data)
        ll_mle = gaussian_log_likelihood(data, mu_mle, sigma_mle)
        ll_bad = gaussian_log_likelihood(data, 10.0, 0.1)
        assert ll_mle > ll_bad


class TestCategoricalLogLikelihood:
    def test_one_hot(self) -> None:
        data = np.array([0, 1, 2])
        log_probs = np.array([0.0, 0.0, 0.0])  # uniform log probs
        ll = categorical_log_likelihood(data, log_probs)
        assert ll == pytest.approx(0.0)

    def test_high_prob_correct(self) -> None:
        data = np.array([0, 1])
        log_probs = np.array([0.0, 0.0])  # uniform
        ll = categorical_log_likelihood(data, log_probs)
        assert ll == pytest.approx(0.0)
