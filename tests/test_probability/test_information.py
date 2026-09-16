"""Tests for information theory (entropy, cross-entropy, KL divergence)."""

import numpy as np
import pytest

from math_for_neural_networks.probability.information import (
    cross_entropy,
    entropy,
    kl_divergence,
    log_probability,
)


class TestEntropy:
    def test_uniform(self) -> None:
        # H(uniform) = log(k) where k is number of outcomes
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        assert entropy(probs) == pytest.approx(np.log(4.0))

    def test_deterministic(self) -> None:
        # H(deterministic) = 0
        probs = np.array([1.0, 0.0, 0.0])
        assert entropy(probs) == pytest.approx(0.0)

    def test_binary(self) -> None:
        # H(bernoulli) = -p log(p) - (1-p) log(1-p)
        p = 0.7
        expected = -p * np.log(p) - (1 - p) * np.log(1 - p)
        assert entropy(np.array([p, 1 - p])) == pytest.approx(expected)

    def test_base_2(self) -> None:
        probs = np.array([0.5, 0.5])
        assert entropy(probs, base=2.0) == pytest.approx(1.0)

    def test_non_normalized(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            entropy(np.array([0.3, 0.3]))

    def test_non_negative(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        assert entropy(probs) >= 0


class TestCrossEntropy:
    def test_identical(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert cross_entropy(p, p) == pytest.approx(entropy(p))

    def test_one_hot(self) -> None:
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.7, 0.2, 0.1])
        assert cross_entropy(p, q) == pytest.approx(-np.log(0.7))

    def test_not_symmetric(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        ce_pq = cross_entropy(p, q)
        ce_qp = cross_entropy(q, p)
        assert ce_pq != pytest.approx(ce_qp)

    def test_q_zero_p_nonzero(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert cross_entropy(p, q) == float("inf")

    def test_relationship_to_entropy(self) -> None:
        # H(p, q) = H(p) + D_KL(p || q)
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        ce = cross_entropy(p, q)
        h = entropy(p)
        kl = kl_divergence(p, q)
        assert ce == pytest.approx(h + kl, rel=1e-10)


class TestKLDivergence:
    def test_identical(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self) -> None:
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        assert kl_divergence(p, q) >= 0

    def test_not_symmetric(self) -> None:
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp)

    def test_q_zero_p_nonzero(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert kl_divergence(p, q) == float("inf")

    def test_base_2(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        kl_nat = kl_divergence(p, q, base=np.e)
        kl_bit = kl_divergence(p, q, base=2.0)
        assert kl_bit == pytest.approx(kl_nat / np.log(2.0))


class TestLogProbability:
    def test_one(self) -> None:
        assert log_probability(1.0) == pytest.approx(0.0)

    def test_half(self) -> None:
        assert log_probability(0.5) == pytest.approx(np.log(0.5))

    def test_zero(self) -> None:
        result = log_probability(0.0)
        assert result == -np.inf

    def test_array(self) -> None:
        p = np.array([1.0, 0.5, 0.0])
        result = log_probability(p)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(np.log(0.5))
        assert result[2] == -np.inf

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            log_probability(-0.1)
