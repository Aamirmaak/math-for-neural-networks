"""Tests for moments (expectation, variance, standard deviation)."""

import numpy as np
import pytest

from math_for_neural_networks.probability.moments import (
    expectation,
    standard_deviation,
    variance,
)


class TestExpectation:
    def test_sample_mean(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert expectation(values) == pytest.approx(2.5)

    def test_weighted(self) -> None:
        values = np.array([0.0, 1.0, 2.0])
        probs = np.array([0.25, 0.5, 0.25])
        assert expectation(values, probs) == pytest.approx(1.0)

    def test_constant(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        probs = np.array([0.3, 0.3, 0.4])
        assert expectation(values, probs) == pytest.approx(5.0)

    def test_probability_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            expectation(np.array([1.0, 2.0]), np.array([0.3, 0.3]))

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            expectation(np.array([1.0, 2.0]), np.array([0.5, 0.3, 0.2]))


class TestVariance:
    def test_sample_variance(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert variance(values) == pytest.approx(2.0)

    def test_weighted_variance(self) -> None:
        values = np.array([0.0, 1.0])
        probs = np.array([0.5, 0.5])
        assert variance(values, probs) == pytest.approx(0.25)

    def test_constant_variance(self) -> None:
        values = np.array([5.0, 5.0, 5.0])
        probs = np.array([0.3, 0.3, 0.4])
        assert variance(values, probs) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        values = np.array([-3.0, -1.0, 2.0, 5.0])
        assert variance(values) >= 0


class TestStandardDeviation:
    def test_sample_std(self) -> None:
        values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert standard_deviation(values) == pytest.approx(2.0)

    def test_constant_std(self) -> None:
        values = np.array([3.0, 3.0, 3.0])
        assert standard_deviation(values) == pytest.approx(0.0)

    def test_weighted_std(self) -> None:
        values = np.array([0.0, 1.0])
        probs = np.array([0.5, 0.5])
        assert standard_deviation(values, probs) == pytest.approx(0.5)
