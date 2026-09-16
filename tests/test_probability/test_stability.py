"""Tests for numerical stability utilities."""

import numpy as np
import pytest

from math_for_neural_networks.probability.stability import (
    binary_cross_entropy_with_logits,
    cross_entropy_with_logits,
    log_softmax,
    log_sum_exp,
    sigmoid,
    softmax,
)


class TestLogSumExp:
    def test_basic(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = log_sum_exp(x)
        expected = np.log(np.sum(np.exp(x)))
        assert result == pytest.approx(expected)

    def test_large_values(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        result = log_sum_exp(x)
        assert np.isfinite(result)

    def test_negative_values(self) -> None:
        x = np.array([-100.0, -200.0, -300.0])
        result = log_sum_exp(x)
        assert np.isfinite(result)


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.sum(result) == pytest.approx(1.0)

    def test_large_values(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert np.sum(result) == pytest.approx(1.0)

    def test_negative_values(self) -> None:
        x = np.array([-100.0, -200.0, -300.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert np.sum(result) == pytest.approx(1.0)

    def test_largest_gets_most_prob(self) -> None:
        x = np.array([0.0, 10.0, 0.0])
        result = softmax(x)
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_equals_naive(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        stable = softmax(x)
        naive = np.exp(x) / np.sum(np.exp(x))
        np.testing.assert_allclose(stable, naive, rtol=1e-10)


class TestLogSoftmax:
    def test_sums_to_log_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = log_softmax(x)
        assert np.sum(np.exp(result)) == pytest.approx(1.0)

    def test_equals_log_softmax(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = log_softmax(x)
        expected = np.log(softmax(x))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestSigmoid:
    def test_zero(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self) -> None:
        assert sigmoid(-100.0) == pytest.approx(0.0)

    def test_array(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        result = sigmoid(x)
        assert result.shape == (3,)
        assert result[0] < result[1] < result[2]

    def test_range(self) -> None:
        x = np.linspace(-10, 10, 100)
        result = sigmoid(x)
        assert np.all(result > 0) and np.all(result < 1)


class TestBinaryCrossEntropyWithLogits:
    def test_correct_prediction(self) -> None:
        logits = np.array([10.0, -10.0])
        targets = np.array([1.0, 0.0])
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert loss < 0.1  # Should be very small

    def test_non_negative(self) -> None:
        logits = np.array([0.0, 0.0])
        targets = np.array([1.0, 0.0])
        loss = binary_cross_entropy_with_logits(logits, targets)
        assert loss >= 0


class TestCrossEntropyWithLogits:
    def test_single_sample(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        targets = np.array([0.0, 0.0, 1.0])
        loss = cross_entropy_with_logits(logits, targets)
        # Should equal -log(softmax(logits)[2])
        expected = -np.log(softmax(logits)[2])
        assert loss == pytest.approx(expected)

    def test_batch(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        targets = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        loss = cross_entropy_with_logits(logits, targets)
        assert np.isscalar(loss) or loss.ndim == 0
