"""Tests for softmax and loss functions."""

from __future__ import annotations

import numpy as np
import pytest

from math_for_neural_networks.neural_networks.attention import softmax
from math_for_neural_networks.neural_networks.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    cross_entropy_with_logits,
    mean_squared_error,
)


class TestSoftmax:
    """Tests for softmax function."""

    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert np.sum(s) == pytest.approx(1.0)

    def test_all_positive(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert all(s > 0)

    def test_all_less_than_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert all(s < 1)

    def test_largest_gets_most_prob(self) -> None:
        x = np.array([1.0, 10.0, 1.0])
        s = softmax(x)
        assert np.argmax(s) == 1

    def test_equal_inputs(self) -> None:
        x = np.array([1.0, 1.0, 1.0])
        s = softmax(x)
        np.testing.assert_array_almost_equal(s, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    def test_translation_invariance(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        s1 = softmax(x)
        s2 = softmax(x + 100.0)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_extreme_positive(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        s = softmax(x)
        assert np.sum(s) == pytest.approx(1.0)
        assert all(np.isfinite(s))

    def test_extreme_negative(self) -> None:
        x = np.array([-1000.0, -1001.0, -1002.0])
        s = softmax(x)
        assert np.sum(s) == pytest.approx(1.0)
        assert all(np.isfinite(s))

    def test_single_element(self) -> None:
        x = np.array([5.0])
        s = softmax(x)
        assert s[0] == pytest.approx(1.0)


class TestMSE:
    """Tests for Mean Squared Error."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 5.0])
        expected = np.mean([(0) ** 2, (1) ** 2, (2) ** 2])
        assert mean_squared_error(y_true, y_pred) == pytest.approx(expected)

    def test_non_negative(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([3.0, 4.0])
        assert mean_squared_error(y_true, y_pred) >= 0.0

    def test_symmetric(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([3.0, 4.0])
        assert mean_squared_error(y_true, y_pred) == pytest.approx(
            mean_squared_error(y_pred, y_true)
        )

    def test_shape_mismatch(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_squared_error(y_true, y_pred)


class TestBinaryCrossEntropy:
    """Tests for Binary Cross-Entropy."""

    def test_perfect_prediction(self) -> None:
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.999, 0.001, 0.999])
        loss = binary_cross_entropy(y_true, y_pred)
        assert loss < 0.01

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.9, 0.1])
        expected = -(np.log(0.9) + np.log(0.9)) / 2.0
        assert binary_cross_entropy(y_true, y_pred) == pytest.approx(expected)

    def test_non_negative(self) -> None:
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.5, 0.5, 0.5])
        assert binary_cross_entropy(y_true, y_pred) >= 0.0

    def test_penalizes_confident_wrong(self) -> None:
        y_true = np.array([1.0])
        y_pred_wrong = np.array([0.001])
        y_pred_neutral = np.array([0.5])
        loss_wrong = binary_cross_entropy(y_true, y_pred_wrong)
        loss_neutral = binary_cross_entropy(y_true, y_pred_neutral)
        assert loss_wrong > loss_neutral

    def test_clipping_stability(self) -> None:
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([1.0, 0.0])
        loss = binary_cross_entropy(y_true, y_pred)
        assert np.isfinite(loss)

    def test_shape_mismatch(self) -> None:
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([1.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            binary_cross_entropy(y_true, y_pred)


class TestCategoricalCrossEntropy:
    """Tests for Categorical Cross-Entropy."""

    def test_perfect_one_hot(self) -> None:
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.999, 0.0005, 0.0005])
        loss = categorical_cross_entropy(y_true, y_pred)
        assert loss < 0.01

    def test_known_values(self) -> None:
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.7, 0.2, 0.1])
        expected = -np.log(0.7)
        assert categorical_cross_entropy(y_true, y_pred) == pytest.approx(expected)

    def test_non_negative(self) -> None:
        y_true = np.array([0.0, 1.0, 0.0])
        y_pred = np.array([0.3, 0.4, 0.3])
        assert categorical_cross_entropy(y_true, y_pred) >= 0.0

    def test_batch(self) -> None:
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        loss = categorical_cross_entropy(y_true, y_pred)
        assert loss >= 0.0

    def test_shape_mismatch(self) -> None:
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.7, 0.3])
        with pytest.raises(ValueError, match="Shape mismatch"):
            categorical_cross_entropy(y_true, y_pred)


class TestCrossEntropyWithLogits:
    """Tests for numerically stable logits-based cross-entropy."""

    def test_known_values(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        targets = np.array([0.0, 0.0, 1.0])
        loss = cross_entropy_with_logits(logits, targets)
        assert loss >= 0.0

    def test_matches_manual(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        targets = np.array([0.0, 0.0, 1.0])
        loss_ours = cross_entropy_with_logits(logits, targets)
        from math_for_neural_networks.neural_networks.attention import softmax as sm

        probs = sm(logits)
        loss_manual = -float(np.sum(targets * np.log(probs)))
        assert loss_ours == pytest.approx(loss_manual, abs=1e-10)

    def test_extreme_logits(self) -> None:
        logits = np.array([1000.0, 1001.0, 1002.0])
        targets = np.array([0.0, 0.0, 1.0])
        loss = cross_entropy_with_logits(logits, targets)
        assert np.isfinite(loss)

    def test_extreme_negative_logits(self) -> None:
        logits = np.array([-1000.0, -1001.0, -1002.0])
        targets = np.array([1.0, 0.0, 0.0])
        loss = cross_entropy_with_logits(logits, targets)
        assert np.isfinite(loss)

    def test_batch(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        targets = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        loss = cross_entropy_with_logits(logits, targets)
        assert loss >= 0.0

    def test_shape_mismatch(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            cross_entropy_with_logits(logits, targets)
