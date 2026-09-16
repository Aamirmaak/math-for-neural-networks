"""Tests for attention mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from math_for_neural_networks.neural_networks.attention import (
    attention_weights,
    scaled_dot_product_attention,
)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_output_shape(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (2, 2)
        assert weights.shape == (2, 3)

    def test_weights_sum_to_one(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        _, weights = scaled_dot_product_attention(Q, K, V)
        for row in weights:
            assert np.sum(row) == pytest.approx(1.0)

    def test_self_attention_identity(self) -> None:
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0]])
        V = np.array([[5.0, 6.0]])
        output, _ = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_array_almost_equal(output, [[5.0, 6.0]])

    def test_equal_keys(self) -> None:
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0], [1.0, 0.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        output, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_array_almost_equal(weights, [[0.5, 0.5]])

    def test_masking(self) -> None:
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        mask = np.array([[True, False]])
        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert weights[0, 0] == pytest.approx(1.0)
        assert weights[0, 1] == pytest.approx(0.0)

    def test_matches_manual_computation(self) -> None:
        Q = np.array([[1.0, 2.0]])
        K = np.array([[3.0, 4.0], [5.0, 6.0]])
        V = np.array([[7.0, 8.0], [9.0, 10.0]])
        d_k = 2
        scores = Q @ K.T / np.sqrt(d_k)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights_manual = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        output_manual = weights_manual @ V
        output, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_array_almost_equal(output, output_manual)
        np.testing.assert_array_almost_equal(weights, weights_manual)

    def test_Q_K_dimension_mismatch(self) -> None:
        Q = np.array([[1.0, 2.0, 3.0]])
        K = np.array([[1.0, 2.0]])
        V = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="same key dimension"):
            scaled_dot_product_attention(Q, K, V)

    def test_K_V_length_mismatch(self) -> None:
        Q = np.array([[1.0, 2.0]])
        K = np.array([[1.0, 2.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="same sequence length"):
            scaled_dot_product_attention(Q, K, V)


class TestAttentionWeights:
    """Tests for attention weights function."""

    def test_shape(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        w = attention_weights(Q, K)
        assert w.shape == (2, 3)

    def test_sums_to_one(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = attention_weights(Q, K)
        for row in w:
            assert np.sum(row) == pytest.approx(1.0)

    def test_masking(self) -> None:
        Q = np.array([[1.0, 0.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        mask = np.array([[True, False]])
        w = attention_weights(Q, K, mask=mask)
        assert w[0, 0] == pytest.approx(1.0)
        assert w[0, 1] == pytest.approx(0.0)
