"""Tests for cosine similarity."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.similarity import cosine_similarity


class TestCosineSimilarity:
    """Test cosine similarity."""

    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_known_angle(self) -> None:
        # 45 degree angle
        a = [1, 0]
        b = [1, 1]
        assert cosine_similarity(a, b) == pytest.approx(np.cos(np.pi / 4))

    def test_range_bounded(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.standard_normal(10)
            b = rng.standard_normal(10)
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0

    def test_zero_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined for zero vectors"):
            cosine_similarity([0, 0], [1, 2])

    def test_both_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined for zero vectors"):
            cosine_similarity([0, 0], [0, 0])

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            cosine_similarity([1, 2], [3, 4, 5])

    def test_matches_numpy(self) -> None:
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_scaled_vectors_same_direction(self) -> None:
        assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)
