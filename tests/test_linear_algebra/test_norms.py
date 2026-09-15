"""Tests for norm and distance operations."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.norms import (
    euclidean_distance,
    l1_norm,
    l2_norm,
)


class TestL1Norm:
    """Test L1 (Manhattan) norm."""

    def test_basic(self) -> None:
        assert l1_norm([3, 4]) == pytest.approx(7.0)

    def test_zero_vector(self) -> None:
        assert l1_norm([0, 0, 0]) == pytest.approx(0.0)

    def test_negative_values(self) -> None:
        assert l1_norm([-3, 4]) == pytest.approx(7.0)

    def test_single_element(self) -> None:
        assert l1_norm([5]) == pytest.approx(5.0)

    def test_matches_numpy(self) -> None:
        v = [1, -2, 3, -4, 5]
        assert l1_norm(v) == pytest.approx(np.linalg.norm(v, ord=1))

    def test_triangle_inequality(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert l1_norm(np.array(a) + np.array(b)) <= l1_norm(a) + l1_norm(b)


class TestL2Norm:
    """Test L2 (Euclidean) norm."""

    def test_basic_3_4(self) -> None:
        assert l2_norm([3, 4]) == pytest.approx(5.0)

    def test_zero_vector(self) -> None:
        assert l2_norm([0, 0, 0]) == pytest.approx(0.0)

    def test_single_element(self) -> None:
        assert l2_norm([5]) == pytest.approx(5.0)

    def test_negative_values(self) -> None:
        assert l2_norm([-3, 4]) == pytest.approx(5.0)

    def test_matches_numpy(self) -> None:
        v = [1, 2, 3, 4, 5]
        assert l2_norm(v) == pytest.approx(np.linalg.norm(v))

    def test_self_dot_equals_norm_squared(self) -> None:
        from math_for_neural_networks.linear_algebra.vectors import dot_product

        v = [3, 4, 5]
        assert l2_norm(v) ** 2 == pytest.approx(dot_product(v, v))

    def test_triangle_inequality(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert l2_norm(np.array(a) + np.array(b)) <= l2_norm(a) + l2_norm(b)


class TestEuclideanDistance:
    """Test Euclidean distance."""

    def test_same_point(self) -> None:
        assert euclidean_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_basic(self) -> None:
        assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_symmetric(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_matches_numpy(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert euclidean_distance(a, b) == pytest.approx(np.linalg.norm(np.array(a) - np.array(b)))

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            euclidean_distance([1, 2], [3, 4, 5])
