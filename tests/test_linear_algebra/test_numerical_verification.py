"""Numerical verification tests: compare implementations against NumPy.

These tests verify that our educational implementations produce the same
results as NumPy's optimized routines, confirming mathematical correctness.

Tolerance rationale:
- Our implementations use float64 and direct summation
- NumPy uses optimized BLAS routines
- For direct summation of moderate-sized vectors/matrices, differences
  should be < 1e-12 (machine epsilon accumulation)
- We use 1e-10 as a conservative tolerance for most comparisons
"""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.matrices import (
    matrix_add,
    matrix_subtract,
    transpose,
)
from math_for_neural_networks.linear_algebra.norms import (
    euclidean_distance,
    l1_norm,
    l2_norm,
)
from math_for_neural_networks.linear_algebra.operations import (
    matrix_multiply,
    matrix_vector_multiply,
)
from math_for_neural_networks.linear_algebra.similarity import cosine_similarity
from math_for_neural_networks.linear_algebra.vectors import dot_product, vector_add, vector_subtract


class TestNumericalVerification:
    """Verify custom implementations match NumPy."""

    @pytest.mark.numerical
    def test_dot_product_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            a = rng.standard_normal(size)
            b = rng.standard_normal(size)
            custom = dot_product(a, b)
            expected = np.dot(a, b)
            assert abs(custom - expected) < 1e-10, (
                f"dot_product mismatch: custom={custom}, expected={expected}"
            )

    @pytest.mark.numerical
    def test_vector_add_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            a = rng.standard_normal(size)
            b = rng.standard_normal(size)
            custom = vector_add(a, b)
            expected = a + b
            np.testing.assert_allclose(custom, expected, atol=1e-12)

    @pytest.mark.numerical
    def test_vector_subtract_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            a = rng.standard_normal(size)
            b = rng.standard_normal(size)
            custom = vector_subtract(a, b)
            expected = a - b
            np.testing.assert_allclose(custom, expected, atol=1e-12)

    @pytest.mark.numerical
    def test_l1_norm_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            v = rng.standard_normal(size)
            custom = l1_norm(v)
            expected = np.linalg.norm(v, ord=1)
            assert abs(custom - expected) < 1e-10

    @pytest.mark.numerical
    def test_l2_norm_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            v = rng.standard_normal(size)
            custom = l2_norm(v)
            expected = np.linalg.norm(v)
            assert abs(custom - expected) < 1e-10

    @pytest.mark.numerical
    def test_euclidean_distance_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(1, 100)
            a = rng.standard_normal(size)
            b = rng.standard_normal(size)
            custom = euclidean_distance(a, b)
            expected = np.linalg.norm(a - b)
            assert abs(custom - expected) < 1e-10

    @pytest.mark.numerical
    def test_cosine_similarity_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            size = rng.integers(2, 100)
            a = rng.standard_normal(size)
            b = rng.standard_normal(size)
            custom = cosine_similarity(a, b)
            expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert abs(custom - expected) < 1e-10

    @pytest.mark.numerical
    def test_matrix_add_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(30):
            m, n = rng.integers(1, 20, size=2)
            A = rng.standard_normal((m, n))
            B = rng.standard_normal((m, n))
            custom = matrix_add(A, B)
            np.testing.assert_allclose(custom, A + B, atol=1e-12)

    @pytest.mark.numerical
    def test_matrix_subtract_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(30):
            m, n = rng.integers(1, 20, size=2)
            A = rng.standard_normal((m, n))
            B = rng.standard_normal((m, n))
            custom = matrix_subtract(A, B)
            np.testing.assert_allclose(custom, A - B, atol=1e-12)

    @pytest.mark.numerical
    def test_transpose_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(30):
            m, n = rng.integers(1, 20, size=2)
            A = rng.standard_normal((m, n))
            custom = transpose(A)
            np.testing.assert_allclose(custom, A.T, atol=1e-12)

    @pytest.mark.numerical
    def test_matrix_vector_multiply_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            m = rng.integers(1, 20)
            n = rng.integers(1, 20)
            A = rng.standard_normal((m, n))
            x = rng.standard_normal(n)
            custom = matrix_vector_multiply(A, x)
            expected = A @ x
            np.testing.assert_allclose(custom, expected, atol=1e-10)

    @pytest.mark.numerical
    def test_matrix_multiply_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(30):
            m = rng.integers(1, 15)
            n = rng.integers(1, 15)
            p = rng.integers(1, 15)
            A = rng.standard_normal((m, n))
            B = rng.standard_normal((n, p))
            custom = matrix_multiply(A, B)
            expected = A @ B
            np.testing.assert_allclose(custom, expected, atol=1e-10)
