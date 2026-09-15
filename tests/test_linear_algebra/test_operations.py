"""Tests for matrix operations (multiply, mat-vec)."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.operations import (
    matrix_multiply,
    matrix_vector_multiply,
)


class TestMatrixVectorMultiply:
    """Test matrix-vector multiplication."""

    def test_basic(self) -> None:
        A = [[1, 2], [3, 4]]
        x = [5, 6]
        result = matrix_vector_multiply(A, x)
        np.testing.assert_allclose(result, [17, 39])

    def test_identity(self) -> None:
        I = np.eye(3)
        x = [1, 2, 3]
        result = matrix_vector_multiply(I, x)
        np.testing.assert_allclose(result, [1, 2, 3])

    def test_column_vector(self) -> None:
        A = [[1, 0], [0, 1]]
        x = [3, 4]
        result = matrix_vector_multiply(A, x)
        np.testing.assert_allclose(result, [3, 4])

    def test_scaling(self) -> None:
        A = [[2, 0], [0, 3]]
        x = [1, 1]
        result = matrix_vector_multiply(A, x)
        np.testing.assert_allclose(result, [2, 3])

    def test_incompatible_shapes(self) -> None:
        A = [[1, 2, 3], [4, 5, 6]]
        x = [1, 2]
        with pytest.raises(ValueError, match="Incompatible"):
            matrix_vector_multiply(A, x)

    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 3))
        x = rng.standard_normal(3)
        result = matrix_vector_multiply(A, x)
        expected = A @ x
        np.testing.assert_allclose(result, expected)


class TestMatrixMultiply:
    """Test matrix-matrix multiplication."""

    def test_basic_2x2(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_identity(self) -> None:
        A = [[1, 2], [3, 4]]
        I = np.eye(2)
        np.testing.assert_allclose(matrix_multiply(A, I), A)
        np.testing.assert_allclose(matrix_multiply(I, A), A)

    def test_not_commutative(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        AB = matrix_multiply(A, B)
        BA = matrix_multiply(B, A)
        assert not np.allclose(AB, BA)

    def test_associativity(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = [[9, 10], [11, 12]]
        np.testing.assert_allclose(
            matrix_multiply(matrix_multiply(A, B), C),
            matrix_multiply(A, matrix_multiply(B, C)),
        )

    def test_distributivity(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        C = [[9, 10], [11, 12]]
        np.testing.assert_allclose(
            matrix_multiply(
                A, [[b + c for b, c in zip(row_b, row_c)] for row_b, row_c in zip(B, C)]
            ),
            [
                [a + b for a, b in zip(row_ab, row_ac)]
                for row_ab, row_ac in zip(matrix_multiply(A, B), matrix_multiply(A, C))
            ],
        )

    def test_incompatible_shapes(self) -> None:
        A = [[1, 2], [3, 4]]
        B = [[1, 2, 3]]
        with pytest.raises(ValueError, match="Incompatible"):
            matrix_multiply(A, B)

    def test_rectangular(self) -> None:
        A = [[1, 2, 3], [4, 5, 6]]  # 2x3
        B = [[7, 8], [9, 10], [11, 12]]  # 3x2
        result = matrix_multiply(A, B)
        assert result.shape == (2, 2)
        expected = A @ np.array(B)
        np.testing.assert_allclose(result, expected)

    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 3))
        B = rng.standard_normal((3, 5))
        result = matrix_multiply(A, B)
        expected = A @ B
        np.testing.assert_allclose(result, expected)
