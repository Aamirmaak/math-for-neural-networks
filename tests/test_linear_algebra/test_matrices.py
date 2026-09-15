"""Tests for matrix operations."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.matrices import (
    _to_matrix,
    identity_matrix,
    matrix_add,
    matrix_subtract,
    scalar_multiply_matrix,
    transpose,
)


class TestMatrixCreation:
    """Test matrix input conversion and validation."""

    def test_list_input(self) -> None:
        m = _to_matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)
        assert m.dtype == np.float64

    def test_numpy_input(self) -> None:
        m = _to_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
        assert m.shape == (2, 3)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _to_matrix(np.empty((0, 3)))

    def test_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="not a scalar"):
            _to_matrix(5.0)

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            _to_matrix([1, 2, 3])


class TestMatrixAddition:
    """Test matrix addition."""

    def test_basic(self) -> None:
        result = matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        np.testing.assert_allclose(result, [[6, 8], [10, 12]])

    def test_commutativity(self) -> None:
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        np.testing.assert_allclose(matrix_add(a, b), matrix_add(b, a))

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            matrix_add([[1, 2]], [[3, 4], [5, 6]])


class TestMatrixSubtraction:
    """Test matrix subtraction."""

    def test_basic(self) -> None:
        result = matrix_subtract([[5, 6], [7, 8]], [[1, 2], [3, 4]])
        np.testing.assert_allclose(result, [[4, 4], [4, 4]])

    def test_self_subtraction(self) -> None:
        a = [[1, 2], [3, 4]]
        np.testing.assert_allclose(matrix_subtract(a, a), [[0, 0], [0, 0]])


class TestScalarMultiplyMatrix:
    """Test scalar multiplication of matrices."""

    def test_basic(self) -> None:
        result = scalar_multiply_matrix(2, [[1, 2], [3, 4]])
        np.testing.assert_allclose(result, [[2, 4], [6, 8]])

    def test_zero_scalar(self) -> None:
        result = scalar_multiply_matrix(0, [[1, 2], [3, 4]])
        np.testing.assert_allclose(result, [[0, 0], [0, 0]])


class TestTranspose:
    """Test matrix transpose."""

    def test_square(self) -> None:
        m = [[1, 2], [3, 4]]
        np.testing.assert_allclose(transpose(m), [[1, 3], [2, 4]])

    def test_rectangular(self) -> None:
        m = [[1, 2, 3], [4, 5, 6]]
        result = transpose(m)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, [[1, 4], [2, 5], [3, 6]])

    def test_double_transpose(self) -> None:
        m = [[1, 2, 3], [4, 5, 6]]
        np.testing.assert_allclose(transpose(transpose(m)), m)

    def test_row_becomes_column(self) -> None:
        m = [[1, 2, 3]]
        np.testing.assert_allclose(transpose(m), [[1], [2], [3]])


class TestIdentityMatrix:
    """Test identity matrix creation."""

    def test_1x1(self) -> None:
        np.testing.assert_allclose(identity_matrix(1), [[1.0]])

    def test_3x3(self) -> None:
        I = identity_matrix(3)
        assert I.shape == (3, 3)
        np.testing.assert_allclose(I, np.eye(3))

    def test_identity_property(self) -> None:
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        I = identity_matrix(2)
        np.testing.assert_allclose(A @ I, A)
        np.testing.assert_allclose(I @ A, A)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            identity_matrix(-1)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            identity_matrix(0)
