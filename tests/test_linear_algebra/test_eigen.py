"""Tests for eigenvalue/eigenvector operations."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.eigen import (
    condition_number,
    eigen_decomposition,
    verify_eigenvector,
)


class TestEigenDecomposition:
    """Test eigenvalue decomposition."""

    def test_diagonal_matrix(self) -> None:
        A = np.diag([2.0, 3.0, 4.0])
        result = eigen_decomposition(A)
        eigenvalues = np.sort(np.real(result["eigenvalues"]))
        np.testing.assert_allclose(eigenvalues, [2.0, 3.0, 4.0])
        assert result["verification_error"] < 1e-12

    def test_identity_matrix(self) -> None:
        A = np.eye(3)
        result = eigen_decomposition(A)
        eigenvalues = np.sort(np.real(result["eigenvalues"]))
        np.testing.assert_allclose(eigenvalues, [1.0, 1.0, 1.0])

    def test_symmetric_matrix(self) -> None:
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        result = eigen_decomposition(A)
        eigenvalues = np.sort(np.real(result["eigenvalues"]))
        np.testing.assert_allclose(eigenvalues, [1.0, 3.0])

    def test_verification_error_small(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        A = A + A.T  # symmetric
        result = eigen_decomposition(A)
        assert result["verification_error"] < 1e-10

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            eigen_decomposition(np.array([[1, 2, 3], [4, 5, 6]]))


class TestVerifyEigenvector:
    """Test eigenvector verification."""

    def test_known_eigenvector(self) -> None:
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        v = np.array([1.0, 1.0])
        lam = 3.0
        result = verify_eigenvector(A, lam, v)
        assert result["error"] < 1e-12

    def test_wrong_eigenvalue(self) -> None:
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        v = np.array([1.0, 1.0])
        result = verify_eigenvector(A, 1.0, v)  # wrong eigenvalue
        assert result["error"] > 0.1


class TestConditionNumber:
    """Test condition number."""

    def test_identity(self) -> None:
        assert condition_number(np.eye(3)) == pytest.approx(1.0)

    def test_diagonal(self) -> None:
        A = np.diag([1.0, 10.0])
        assert condition_number(A) == pytest.approx(10.0)

    def test_singular(self) -> None:
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert condition_number(A) == float("inf")
