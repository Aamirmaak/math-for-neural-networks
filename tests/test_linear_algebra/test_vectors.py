"""Tests for vector operations."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.vectors import (
    _to_vector,
    dot_product,
    scalar_multiply,
    vector_add,
    vector_subtract,
)


class TestVectorCreation:
    """Test vector input conversion and validation."""

    def test_list_input(self) -> None:
        v = _to_vector([1.0, 2.0, 3.0])
        assert v.shape == (3,)
        assert v.dtype == np.float64

    def test_numpy_input(self) -> None:
        v = _to_vector(np.array([1, 2, 3]))
        assert v.shape == (3,)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _to_vector([])

    def test_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="not a scalar"):
            _to_vector(5.0)

    def test_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            _to_vector(np.array([[1, 2], [3, 4]]))


class TestVectorAddition:
    """Test vector addition."""

    def test_basic(self) -> None:
        result = vector_add([1, 2, 3], [4, 5, 6])
        np.testing.assert_allclose(result, [5, 7, 9])

    def test_commutativity(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        np.testing.assert_allclose(vector_add(a, b), vector_add(b, a))

    def test_associativity(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        np.testing.assert_allclose(
            vector_add(vector_add(a, b), c),
            vector_add(a, vector_add(b, c)),
        )

    def test_identity(self) -> None:
        a = [1, 2, 3]
        zero = [0, 0, 0]
        np.testing.assert_allclose(vector_add(a, zero), a)

    def test_negative_values(self) -> None:
        result = vector_add([-1, -2], [3, 4])
        np.testing.assert_allclose(result, [2, 2])

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            vector_add([1, 2], [3, 4, 5])


class TestVectorSubtraction:
    """Test vector subtraction."""

    def test_basic(self) -> None:
        result = vector_subtract([4, 5, 6], [1, 2, 3])
        np.testing.assert_allclose(result, [3, 3, 3])

    def test_self_subtraction(self) -> None:
        a = [1, 2, 3]
        np.testing.assert_allclose(vector_subtract(a, a), [0, 0, 0])

    def test_negative_results(self) -> None:
        result = vector_subtract([1, 2], [3, 4])
        np.testing.assert_allclose(result, [-2, -2])

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            vector_subtract([1, 2, 3], [4, 5])


class TestScalarMultiplication:
    """Test scalar multiplication."""

    def test_basic(self) -> None:
        result = scalar_multiply(3, [1, 2, 3])
        np.testing.assert_allclose(result, [3, 6, 9])

    def test_zero_scalar(self) -> None:
        result = scalar_multiply(0, [1, 2, 3])
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_negative_scalar(self) -> None:
        result = scalar_multiply(-1, [1, 2, 3])
        np.testing.assert_allclose(result, [-1, -2, -3])

    def test_fractional_scalar(self) -> None:
        result = scalar_multiply(0.5, [2, 4, 6])
        np.testing.assert_allclose(result, [1, 2, 3])

    def test_distributivity(self) -> None:
        c = 3
        a = [1, 2, 3]
        b = [4, 5, 6]
        np.testing.assert_allclose(
            scalar_multiply(c, vector_add(a, b)),
            vector_add(scalar_multiply(c, a), scalar_multiply(c, b)),
        )


class TestDotProduct:
    """Test dot product."""

    def test_basic(self) -> None:
        assert dot_product([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)

    def test_commutativity(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert dot_product(a, b) == pytest.approx(dot_product(b, a))

    def test_orthogonal_vectors(self) -> None:
        assert dot_product([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_self_dot_product(self) -> None:
        v = [3, 4]
        assert dot_product(v, v) == pytest.approx(25.0)

    def test_negative_values(self) -> None:
        assert dot_product([-1, 2], [3, -4]) == pytest.approx(-11.0)

    def test_distributivity(self) -> None:
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        assert dot_product(a, vector_add(b, c)) == pytest.approx(
            dot_product(a, b) + dot_product(a, c)
        )

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            dot_product([1, 2], [3, 4, 5])
