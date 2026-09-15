"""Tests for vector geometry: projections and linear transformations."""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.geometry import (
    apply_transformation,
    orthogonal_component,
    project_vector,
    reflection_matrix_x,
    reflection_matrix_y,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    verify_projection,
)


class TestProjection:
    """Test vector projection."""

    def test_basic(self) -> None:
        # Project [3, 4] onto [1, 0] should give [3, 0]
        result = project_vector([3, 4], [1, 0])
        np.testing.assert_allclose(result, [3, 0])

    def test_parallel_vector(self) -> None:
        # Project [2, 4] onto [1, 2] should give [2, 4] (already parallel)
        result = project_vector([2, 4], [1, 2])
        np.testing.assert_allclose(result, [2, 4])

    def test_orthogonal_vector(self) -> None:
        # Project [0, 1] onto [1, 0] should give [0, 0]
        result = project_vector([0, 1], [1, 0])
        np.testing.assert_allclose(result, [0, 0])

    def test_reconstruction(self) -> None:
        # a = proj_b(a) + orth_b(a)
        a = [3, 4, 5]
        b = [1, 0, 0]
        proj = project_vector(a, b)
        orth = orthogonal_component(a, b)
        np.testing.assert_allclose(proj + orth, a)

    def test_zero_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="zero vector"):
            project_vector([1, 2], [0, 0])

    def test_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            project_vector([1, 2], [3, 4, 5])


class TestVerifyProjection:
    """Test projection verification."""

    def test_reconstruction_error_near_zero(self) -> None:
        a = [3, 4, 5]
        b = [1, 2, 3]
        result = verify_projection(a, b)
        assert result["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)
        assert result["orthogonality_check"] == pytest.approx(0.0, abs=1e-12)


class TestLinearTransformations:
    """Test linear transformation matrices."""

    def test_scaling(self) -> None:
        M = scaling_matrix(2, 3)
        v = np.array([1, 1])
        result = M @ v
        np.testing.assert_allclose(result, [2, 3])

    def test_rotation_90_degrees(self) -> None:
        M = rotation_matrix(np.pi / 2)
        v = np.array([1, 0])
        result = M @ v
        np.testing.assert_allclose(result, [0, 1], atol=1e-12)

    def test_rotation_preserves_norm(self) -> None:
        M = rotation_matrix(np.pi / 3)
        v = np.array([3, 4])
        np.testing.assert_allclose(np.linalg.norm(M @ v), np.linalg.norm(v))

    def test_rotation_orthogonal(self) -> None:
        M = rotation_matrix(np.pi / 4)
        np.testing.assert_allclose(M.T @ M, np.eye(2), atol=1e-12)

    def test_reflection_x(self) -> None:
        M = reflection_matrix_x()
        v = np.array([1, 2])
        result = M @ v
        np.testing.assert_allclose(result, [1, -2])

    def test_reflection_y(self) -> None:
        M = reflection_matrix_y()
        v = np.array([1, 2])
        result = M @ v
        np.testing.assert_allclose(result, [-1, 2])

    def test_shear(self) -> None:
        M = shear_matrix(1, 0)
        v = np.array([1, 1])
        result = M @ v
        np.testing.assert_allclose(result, [1, 2])

    def test_apply_transformation(self) -> None:
        M = scaling_matrix(2, 2)
        v = [3, 4]
        result = apply_transformation(M, v)
        np.testing.assert_allclose(result, [6, 8])
