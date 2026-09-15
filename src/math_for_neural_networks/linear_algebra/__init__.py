"""
Linear algebra primitives for neural network mathematics.

This submodule provides educational implementations of fundamental
linear algebra operations used throughout neural networks.

Public API:
    # Vectors
    vector_add, vector_subtract, scalar_multiply, dot_product

    # Norms
    l1_norm, l2_norm, euclidean_distance

    # Similarity
    cosine_similarity

    # Matrices
    matrix_add, matrix_subtract, scalar_multiply_matrix, transpose, identity_matrix

    # Operations
    matrix_vector_multiply, matrix_multiply

    # Geometry
    project_vector, orthogonal_component, verify_projection

    # Linear transformations
    scaling_matrix, rotation_matrix, shear_matrix, apply_transformation

    # Eigendecomposition
    eigen_decomposition, verify_eigenvector, condition_number

Neural network connections:
    - Vectors → feature representations, embeddings
    - Dot product → weighted sums, attention scores
    - Matrix multiply → dense/linear layers
    - Norms → regularization, gradient clipping
    - Cosine similarity → embedding similarity
    - Projections → representation geometry
    - Eigenvalues → dimensionality analysis, conditioning
"""

from math_for_neural_networks.linear_algebra.eigen import (
    condition_number,
    eigen_decomposition,
    verify_eigenvector,
)
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
from math_for_neural_networks.linear_algebra.matrices import (
    identity_matrix,
    matrix_add,
    matrix_subtract,
    scalar_multiply_matrix,
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
from math_for_neural_networks.linear_algebra.similarity import (
    cosine_similarity,
)
from math_for_neural_networks.linear_algebra.vectors import (
    dot_product,
    scalar_multiply,
    vector_add,
    vector_subtract,
)

__all__ = [
    # Vectors
    "vector_add",
    "vector_subtract",
    "scalar_multiply",
    "dot_product",
    # Norms
    "l1_norm",
    "l2_norm",
    "euclidean_distance",
    # Matrices
    "matrix_add",
    "matrix_subtract",
    "scalar_multiply_matrix",
    "transpose",
    "identity_matrix",
    # Operations
    "matrix_vector_multiply",
    "matrix_multiply",
    # Similarity
    "cosine_similarity",
    # Geometry
    "project_vector",
    "orthogonal_component",
    "verify_projection",
    "scaling_matrix",
    "rotation_matrix",
    "shear_matrix",
    "reflection_matrix_x",
    "reflection_matrix_y",
    "apply_transformation",
    # Eigen
    "eigen_decomposition",
    "verify_eigenvector",
    "condition_number",
]
