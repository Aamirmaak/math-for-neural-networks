"""
Eigenvalues and eigenvectors for the Math for Neural Networks toolkit.

Mathematics:
    A v = lambda v

An eigenvector v of matrix A is a non-zero vector that, when multiplied
by A, only changes by a scalar factor lambda (the eigenvalue).

Neural network connection:
- Eigenvectors reveal the principal directions of transformation
- PCA uses eigenvectors of the covariance matrix for dimensionality reduction
- Understanding eigenvalues helps with weight initialization and
  understanding training dynamics (conditioning of loss landscape)
- The spectral norm of a weight matrix equals its largest singular value

Design decision:
    We use NumPy's eigenvalue decomposition (np.linalg.eig) as the
    numerical reference. A from-scratch eigenvalue algorithm would add
    excessive complexity without proportional educational value for this
    stage. Instead, we verify the eigenvalue relationship numerically
    and explain the mathematics clearly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.linear_algebra.matrices import _to_matrix
from math_for_neural_networks.linear_algebra.vectors import _to_vector


def eigen_decomposition(
    m: NDArray,
) -> dict[str, Any]:
    """Compute eigenvalues and eigenvectors of a square matrix.

    Uses NumPy's eigendecomposition. Returns eigenvalues and eigenvectors
    along with verification of the eigenvalue relationship A v = lambda v.

    Neural network connection:
    - Principal directions of weight transformations
    - PCA: eigenvectors of covariance = principal components
    - Conditioning: ratio of max/min eigenvalue = condition number

    Args:
        m: Square matrix (2D).

    Returns:
        Dictionary with:
        - eigenvalues: array of eigenvalues
        - eigenvectors: matrix where columns are eigenvectors
        - verification: max error of A @ v - lambda * v

    Raises:
        ValueError: If matrix is not square.
    """
    mat = _to_matrix(m)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Eigen decomposition requires a square matrix, got shape {mat.shape}.")

    # NumPy reference computation
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # Verify: A @ v_i = lambda_i * v_i for each eigenvector
    max_error = 0.0
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        # A @ v
        av = mat @ v
        # lambda * v
        lv = lam * v
        error = float(np.max(np.abs(av - lv)))
        max_error = max(max_error, error)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "verification_error": max_error,
    }


def verify_eigenvector(
    m: NDArray,
    eigenvalue: complex | float,
    eigenvector: NDArray,
) -> dict[str, Any]:
    """Verify that a specific eigenvector satisfies A v = lambda v.

    Args:
        m: Square matrix.
        eigenvalue: Suspected eigenvalue.
        eigenvector: Suspected eigenvector.

    Returns:
        Dictionary with:
        - av: A @ v
        - lam_v: lambda * v
        - error: max absolute difference
        - relative_error: error / max(|lambda| * ||v||, eps)
    """
    mat = _to_matrix(m)
    vec = _to_vector(eigenvector)

    av = mat @ vec
    lam_v = eigenvalue * vec

    error = float(np.max(np.abs(av - lam_v)))
    scale = max(abs(eigenvalue) * float(np.sqrt(vec @ vec)), 1e-15)
    relative_error = error / scale

    return {
        "av": av,
        "lam_v": lam_v,
        "error": error,
        "relative_error": relative_error,
    }


def condition_number(m: NDArray) -> float:
    """Compute the condition number of a matrix using eigenvalues.

    The condition number measures how sensitive the matrix is to input
    perturbations. Large condition numbers indicate ill-conditioning.

    kappa(A) = |lambda_max| / |lambda_min|

    Neural network connection:
    - Ill-conditioned weight matrices cause training instability
    - The condition number of the data covariance affects convergence
    - Batch normalization reduces condition numbers

    Args:
        m: Square matrix.

    Returns:
        Condition number (>= 1.0).
    """
    mat = _to_matrix(m)
    eigenvalues = np.linalg.eigvals(mat)
    abs_eigenvalues = np.abs(eigenvalues)
    min_eig = float(np.min(abs_eigenvalues))
    if min_eig < 1e-15:
        return float("inf")
    return float(np.max(abs_eigenvalues)) / min_eig
