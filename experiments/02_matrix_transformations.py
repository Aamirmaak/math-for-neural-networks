"""
Experiment 2: Matrix Transformations
=====================================

HYPOTHESIS: Linear transformations (scaling, rotation, reflection, shear)
can be composed via matrix multiplication, and correspond to neural network
affine transformations.

OBJECTIVE: Demonstrate 2D linear transformations visually and mathematically.

METHOD: Apply scaling, rotation, reflection, shear to a unit square and grid.
Visualize original vs transformed geometry.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.linear_algebra import (
    scaling_matrix,
    rotation_matrix,
    reflection_matrix_x,
    shear_matrix,
    apply_transformation,
    identity_matrix,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed


def transform_grid(M: np.ndarray, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Transform a grid of points by matrix M."""
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    transformed = points @ M.T
    return points, transformed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the matrix transformation experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Matrix Transformations",
        hypothesis="Matrix multiplication implements linear transformations",
        method="Apply scaling, rotation, reflection, shear to 2D point sets",
    )

    transforms = {
        "Identity": identity_matrix(2),
        "Scaling (2x, 0.5y)": scaling_matrix(2.0, 0.5),
        "Rotation 45 deg": rotation_matrix(np.pi / 4),
        "Reflection (x-axis)": reflection_matrix_x(),
        "Shear (0.5, 0)": shear_matrix(0.5, 0.0),
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (name, M) in enumerate(transforms.items()):
        points, transformed = transform_grid(M)
        ax = axes[idx]
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5, label="Original")
        ax.scatter(transformed[:, 0], transformed[:, 1], s=5, alpha=0.5, label="Transformed")
        ax.set_title(name)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Experiment 2: 2D Linear Transformations", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "matrix_transformations", experiment_name="02_matrix_transformations")

    # Verify composition
    S = scaling_matrix(2.0, 0.5)
    R = rotation_matrix(np.pi / 4)
    SR = S @ R
    v = np.array([1.0, 0.0])
    composed = apply_transformation(SR, v)
    sequential = apply_transformation(R, v)
    sequential = apply_transformation(S, sequential)
    result.metrics["composition_error"] = float(np.max(np.abs(composed - sequential)))
    result.observations.append(
        f"Composition via matrix multiply matches sequential application: "
        f"error = {result.metrics['composition_error']:.2e}"
    )
    result.observations.append(
        "Neural network affine transformation z = Wx + b is a linear "
        "transformation (W) plus a translation (b)"
    )
    result.conclusion = (
        "Matrices represent linear transformations. Composition = matrix multiplication. "
        "Neural network layers apply affine transformations to their inputs."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
