"""
Experiment 4: Gradient Field
=============================

HYPOTHESIS: The gradient of f(x,y) = x^2 + y^2 points radially outward
(from origin), and negative gradient points toward the minimum.

OBJECTIVE: Visualize contour lines and gradient vectors for a 2D function.

METHOD: Compute numerical gradient on a grid. Plot contour lines with
gradient vectors overlaid. Show that gradients point toward increasing
function value.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.calculus.gradients import (
    numerical_gradient,
    gradient_magnitude,
    gradient_direction,
)
from math_for_neural_networks.optimization.objectives import quadratic, quadratic_gradient
from experiment_utils import ExperimentResult, save_fig, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the gradient field experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Gradient Field",
        hypothesis="Gradient of x^2+y^2 points radially outward from origin",
        method="Compute gradient on grid, overlay on contour plot",
    )

    x_range = np.linspace(-3, 3, 15)
    y_range = np.linspace(-3, 3, 15)
    xx, yy = np.meshgrid(x_range, y_range)

    grad_x = np.zeros_like(xx)
    grad_y = np.zeros_like(yy)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            g = quadratic_gradient(point)
            grad_x[i, j] = g[0]
            grad_y[i, j] = g[1]

    # Contour
    fig, ax = plt.subplots(figsize=(8, 6))
    x_fine = np.linspace(-3, 3, 100)
    y_fine = np.linspace(-3, 3, 100)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
    zz = xx_fine**2 + yy_fine**2

    contour = ax.contourf(xx_fine, yy_fine, zz, levels=20, cmap="viridis", alpha=0.6)
    fig.colorbar(contour, ax=ax, label="f(x,y) = x^2 + y^2")

    skip = 3
    ax.quiver(
        xx[::skip, ::skip],
        yy[::skip, ::skip],
        grad_x[::skip, ::skip],
        grad_y[::skip, ::skip],
        color="red",
        alpha=0.8,
        label="Gradient",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Experiment 4: Gradient Field of f(x,y) = x^2 + y^2")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig(fig, "gradient_field", experiment_name="04_gradient_field")

    # Verify gradient direction at specific points
    test_points = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
    for p in test_points:
        g = quadratic_gradient(p)
        mag = gradient_magnitude(g)
        direction = gradient_direction(g)
        result.metrics[f"gradient_at_{p}"] = list(g)
        result.metrics[f"gradient_mag_at_{p}"] = mag
        result.observations.append(f"gradient at {p}: {g}, magnitude: {mag:.4f}")

    result.conclusion = (
        "The gradient of x^2+y^2 = [2x, 2y], pointing radially outward. "
        "Negative gradient [-2x, -2y] points toward the origin (minimum). "
        "Gradient descent uses -gradient to move toward lower function values."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
