"""
Optimization module for Math for Neural Networks.

Provides gradient descent, momentum, and Adam optimizers with
convergence diagnostics for understanding neural network training.

Public API:
    # Diagnostics
    OptResult
    check_convergence
    has_finite_values

    # Objective functions
    quadratic, quadratic_gradient
    quartic, quartic_gradient
    sphere, sphere_gradient
    rosenbrock, rosenbrock_gradient
    beale, beale_gradient
    ackley, ackley_gradient
    linear_regression_loss, linear_regression_gradient
    logistic_loss, logistic_gradient

    # Optimizers
    gradient_descent
    momentum
    adam
    adam_step

Neural network connection:
    Gradients -> Optimizer -> Parameter updates -> Loss minimization -> Training
"""

from math_for_neural_networks.optimization.adam import adam, adam_step
from math_for_neural_networks.optimization.diagnostics import (
    OptResult,
    check_convergence,
    has_finite_values,
)
from math_for_neural_networks.optimization.gradient_descent import gradient_descent
from math_for_neural_networks.optimization.momentum import momentum
from math_for_neural_networks.optimization.objectives import (
    ackley,
    ackley_gradient,
    beale,
    beale_gradient,
    linear_regression_gradient,
    linear_regression_loss,
    logistic_gradient,
    logistic_loss,
    quadratic,
    quadratic_gradient,
    quartic,
    quartic_gradient,
    rosenbrock,
    rosenbrock_gradient,
    sphere,
    sphere_gradient,
)

__all__ = [
    # Diagnostics
    "OptResult",
    "check_convergence",
    "has_finite_values",
    # Objectives
    "quadratic",
    "quadratic_gradient",
    "quartic",
    "quartic_gradient",
    "sphere",
    "sphere_gradient",
    "rosenbrock",
    "rosenbrock_gradient",
    "beale",
    "beale_gradient",
    "ackley",
    "ackley_gradient",
    "linear_regression_loss",
    "linear_regression_gradient",
    "logistic_loss",
    "logistic_gradient",
    # Optimizers
    "gradient_descent",
    "momentum",
    "adam",
    "adam_step",
]
