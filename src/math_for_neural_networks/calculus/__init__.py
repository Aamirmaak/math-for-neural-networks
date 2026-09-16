"""
Calculus module for Math for Neural Networks.

Provides numerical and analytical differentiation tools for understanding
neural network training mathematics.

Public API:
    # Analytical derivatives
    quadratic, quadratic_derivative
    cubic, cubic_derivative
    polynomial, polynomial_derivative
    sin_func, sin_derivative
    cos_func, cos_derivative
    exp_func, exp_derivative
    log_func, log_derivative
    sigmoid, sigmoid_derivative
    tanh_func, tanh_derivative
    relu, relu_derivative

    # Numerical differentiation
    forward_difference
    central_difference
    numerical_derivative
    step_size_analysis

    # Partial derivatives
    partial_derivative
    partial_derivative_forward

    # Gradients
    numerical_gradient
    gradient_magnitude
    gradient_direction

    # Chain rule
    chain_rule_scalar
    chain_rule_multi
    demonstrate_chain_rule
    neural_network_chain_rule_demo

Neural network connection:
    Derivatives → gradients → optimization → backpropagation → training
"""

from math_for_neural_networks.calculus.chain_rule import (
    chain_rule_multi,
    chain_rule_scalar,
    demonstrate_chain_rule,
    neural_network_chain_rule_demo,
)
from math_for_neural_networks.calculus.derivatives import (
    cos_derivative,
    cos_func,
    cubic,
    cubic_derivative,
    exp_derivative,
    exp_func,
    log_derivative,
    log_func,
    polynomial,
    polynomial_derivative,
    quadratic,
    quadratic_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    sin_derivative,
    sin_func,
    tanh_derivative,
    tanh_func,
)
from math_for_neural_networks.calculus.finite_differences import (
    central_difference,
    forward_difference,
    numerical_derivative,
    step_size_analysis,
)
from math_for_neural_networks.calculus.gradients import (
    gradient_direction,
    gradient_magnitude,
    numerical_gradient,
)
from math_for_neural_networks.calculus.partials import (
    partial_derivative,
    partial_derivative_forward,
)

__all__ = [
    # Analytical derivatives
    "quadratic",
    "quadratic_derivative",
    "cubic",
    "cubic_derivative",
    "polynomial",
    "polynomial_derivative",
    "sin_func",
    "sin_derivative",
    "cos_func",
    "cos_derivative",
    "exp_func",
    "exp_derivative",
    "log_func",
    "log_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "tanh_func",
    "tanh_derivative",
    "relu",
    "relu_derivative",
    # Numerical differentiation
    "forward_difference",
    "central_difference",
    "numerical_derivative",
    "step_size_analysis",
    # Partial derivatives
    "partial_derivative",
    "partial_derivative_forward",
    # Gradients
    "numerical_gradient",
    "gradient_magnitude",
    "gradient_direction",
    # Chain rule
    "chain_rule_scalar",
    "chain_rule_multi",
    "demonstrate_chain_rule",
    "neural_network_chain_rule_demo",
]
