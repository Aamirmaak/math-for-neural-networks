"""
Neural Network Mathematics
==========================

Mathematical building blocks for neural networks.

This module connects linear algebra, calculus, probability, and optimization
into the actual components used by neural networks.

Public API:
    # Layers
    affine_transform

    # Activations
    sigmoid, sigmoid_derivative
    tanh, tanh_derivative
    relu, relu_derivative
    gelu, gelu_derivative

    # Losses
    mean_squared_error
    binary_cross_entropy
    categorical_cross_entropy
    cross_entropy_with_logits

    # Attention
    scaled_dot_product_attention
    attention_weights

    # Normalization
    layer_norm
    layer_norm_stats

    # Backpropagation
    gradient_check
    gradient_check_scalar
    affine_backward
    sigmoid_backward
    relu_backward
    tanh_backward
    mse_backward
    softmax_backward
    sigmoid_bce_backward

    # Training
    NetworkParams
    TrainingMetrics
    init_params
    forward
    backward
    sgd_step
    train
    compute_numerical_gradients

Neural network flow:
    Input -> Affine -> Activation -> ... -> Logits -> Softmax -> Prediction -> Loss
    Loss -> Gradients -> Parameter Update -> Lower Loss
"""

from math_for_neural_networks.neural_networks.activations import (
    gelu,
    gelu_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)
from math_for_neural_networks.neural_networks.attention import (
    attention_weights,
    scaled_dot_product_attention,
    softmax,
)
from math_for_neural_networks.neural_networks.backprop import (
    affine_backward,
    gradient_check,
    gradient_check_scalar,
    mse_backward,
    relu_backward,
    sigmoid_backward,
    sigmoid_bce_backward,
    softmax_backward,
    tanh_backward,
)
from math_for_neural_networks.neural_networks.layers import affine_transform
from math_for_neural_networks.neural_networks.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    cross_entropy_with_logits,
    mean_squared_error,
)
from math_for_neural_networks.neural_networks.normalization import (
    layer_norm,
    layer_norm_stats,
)
from math_for_neural_networks.neural_networks.training import (
    NetworkParams,
    TrainingMetrics,
    backward,
    compute_numerical_gradients,
    forward,
    init_params,
    sgd_step,
    train,
)

__all__ = [
    # Layers
    "affine_transform",
    # Activations
    "sigmoid",
    "sigmoid_derivative",
    "tanh",
    "tanh_derivative",
    "relu",
    "relu_derivative",
    "gelu",
    "gelu_derivative",
    # Losses
    "mean_squared_error",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "cross_entropy_with_logits",
    # Attention
    "softmax",
    "scaled_dot_product_attention",
    "attention_weights",
    # Normalization
    "layer_norm",
    "layer_norm_stats",
    # Backpropagation
    "gradient_check",
    "gradient_check_scalar",
    "affine_backward",
    "sigmoid_backward",
    "relu_backward",
    "tanh_backward",
    "mse_backward",
    "softmax_backward",
    "sigmoid_bce_backward",
    # Training
    "NetworkParams",
    "TrainingMetrics",
    "init_params",
    "forward",
    "backward",
    "sgd_step",
    "train",
    "compute_numerical_gradients",
]
