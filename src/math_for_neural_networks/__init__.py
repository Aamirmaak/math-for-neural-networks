"""
math-for-neural-networks

An educational/research-oriented Python toolkit for learning the mathematics
behind neural networks through implementation, visualization, and numerical verification.

Status: Stage 7 — Integrated Experiments & Visualization (Pre-Alpha)
"""

__version__ = "0.8.0"
__author__ = "Math for Neural Networks Contributors"
__license__ = "MIT"

# Package metadata for introspection
__description__ = (
    "Educational toolkit for neural network mathematics: "
    "linear algebra, calculus, probability, optimization, "
    "neural-network mathematics, autograd, and backpropagation"
)
__url__ = "https://github.com/Aamirmaak/math-for-neural-networks"

from math_for_neural_networks import (
    autograd,
    calculus,
    linear_algebra,
    neural_networks,
    optimization,
    probability,
)

__all__ = [
    "linear_algebra",
    "calculus",
    "probability",
    "optimization",
    "neural_networks",
    "autograd",
]
