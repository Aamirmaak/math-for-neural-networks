"""
math-for-neural-networks

An educational/research-oriented Python toolkit for learning the mathematics
behind neural networks through implementation, visualization, and numerical verification.

Status: Stage 5 — Neural Network Mathematics (Pre-Alpha)
"""

__version__ = "0.6.0"
__author__ = "Math for Neural Networks Contributors"
__license__ = "MIT"

# Package metadata for introspection
__description__ = (
    "Educational toolkit for neural network mathematics: "
    "linear algebra, calculus, probability, optimization, and neural-network mathematics"
)
__url__ = "https://github.com/Aamirmaak/math-for-neural-networks"

from math_for_neural_networks import (
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
]
