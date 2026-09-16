"""
Micro Autograd Engine
=====================

Educational automatic differentiation using reverse-mode differentiation.

Core component:
- Value: scalar node in a computational graph with forward/backward passes
"""

from math_for_neural_networks.autograd.value import Value, get_topo_order

__all__ = ["Value", "get_topo_order"]
