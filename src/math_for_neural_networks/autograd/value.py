"""
Micro Autograd Engine
=====================

A minimal educational automatic differentiation engine using reverse-mode
differentiation (backpropagation) on scalar values.

This module demonstrates:
1. Computational graph construction
2. Forward pass (evaluation)
3. Backward pass (gradient computation)
4. Gradient accumulation across multiple paths
5. Topological ordering for correct reverse traversal

Supported operations:
- Addition, subtraction, multiplication, division
- Power (constant exponent)
- ReLU, sigmoid, tanh activation functions
- Negation

Mathematical foundation:
    For a scalar function y = f(x1, x2, ..., xn), reverse-mode differentiation
    computes all partial derivatives dy/dxi in a single backward pass through
    the computational graph.
"""

from __future__ import annotations

from collections.abc import Callable


class Value:
    """A scalar value node in a computational graph.

    Tracks the value, gradient, and graph connections for automatic
    differentiation via reverse-mode (backpropagation).

    Attributes:
        data: The scalar value.
        grad: The gradient (initialized to 0, accumulated during backward).
        _prev: Set of parent Value nodes.
        _op: String label for the operation that created this node.
        _backward: Function that propagates gradient to parents.
    """

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: set[Value] = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other: Value | float) -> Value:
        """Addition: out = self + other.

        Local derivatives:
            d(out)/d(self) = 1
            d(out)/d(other) = 1
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Value:
        """Supports float + Value."""
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        """Multiplication: out = self * other.

        Local derivatives:
            d(out)/d(self) = other
            d(out)/d(other) = self
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Value:
        """Supports float * Value."""
        return self * other

    def __pow__(self, other: float) -> Value:
        """Power: out = self^other (other must be a constant float).

        Local derivative:
            d(out)/d(self) = other * self^(other - 1)
        """
        assert isinstance(other, (int, float)), "power only supports constant exponents"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        """Negation: out = -self.

        Local derivative:
            d(out)/d(self) = -1
        """
        out = Value(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Value | float) -> Value:
        """Subtraction: out = self - other."""
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        """Supports float - Value."""
        return (-self) + other

    def __truediv__(self, other: Value | float) -> Value:
        """Division: out = self * other^(-1)."""
        return self * other**-1

    def __rtruediv__(self, other: float) -> Value:
        """Supports float / Value."""
        return (self**-1) * other

    def relu(self) -> Value:
        """ReLU: out = max(0, self).

        Local derivative:
            d(out)/d(self) = 1 if self > 0, else 0
        """
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """Sigmoid: out = 1 / (1 + exp(-self)).

        Local derivative:
            d(out)/d(self) = out * (1 - out)
        """
        s = 1.0 / (1.0 + (-self).exp().data)
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Tanh: out = (exp(2x) - 1) / (exp(2x) + 1).

        Local derivative:
            d(out)/d(self) = 1 - out^2
        """
        import math

        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Exponential: out = exp(self).

        Local derivative:
            d(out)/d(self) = exp(self) = out
        """
        import math

        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm: out = log(self).

        Local derivative:
            d(out)/d(self) = 1 / self
        """
        import math

        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute gradients via reverse-mode differentiation.

        Traverses the computational graph in reverse topological order,
        calling each node's _backward function to propagate gradients.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def get_topo_order(root: Value) -> list[Value]:
    """Return topological ordering of the computational graph rooted at root.

    Useful for inspecting the graph structure and verifying traversal order.
    """
    topo: list[Value] = []
    visited: set[Value] = set()

    def build_topo(v: Value) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(root)
    return topo
