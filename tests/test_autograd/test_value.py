"""Tests for micro autograd engine (scalar Value class)."""

import math

import pytest

from math_for_neural_networks.autograd.value import Value, get_topo_order


class TestValueBasicOperations:
    """Test basic arithmetic operations and their values."""

    def test_create_value(self) -> None:
        v = Value(3.0)
        assert v.data == 3.0
        assert v.grad == 0.0

    def test_add(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0
        assert c._op == "+"

    def test_mul(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0
        assert c._op == "*"

    def test_pow(self) -> None:
        a = Value(3.0)
        b = a**2
        assert b.data == 9.0

    def test_neg(self) -> None:
        a = Value(3.0)
        b = -a
        assert b.data == -3.0

    def test_sub(self) -> None:
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_div(self) -> None:
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        assert c.data == 3.0

    def test_radd(self) -> None:
        a = Value(3.0)
        b = 2.0 + a
        assert b.data == 5.0

    def test_rmul(self) -> None:
        a = Value(3.0)
        b = 2.0 * a
        assert b.data == 6.0

    def test_repr(self) -> None:
        v = Value(3.14159)
        r = repr(v)
        assert "data=" in r
        assert "grad=" in r


class TestValueGradients:
    """Test gradient computation via backward()."""

    def test_simple_add(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_simple_mul(self) -> None:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0  # dc/da = b
        assert b.grad == 2.0  # dc/db = a

    def test_chain_mul(self) -> None:
        # c = a * b * d
        a = Value(2.0)
        b = Value(3.0)
        d = Value(4.0)
        c = a * b * d
        c.backward()
        assert a.grad == 12.0  # b * d
        assert b.grad == 8.0  # a * d
        assert d.grad == 6.0  # a * b

    def test_power(self) -> None:
        a = Value(3.0)
        b = a**2
        b.backward()
        assert a.grad == 6.0  # 2 * a

    def test_power_cube(self) -> None:
        a = Value(2.0)
        b = a**3
        b.backward()
        assert a.grad == 12.0  # 3 * a^2

    def test_negation(self) -> None:
        a = Value(3.0)
        b = -a
        b.backward()
        assert a.grad == -1.0

    def test_subtraction(self) -> None:
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_division(self) -> None:
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()
        # c = a * b^(-1)
        # dc/da = b^(-1) = 0.5
        # dc/db = -a * b^(-2) = -1.5
        assert abs(a.grad - 0.5) < 1e-10
        assert abs(b.grad - (-1.5)) < 1e-10

    def test_complex_expression(self) -> None:
        # f(a,b) = (a + b) * (a - b) = a^2 - b^2
        a = Value(3.0)
        b = Value(2.0)
        f = (a + b) * (a - b)
        f.backward()
        # df/da = 2a = 6
        # df/db = -2b = -4
        assert abs(a.grad - 6.0) < 1e-10
        assert abs(b.grad - (-4.0)) < 1e-10


class TestActivationGradients:
    """Test activation function gradients."""

    def test_sigmoid(self) -> None:
        x = Value(0.0)
        y = x.sigmoid()
        y.backward()
        # sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        assert abs(x.grad - 0.25) < 1e-10

    def test_sigmoid_positive(self) -> None:
        x = Value(2.0)
        y = x.sigmoid()
        y.backward()
        s = 1.0 / (1.0 + math.exp(-2.0))
        expected = s * (1.0 - s)
        assert abs(x.grad - expected) < 1e-10

    def test_tanh(self) -> None:
        x = Value(0.0)
        y = x.tanh()
        y.backward()
        # tanh(0) = 0, derivative = 1 - 0 = 1
        assert abs(x.grad - 1.0) < 1e-10

    def test_tanh_nonzero(self) -> None:
        x = Value(1.0)
        y = x.tanh()
        y.backward()
        t = math.tanh(1.0)
        expected = 1.0 - t**2
        assert abs(x.grad - expected) < 1e-10

    def test_relu_positive(self) -> None:
        x = Value(2.0)
        y = x.relu()
        y.backward()
        assert x.grad == 1.0

    def test_relu_negative(self) -> None:
        x = Value(-2.0)
        y = x.relu()
        y.backward()
        assert x.grad == 0.0

    def test_relu_zero(self) -> None:
        x = Value(0.0)
        y = x.relu()
        y.backward()
        assert x.grad == 0.0

    def test_exp(self) -> None:
        x = Value(1.0)
        y = x.exp()
        y.backward()
        # exp(1) = e, derivative = e
        assert abs(x.grad - math.e) < 1e-10


class TestGradientAccumulation:
    """Test that gradients accumulate across multiple paths."""

    def test_two_paths(self) -> None:
        # y = x * x + x
        # dy/dx = 2x + 1
        x = Value(3.0)
        y = x * x + x
        y.backward()
        assert abs(x.grad - 7.0) < 1e-10  # 2*3 + 1

    def test_three_paths(self) -> None:
        # y = x * x * x + x * x + x
        # dy/dx = 3x^2 + 2x + 1
        x = Value(2.0)
        y = x * x * x + x * x + x
        y.backward()
        expected = 3 * 4 + 2 * 2 + 1  # 12 + 4 + 1 = 17
        assert abs(x.grad - expected) < 1e-10

    def test_shared_node(self) -> None:
        # f = a * b + a * c
        # df/da = b + c
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        f = a * b + a * c
        f.backward()
        assert abs(a.grad - 7.0) < 1e-10  # b + c = 3 + 4
        assert abs(b.grad - 2.0) < 1e-10  # a = 2
        assert abs(c.grad - 2.0) < 1e-10  # a = 2

    def test_deep_chain(self) -> None:
        # y = ((x + 1) * 2)^2
        # At x = 1: y = (2*2)^2 = 16
        # dy/dx = 2 * (x+1) * 2 * 2 = 8(x+1)
        x = Value(1.0)
        y = ((x + 1) * 2) ** 2
        y.backward()
        assert abs(x.grad - 16.0) < 1e-10  # 8 * (1+1) = 16


class TestTopologicalOrder:
    """Test topological ordering of computational graph."""

    def test_simple_graph(self) -> None:
        a = Value(1.0)
        b = Value(2.0)
        c = a + b
        topo = get_topo_order(c)
        # a and b should come before c
        assert topo[-1] == c
        assert a in topo
        assert b in topo

    def test_chain_graph(self) -> None:
        a = Value(1.0)
        b = a + 1
        c = b * 2
        d = c**2
        topo = get_topo_order(d)
        # d should be last, c before d, b before c, a before b
        assert topo[-1] == d
        assert topo.index(c) < topo.index(d)
        assert topo.index(b) < topo.index(c)
        assert topo.index(a) < topo.index(b)


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_zero_value(self) -> None:
        a = Value(0.0)
        b = a * 5
        b.backward()
        assert a.grad == 5.0

    def test_negative_values(self) -> None:
        a = Value(-3.0)
        b = Value(-2.0)
        c = a * b
        c.backward()
        assert c.data == 6.0
        assert a.grad == -2.0
        assert b.grad == -3.0

    def test_repeated_operations(self) -> None:
        a = Value(2.0)
        b = a + a
        b.backward()
        assert a.grad == 2.0  # accumulated from both paths

    def test_repeated_backward(self) -> None:
        a = Value(2.0)
        b = a * 3
        b.backward()
        first_grad = a.grad
        b.backward()
        # Second backward should add to existing gradient
        assert a.grad == first_grad * 2
