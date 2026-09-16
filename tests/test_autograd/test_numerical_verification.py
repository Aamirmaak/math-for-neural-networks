"""Numerical verification tests for autograd engine."""

import math

import numpy as np
import pytest

from math_for_neural_networks.autograd.value import Value


def numerical_gradient(func: callable, x: float, h: float = 1e-5) -> float:
    """Compute numerical gradient using central differences."""
    return (func(x + h) - func(x - h)) / (2.0 * h)


class TestAutogradNumericalVerification:
    """Verify autograd gradients against numerical differentiation."""

    def test_addition(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = Value(3.0)
            c = a + b
            c.backward()
            return a.grad

        for x in [-2.0, 0.0, 1.0, 5.0]:
            # d/dx (x + 3) = 1
            assert abs(func(x) - 1.0) < 1e-10

    def test_multiplication(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = Value(3.0)
            c = a * b
            c.backward()
            return c.data

        for x in [-2.0, 0.0, 1.0, 5.0]:
            analytical = x * 3.0
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_power(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = a**2
            b.backward()
            return a.grad

        for x in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            analytical = 2.0 * x
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_chain_rule(self) -> None:
        # f(x) = (x + 1)^2
        def func(x: float) -> float:
            a = Value(x)
            b = a + 1
            c = b**2
            c.backward()
            return a.grad

        for x in [-3.0, 0.0, 2.0, 5.0]:
            analytical = 2.0 * (x + 1.0)
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_sigmoid_gradient(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = a.sigmoid()
            b.backward()
            return a.grad

        for x in [-3.0, 0.0, 2.0, 5.0]:
            s = 1.0 / (1.0 + math.exp(-x))
            analytical = s * (1.0 - s)
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_tanh_gradient(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = a.tanh()
            b.backward()
            return a.grad

        for x in [-3.0, 0.0, 2.0, 5.0]:
            t = math.tanh(x)
            analytical = 1.0 - t**2
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_relu_gradient(self) -> None:
        def func(x: float) -> float:
            a = Value(x)
            b = a.relu()
            b.backward()
            return a.grad

        for x in [-3.0, -0.1, 0.1, 3.0]:
            analytical = 1.0 if x > 0 else 0.0
            numerical = func(x)
            assert abs(analytical - numerical) < 1e-10

    def test_complex_function(self) -> None:
        # f(x) = sigmoid(x) * tanh(x)
        def func(x: float) -> float:
            a = Value(x)
            b = a.sigmoid() * a.tanh()
            b.backward()
            return a.grad

        for x in [-2.0, 0.0, 2.0]:
            s = 1.0 / (1.0 + math.exp(-x))
            t = math.tanh(x)
            f_val = s * t
            # Numerical gradient
            h = 1e-5
            s_plus = 1.0 / (1.0 + math.exp(-(x + h)))
            t_plus = math.tanh(x + h)
            s_minus = 1.0 / (1.0 + math.exp(-(x - h)))
            t_minus = math.tanh(x - h)
            numerical = (s_plus * t_plus - s_minus * t_minus) / (2.0 * h)
            analytical = func(x)
            assert abs(analytical - numerical) < 1e-5

    def test_multi_variable(self) -> None:
        # f(x, y) = x^2 * y
        def func_f(x: float, y: float) -> float:
            a = Value(x)
            b = Value(y)
            c = a**2 * b
            c.backward()
            return a.grad, b.grad

        x, y = 2.0, 3.0
        dx_analytical, dy_analytical = func_f(x, y)
        # df/dx = 2xy = 12
        # df/dy = x^2 = 4
        assert abs(dx_analytical - 12.0) < 1e-10
        assert abs(dy_analytical - 4.0) < 1e-10


class TestNumericalVsAutogradDetailed:
    """More detailed numerical verification with error reporting."""

    def _check_gradient(
        self,
        func: callable,
        grad_func: callable,
        x_val: float,
        h: float = 1e-5,
        rtol: float = 1e-5,
        atol: float = 1e-7,
    ) -> None:
        """Helper to check gradient and report details."""
        analytical = grad_func(x_val)
        numerical = (func(x_val + h) - func(x_val - h)) / (2.0 * h)

        abs_error = abs(analytical - numerical)
        denom = max(abs(analytical), abs(numerical))
        rel_error = abs_error / denom if denom > 0 else 0.0

        assert abs_error <= atol + rtol * abs(numerical), (
            f"Gradient check failed at x={x_val}: "
            f"analytical={analytical:.10f}, numerical={numerical:.10f}, "
            f"abs_error={abs_error:.2e}, rel_error={rel_error:.2e}"
        )

    def test_sigmoid_chain(self) -> None:
        # f(x) = sigmoid(x^2)
        def func(x: float) -> float:
            a = Value(x)
            b = a**2
            c = b.sigmoid()
            return c.data

        def ref(x: float) -> float:
            s = 1.0 / (1.0 + math.exp(-x**2))
            return s * (1.0 - s) * 2.0 * x

        for x in [-2.0, -1.0, 0.5, 1.0, 2.0]:
            self._check_gradient(func, ref, x)

    def test_tanh_squared(self) -> None:
        # f(x) = tanh(x)^2
        def func(x: float) -> float:
            a = Value(x)
            b = a.tanh()
            c = b**2
            return c.data

        def ref(x: float) -> float:
            t = math.tanh(x)
            return 2.0 * t * (1.0 - t**2)

        for x in [-2.0, 0.0, 1.0, 3.0]:
            self._check_gradient(func, ref, x)

    def test_exp_times_linear(self) -> None:
        # f(x) = x * exp(x)
        def func(x: float) -> float:
            a = Value(x)
            b = a * a.exp()
            return b.data

        def ref(x: float) -> float:
            return math.exp(x) + x * math.exp(x)

        for x in [-2.0, 0.0, 1.0, 2.0]:
            self._check_gradient(func, ref, x)
