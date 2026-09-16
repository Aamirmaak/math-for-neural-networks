"""Tests for backpropagation utilities."""

import numpy as np
import pytest

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


class TestGradientCheckScalar:
    """Test scalar gradient checking utility."""

    def test_linear_function(self) -> None:
        def f(x: float) -> float:
            return 2.0 * x + 3.0

        def df(x: float) -> float:
            return 2.0

        for x in [-1.0, 0.0, 1.0, 5.0]:
            result = gradient_check_scalar(f, df, x)
            assert result["passed"]

    def test_quadratic(self) -> None:
        def f(x: float) -> float:
            return x**2

        def df(x: float) -> float:
            return 2.0 * x

        for x in [-2.0, 0.0, 1.0, 3.0]:
            result = gradient_check_scalar(f, df, x)
            assert result["passed"]

    def test_sigmoid(self) -> None:
        import math

        def f(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        def df(x: float) -> float:
            s = 1.0 / (1.0 + math.exp(-x))
            return s * (1.0 - s)

        for x in [-3.0, 0.0, 3.0]:
            result = gradient_check_scalar(f, df, x)
            assert result["passed"]


class TestGradientCheck:
    """Test multi-variable gradient checking utility."""

    def test_linear(self) -> None:
        def f(x: np.ndarray) -> float:
            return float(2.0 * x[0] + 3.0 * x[1])

        def df(x: np.ndarray) -> np.ndarray:
            return np.array([2.0, 3.0])

        result = gradient_check(f, df, np.array([1.0, 2.0]))
        assert result["passed"]

    def test_quadratic(self) -> None:
        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + x[1] ** 2)

        def df(x: np.ndarray) -> np.ndarray:
            return np.array([2.0 * x[0], 2.0 * x[1]])

        result = gradient_check(f, df, np.array([1.0, 2.0]))
        assert result["passed"]

    def test_nonlinear(self) -> None:
        import math

        def f(x: np.ndarray) -> float:
            return float(math.sin(x[0]) * math.cos(x[1]))

        def df(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    math.cos(x[0]) * math.cos(x[1]),
                    -math.sin(x[0]) * math.sin(x[1]),
                ]
            )

        result = gradient_check(f, df, np.array([0.5, 0.5]))
        assert result["passed"]


class TestAffineBackward:
    """Test affine layer backward pass gradients."""

    def test_single_sample(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b = np.array([0.1, 0.2])
        dout = np.array([1.0, 0.5])

        dx, dW, db = affine_backward(x, W, b, dout)

        # Manual computation:
        # Z = W @ x + b
        # dL/dx = W^T @ dout
        # dL/dW = outer(dout, x)
        # dL/db = dout
        expected_dx = W.T @ dout
        expected_dW = np.outer(dout, x)
        expected_db = dout

        np.testing.assert_allclose(dx, expected_dx)
        np.testing.assert_allclose(dW, expected_dW)
        np.testing.assert_allclose(db, expected_db)

    def test_batch(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        b = np.array([0.1, 0.2, 0.3])
        dout = np.array([[1.0, 0.5, 0.2], [0.3, 0.7, 0.1]])

        dx, dW, db = affine_backward(x, W, b, dout)

        # Manual computation:
        # Z = x @ W^T + b
        # dL/dX = dout @ W
        # dL/dW = dout^T @ x
        # dL/db = sum(dout, axis=0)
        expected_dx = dout @ W
        expected_dW = dout.T @ x
        expected_db = np.sum(dout, axis=0)

        np.testing.assert_allclose(dx, expected_dx)
        np.testing.assert_allclose(dW, expected_dW)
        np.testing.assert_allclose(db, expected_db)

    def test_shape_consistency(self) -> None:
        batch, inp, out = 4, 3, 2
        x = np.random.randn(batch, inp)
        W = np.random.randn(out, inp)
        b = np.random.randn(out)
        dout = np.random.randn(batch, out)

        dx, dW, db = affine_backward(x, W, b, dout)

        assert dx.shape == x.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape


class TestActivationBackward:
    """Test activation function backward passes."""

    def test_sigmoid_backward(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        dout = np.array([1.0, 1.0, 1.0])

        result = sigmoid_backward(x, dout)

        a = 1.0 / (1.0 + np.exp(-x))
        expected = a * (1.0 - a)

        np.testing.assert_allclose(result, expected)

    def test_relu_backward(self) -> None:
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        dout = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = relu_backward(x, dout)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_tanh_backward(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        dout = np.array([1.0, 1.0, 1.0])

        result = tanh_backward(x, dout)

        a = np.tanh(x)
        expected = 1.0 - a**2

        np.testing.assert_allclose(result, expected)

    def test_sigmoid_backward_chain(self) -> None:
        # Test with non-unit upstream gradient
        x = np.array([0.0, 1.0, -1.0])
        dout = np.array([2.0, 3.0, 0.5])

        result = sigmoid_backward(x, dout)

        a = 1.0 / (1.0 + np.exp(-x))
        expected = dout * a * (1.0 - a)

        np.testing.assert_allclose(result, expected)


class TestLossBackward:
    """Test loss function backward passes."""

    def test_mse_backward(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.0])

        result = mse_backward(y_true, y_pred)

        # dL/dy_pred = (2/n) * (y_pred - y_true)
        expected = (2.0 / 3.0) * (y_pred - y_true)

        np.testing.assert_allclose(result, expected)

    def test_mse_backward_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = mse_backward(y_true, y_pred)

        np.testing.assert_allclose(result, np.zeros(3))


class TestSoftmaxCrossEntropyBackward:
    """Test softmax + cross-entropy gradient identity."""

    def test_softmax_cross_entropy_gradient(self) -> None:
        # Key identity: dL/dz = softmax(z) - y
        logits = np.array([2.0, 1.0, 0.5])
        targets = np.array([1.0, 0.0, 0.0])

        result = softmax_backward(logits, targets)

        # Compute softmax
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)

        expected = probs - targets

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_batch_softmax_cross_entropy(self) -> None:
        logits = np.array([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        result = softmax_backward(logits, targets)

        # Compute softmax for each row
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        expected = probs - targets

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sigmoid_bce_gradient(self) -> None:
        # Key identity: dL/dz = (1/n) * (sigmoid(z) - y)
        # The 1/n factor comes from the mean over the batch
        logits = np.array([0.0, 1.0, -1.0])
        targets = np.array([1.0, 0.0, 1.0])

        result = sigmoid_bce_backward(logits, targets)

        probs = 1.0 / (1.0 + np.exp(-logits))
        n = targets.size
        expected = (probs - targets) / n

        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sigmoid_bce_gradient_numerical(self) -> None:
        """Numerically verify sigmoid + BCE gradient."""
        logits = np.array([0.5, -0.5, 1.0])
        targets = np.array([1.0, 0.0, 1.0])
        h = 1e-5

        # Numerical gradient
        numerical = np.zeros_like(logits)
        for i in range(len(logits)):
            logits_plus = logits.copy()
            logits_minus = logits.copy()
            logits_plus[i] += h
            logits_minus[i] -= h

            probs_plus = 1.0 / (1.0 + np.exp(-logits_plus))
            probs_minus = 1.0 / (1.0 + np.exp(-logits_minus))

            # BCE for each element
            eps = 1e-15
            bce_plus = -np.mean(
                targets * np.log(np.clip(probs_plus, eps, 1 - eps))
                + (1 - targets) * np.log(np.clip(1 - probs_plus, eps, 1 - eps))
            )
            bce_minus = -np.mean(
                targets * np.log(np.clip(probs_minus, eps, 1 - eps))
                + (1 - targets) * np.log(np.clip(1 - probs_minus, eps, 1 - eps))
            )

            numerical[i] = (bce_plus - bce_minus) / (2.0 * h)

        analytical = sigmoid_bce_backward(logits, targets)

        np.testing.assert_allclose(analytical, numerical, atol=1e-5)
