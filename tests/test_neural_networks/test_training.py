"""Tests for training loop and neural network forward/backward."""

import numpy as np
import pytest

from math_for_neural_networks.neural_networks.training import (
    NetworkParams,
    backward,
    compute_numerical_gradients,
    forward,
    init_params,
    sgd_step,
    train,
)


class TestNetworkParams:
    """Test network parameter initialization and management."""

    def test_init_params_shapes(self) -> None:
        params = init_params(input_dim=3, hidden_dim=4, output_dim=2, seed=42)
        assert params.W1.shape == (4, 3)
        assert params.b1.shape == (4,)
        assert params.W2.shape == (2, 4)
        assert params.b2.shape == (2,)

    def test_init_params_deterministic(self) -> None:
        p1 = init_params(3, 4, 2, seed=42)
        p2 = init_params(3, 4, 2, seed=42)
        np.testing.assert_array_equal(p1.W1, p2.W1)
        np.testing.assert_array_equal(p1.W2, p2.W2)

    def test_get_all_params(self) -> None:
        params = init_params(2, 3, 1, seed=42)
        all_params = params.get_all_params()
        assert len(all_params) == 4
        names = [name for name, _ in all_params]
        assert "W1" in names
        assert "b1" in names
        assert "W2" in names
        assert "b2" in names


class TestForwardPass:
    """Test forward pass through the neural network."""

    def test_single_sample(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([1.0, 2.0])
        output, cache = forward(x, params, "sigmoid")
        assert output.shape == (1,)
        assert "x" in cache
        assert "z1" in cache
        assert "a1" in cache
        assert "z2" in cache

    def test_batch(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output, cache = forward(x, params, "sigmoid")
        assert output.shape == (3, 1)

    def test_deterministic(self) -> None:
        params = init_params(2, 3, 1, seed=42)
        x = np.array([[1.0, 2.0]])
        out1, _ = forward(x, params, "sigmoid")
        out2, _ = forward(x, params, "sigmoid")
        np.testing.assert_array_equal(out1, out2)

    def test_different_activations(self) -> None:
        params = init_params(2, 3, 1, seed=42)
        x = np.array([[1.0, 2.0]])
        for act in ["sigmoid", "tanh", "relu"]:
            output, _ = forward(x, params, act)
            assert output.shape == (1, 1)


class TestBackwardPass:
    """Test backward pass gradient computation."""

    def test_gradient_shapes(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        y_pred, cache = forward(x, params, "sigmoid")
        grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

        assert grads["dW1"].shape == params.W1.shape
        assert grads["db1"].shape == params.b1.shape
        assert grads["dW2"].shape == params.W2.shape
        assert grads["db2"].shape == params.b2.shape

    def test_gradient_values_nonzero(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])

        y_pred, cache = forward(x, params, "sigmoid")
        grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

        for name, grad in grads.items():
            assert np.any(grad != 0), f"Gradient {name} is all zeros"

    def test_backward_mse(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        y_pred, cache = forward(x, params, "sigmoid")
        grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

        # Verify gradients are finite
        for name, grad in grads.items():
            assert np.all(np.isfinite(grad)), f"Gradient {name} contains non-finite values"


class TestSGDStep:
    """Test SGD parameter update."""

    def test_update_changes_params(self) -> None:
        params = init_params(2, 3, 1, seed=42)
        W1_before = params.W1.copy()
        grads = {"dW1": np.ones_like(params.W1),
                 "db1": np.ones_like(params.b1),
                 "dW2": np.ones_like(params.W2),
                 "db2": np.ones_like(params.b2)}
        sgd_step(params, grads, lr=0.1)
        assert not np.array_equal(params.W1, W1_before)

    def test_update_direction(self) -> None:
        params = init_params(2, 3, 1, seed=42)
        grads = {"dW1": np.ones_like(params.W1),
                 "db1": np.ones_like(params.b1),
                 "dW2": np.ones_like(params.W2),
                 "db2": np.ones_like(params.b2)}
        W1_before = params.W1.copy()
        sgd_step(params, grads, lr=0.1)
        # W1 should decrease (W1 - lr * 1)
        np.testing.assert_allclose(params.W1, W1_before - 0.1)


class TestNumericalGradients:
    """Test numerical gradient computation for gradient checking."""

    def test_numerical_vs_analytical_single_layer(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])

        # Analytical gradients
        y_pred, cache = forward(x, params, "sigmoid")
        grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

        # Numerical gradients
        num_grads = compute_numerical_gradients(x, y, params, "sigmoid", "mse")

        for name in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_allclose(
                grads[f"d{name}"], num_grads[name], atol=1e-5, rtol=1e-3,
                err_msg=f"Gradient mismatch for {name}"
            )

    def test_numerical_vs_analytical_batch(self) -> None:
        params = init_params(input_dim=2, hidden_dim=3, output_dim=1, seed=42)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([[1.0], [0.0], [1.0]])

        y_pred, cache = forward(x, params, "sigmoid")
        grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

        num_grads = compute_numerical_gradients(x, y, params, "sigmoid", "mse")

        for name in ["W1", "b1", "W2", "b2"]:
            np.testing.assert_allclose(
                grads[f"d{name}"], num_grads[name], atol=1e-5, rtol=1e-3,
                err_msg=f"Gradient mismatch for {name}"
            )


class TestTraining:
    """Test the training loop."""

    def test_loss_decreases(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 2))
        y = (x[:, 0:1] > 0).astype(float)

        params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
        metrics = train(x, y, params, lr=0.1, iterations=100, activation="sigmoid", loss_type="mse")

        assert len(metrics.loss_history) == 100
        # Loss should generally decrease (not necessarily monotonic)
        assert metrics.loss_history[-1] < metrics.loss_history[0]

    def test_training_reproducible(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        params1 = init_params(2, 3, 1, seed=42)
        m1 = train(x, y, params1, lr=0.01, iterations=50)

        params2 = init_params(2, 3, 1, seed=42)
        m2 = train(x, y, params2, lr=0.01, iterations=50)

        np.testing.assert_array_equal(m1.loss_history, m2.loss_history)

    def test_gradient_norm_recorded(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        params = init_params(2, 3, 1, seed=42)
        metrics = train(x, y, params, lr=0.01, iterations=10)

        assert len(metrics.gradient_norm_history) == 10
        assert all(g > 0 for g in metrics.gradient_norm_history)

    def test_different_activations(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        for act in ["sigmoid", "tanh", "relu"]:
            params = init_params(2, 3, 1, seed=42)
            metrics = train(x, y, params, lr=0.01, iterations=10, activation=act)
            assert len(metrics.loss_history) == 10

    def test_classification_training(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((20, 2))
        y = (x[:, 0:1] > 0).astype(float)

        params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
        metrics = train(x, y, params, lr=0.1, iterations=200, activation="sigmoid", loss_type="bce")

        # Loss should decrease for classification
        assert metrics.loss_history[-1] < metrics.loss_history[0]
