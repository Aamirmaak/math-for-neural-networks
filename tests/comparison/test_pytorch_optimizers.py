"""PyTorch comparison: optimizer parity.

Compares our optimizer update equations against PyTorch equivalents.

Tests:
- SGD (vanilla gradient descent) step
- SGD with momentum step
- Adam step
- Multi-step convergence behavior
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.comparison.conftest import assert_close, requires_torch

torch = pytest.importorskip("torch", reason="PyTorch not installed")


@requires_torch
class TestSGDParity:
    """Compare vanilla SGD update."""

    def test_single_step(self) -> None:
        """SGD step matches PyTorch."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))

        # Our implementation: manual SGD step
        grad = 2.0 * x_np
        lr = 0.01
        our_x = x_np - lr * grad

        # PyTorch SGD
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        loss = torch.sum(x_t**2)
        loss.backward()

        optimizer = torch.optim.SGD([x_t], lr=0.01, momentum=0.0)
        optimizer.step()

        pt_x = x_t.detach().numpy()
        assert_close(our_x, pt_x, operation="SGD single step")


@requires_torch
class TestMomentumParity:
    """Compare SGD with momentum update."""

    def test_single_step(self) -> None:
        """Momentum step matches PyTorch SGD with momentum."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        grad_np = rng.standard_normal((3,))
        lr = 0.01

        # Our implementation: v = beta*v + grad, x = x - lr*v
        v = grad_np  # v_0 = 0, v_1 = 0*beta + grad = grad
        our_x = x_np - lr * v

        # PyTorch SGD with momentum
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        grad_t = torch.tensor(grad_np, dtype=torch.float64)

        optimizer = torch.optim.SGD([x_t], lr=0.01, momentum=0.9)
        x_t.grad = grad_t
        optimizer.step()

        pt_x = x_t.detach().numpy()
        # Note: PyTorch SGD does not apply weight decay by default
        # and uses the same v = momentum * v + grad convention
        # For the first step, our v=grad matches PyTorch's behavior
        # The slight difference is because PyTorch applies lr to (momentum*v + grad)
        # which on first step is lr * (0.9*0 + grad) = lr * grad
        # Our implementation: v = 0.9*0 + grad = grad, then x -= lr*v = lr*grad
        assert_close(our_x, pt_x, rtol=1e-5, operation="momentum single step")

    def test_multi_step(self) -> None:
        """Momentum multi-step comparison."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        lr = 0.01
        beta = 0.9

        # Our implementation: run 5 steps
        v = np.zeros_like(x_np)
        x_ours = x_np.copy()
        for _ in range(5):
            grad = 2.0 * x_ours
            v = beta * v + grad
            x_ours = x_ours - lr * v

        # PyTorch: run 5 steps
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([x_t], lr=0.01, momentum=0.9)

        for _ in range(5):
            loss = torch.sum(x_t**2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pt_x = x_t.detach().numpy()
        # After multiple steps with same gradients, should converge
        assert_close(x_ours, pt_x, rtol=1e-4, operation="momentum multi-step")


@requires_torch
class TestAdamParity:
    """Compare Adam optimizer update."""

    def test_single_step(self) -> None:
        """Adam step matches PyTorch."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        grad_np = rng.standard_normal((3,))
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 1

        # Our implementation
        m = (1 - beta1) * grad_np
        v = (1 - beta2) * (grad_np**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        our_x = x_np - lr * m_hat / (np.sqrt(v_hat) + eps)

        # PyTorch Adam
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        grad_t = torch.tensor(grad_np, dtype=torch.float64)

        optimizer = torch.optim.Adam([x_t], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        x_t.grad = grad_t
        optimizer.step()

        pt_x = x_t.detach().numpy()
        assert_close(our_x, pt_x, rtol=1e-5, operation="Adam single step")

    def test_multi_step(self) -> None:
        """Adam multi-step comparison."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3,))
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # Our implementation: run 10 steps
        m = np.zeros_like(x_np)
        v = np.zeros_like(x_np)
        x_ours = x_np.copy()
        for step in range(10):
            grad = 2.0 * x_ours
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1 ** (step + 1))
            v_hat = v / (1 - beta2 ** (step + 1))
            x_ours = x_ours - lr * m_hat / (np.sqrt(v_hat) + eps)

        # PyTorch: run 10 steps
        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([x_t], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        for _ in range(10):
            loss = torch.sum(x_t**2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pt_x = x_t.detach().numpy()
        assert_close(x_ours, pt_x, rtol=1e-4, operation="Adam multi-step")
