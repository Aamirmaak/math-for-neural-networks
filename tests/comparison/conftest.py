"""Configuration for PyTorch comparison tests.

Provides skip markers and fixtures for PyTorch availability detection.
"""

from __future__ import annotations

import pytest

# Attempt to import torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed (pip install torch)",
)


def assert_close(
    a,
    b,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    operation: str = "",
) -> None:
    """Assert two NumPy arrays are approximately equal with diagnostic output.

    Uses criterion: |a - b| <= atol + rtol * |b|

    Args:
        a: First array (NumPy).
        b: Second array (NumPy).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        operation: Name of operation for diagnostic messages.
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        msg = f"Shape mismatch [{operation}]: a={a.shape}, b={b.shape}"
        raise AssertionError(msg)

    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    threshold = atol + rtol * np.abs(b)
    max_threshold = float(np.max(threshold))

    passed = bool(np.all(diff <= threshold))

    if not passed:
        fail_count = int(np.sum(diff > threshold))
        total = a.size
        msg = (
            f"Numerical comparison failed [{operation}]:\n"
            f"  Shape: {a.shape}\n"
            f"  Max absolute error: {max_diff:.2e}\n"
            f"  Mean absolute error: {mean_diff:.2e}\n"
            f"  Max threshold: {max_threshold:.2e}\n"
            f"  Failed elements: {fail_count}/{total}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        raise AssertionError(msg)


def assert_close_scalar(
    a: float,
    b: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    operation: str = "",
) -> None:
    """Assert two scalars are approximately equal.

    Args:
        a: First scalar.
        b: Second scalar.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        operation: Name of operation for diagnostic messages.
    """
    diff = abs(a - b)
    threshold = atol + rtol * abs(b)

    if diff > threshold:
        msg = (
            f"Scalar comparison failed [{operation}]:\n"
            f"  Values: {a:.6e} vs {b:.6e}\n"
            f"  Absolute error: {diff:.2e}\n"
            f"  Threshold: {threshold:.2e}\n"
            f"  rtol={rtol}, atol={atol}"
        )
        raise AssertionError(msg)
