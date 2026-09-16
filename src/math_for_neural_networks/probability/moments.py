"""
Moments — Expectation, Variance, Standard Deviation
====================================================

Statistical moments describe the shape of probability distributions.

First moment: E[X] — expectation (location/center of mass)
Second central moment: Var(X) = E[(X - E[X])²] — variance (spread)
Standard deviation: σ = √Var(X) — spread in original units

Mathematical Properties:
- E[aX + b] = aE[X] + b (linearity of expectation)
- Var(aX + b) = a²Var(X) (scaling property)
- Var(X) = E[X²] - (E[X])² (computational formula)

ML Connection:
- Expectation: average model behavior, expected loss
- Variance: uncertainty, data spread, regularization, normalization
- Standard deviation: units of the original variable, used in z-scores
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expectation(values: NDArray, probs: NDArray | None = None) -> float:
    """Compute expectation E[X].

    For discrete distribution:
        E[X] = Σ x_i × p(x_i)

    For a sample (empirical mean):
        E[X] ≈ (1/n) × Σ x_i

    Args:
        values: Values x_i.
        probs: Probability weights p(x_i). If None, treats values as equally likely (sample mean).

    Returns:
        Expected value.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.ndim}D")

    if probs is None:
        return float(np.mean(values))

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got {probs.ndim}D")
    if len(values) != len(probs):
        raise ValueError("values and probs must have same length")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")

    return float(np.sum(values * probs))


def variance(values: NDArray, probs: NDArray | None = None, ddof: int = 0) -> float:
    """Compute variance Var(X) = E[(X - E[X])²].

    Two equivalent formulations:
        Var(X) = E[(X - μ)²]  (definition)
        Var(X) = E[X²] - E[X]²  (computational formula)

    For a sample (ddof=0):
        Var(X) = (1/n) × Σ (x_i - x̄)²

    For sample estimate (ddof=1):
        Var(X) = (1/(n-1)) × Σ (x_i - x̄)²

    Args:
        values: Values x_i.
        probs: Probability weights. If None, computes sample variance.
        ddof: Delta degrees of freedom (0 for population, 1 for sample estimate).

    Returns:
        Variance.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.ndim}D")

    if probs is None:
        return float(np.var(values, ddof=ddof))

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got {probs.ndim}D")
    if len(values) != len(probs):
        raise ValueError("values and probs must have same length")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")

    mu = float(np.sum(values * probs))
    return float(np.sum(probs * (values - mu) ** 2))


def standard_deviation(values: NDArray, probs: NDArray | None = None, ddof: int = 0) -> float:
    """Compute standard deviation σ = √Var(X).

    Args:
        values: Values x_i.
        probs: Probability weights. If None, computes sample std.
        ddof: Delta degrees of freedom (0 for population, 1 for sample estimate).

    Returns:
        Standard deviation.
    """
    return float(np.sqrt(variance(values, probs, ddof=ddof)))


def expectation_linear(
    a: float, values: NDArray, b: float = 0.0, probs: NDArray | None = None
) -> float:
    """Verify linearity of expectation: E[aX + b] = aE[X] + b.

    Args:
        a: Scaling coefficient.
        values: Values x_i.
        b: Shift coefficient.
        probs: Probability weights.

    Returns:
        E[aX + b].
    """
    e_x = expectation(values, probs)
    return a * e_x + b
