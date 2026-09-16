"""
Probability Distributions
==========================

Discrete and continuous probability distributions relevant to neural networks.

Implemented distributions:
- Bernoulli(p): single binary trial
- Binomial(n, p): count of successes in n independent trials
- Categorical(p_vec): single choice among k categories
- Uniform(a, b): continuous uniform distribution
- Normal(mu, sigma): Gaussian distribution

Each distribution provides:
- PMF/PDF evaluation
- Expectation and variance
- Support information
- ML connection explanation

Mathematical Foundation:
- A probability distribution assigns probabilities to outcomes
- Discrete: probability mass function (PMF) P(X = x)
- Continuous: probability density function (PDF) f(x), where P(a ≤ X ≤ b) = ∫_a^b f(x) dx
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.probability.fundamentals import validate_probability

# ---------------------------------------------------------------------------
# Bernoulli distribution
# ---------------------------------------------------------------------------


def bernoulli_pmf(x: int, p: float) -> float:
    """Probability mass function of Bernoulli distribution.

    PMF: P(X = x) = p^x × (1-p)^(1-x)

    Args:
        x: Outcome (0 or 1).
        p: Probability of success (1).

    Returns:
        P(X = x).

    Raises:
        ValueError: If x not in {0, 1} or p not in [0, 1].
    """
    if x not in (0, 1):
        raise ValueError(f"Bernoulli outcome must be 0 or 1, got {x}")
    validate_probability(p, "p")
    if x == 1:
        return p
    return 1.0 - p


def bernoulli_expectation(p: float) -> float:
    """Expected value of Bernoulli(p): E[X] = p.

    Args:
        p: Probability of success.

    Returns:
        Expected value.
    """
    validate_probability(p)
    return p


def bernoulli_variance(p: float) -> float:
    """Variance of Bernoulli(p): Var(X) = p(1-p).

    Args:
        p: Probability of success.

    Returns:
        Variance.
    """
    validate_probability(p)
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# Binomial distribution
# ---------------------------------------------------------------------------


def _log_binomial_coefficient(n: int, k: int) -> float:
    """Compute log of binomial coefficient C(n, k) = log(n! / (k! × (n-k)!)).

    Uses log-gamma function for numerical stability.
    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Probability mass function of Binomial(n, p) distribution.

    PMF: P(X = k) = C(n, k) × p^k × (1-p)^(n-k)

    Args:
        k: Number of successes (0 ≤ k ≤ n).
        n: Number of trials.
        p: Probability of success per trial.

    Returns:
        P(X = k).
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    if not isinstance(k, int) or k < 0 or k > n:
        raise ValueError(f"k must be integer with 0 ≤ k ≤ n, got k={k}, n={n}")
    validate_probability(p, "p")

    log_pmf = (
        _log_binomial_coefficient(n, k)
        + k * math.log(p + 1e-300)
        + (n - k) * math.log(1.0 - p + 1e-300)
    )
    return math.exp(log_pmf)


def binomial_expectation(n: int, p: float) -> float:
    """Expected value of Binomial(n, p): E[X] = np.

    Args:
        n: Number of trials.
        p: Probability of success.

    Returns:
        Expected value.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    validate_probability(p)
    return n * p


def binomial_variance(n: int, p: float) -> float:
    """Variance of Binomial(n, p): Var(X) = np(1-p).

    Args:
        n: Number of trials.
        p: Probability of success.

    Returns:
        Variance.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")
    validate_probability(p)
    return n * p * (1.0 - p)


# ---------------------------------------------------------------------------
# Categorical distribution
# ---------------------------------------------------------------------------


def categorical_pmf(x: int, probs: NDArray) -> float:
    """Probability mass function of Categorical distribution.

    PMF: P(X = x) = probs[x]

    Args:
        x: Category index (0 ≤ x < len(probs)).
        probs: Probability vector (must sum to 1).

    Returns:
        P(X = x).
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got {probs.ndim}D")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")
    if x < 0 or x >= len(probs):
        raise ValueError(f"x = {x} is out of bounds for {len(probs)} categories")
    if probs[x] < 0:
        raise ValueError(f"probs[{x}] = {probs[x]} is negative")
    return float(probs[x])


def categorical_expectation(probs: NDArray, values: NDArray) -> float:
    """Expected value of a categorical random variable.

    E[X] = Σ x_i × p(x_i)

    Args:
        probs: Probability vector.
        values: Values assigned to each category.

    Returns:
        Expected value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(probs) != len(values):
        raise ValueError("probs and values must have same length")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")
    return float(np.sum(probs * values))


def categorical_variance(probs: NDArray, values: NDArray) -> float:
    """Variance of a categorical random variable.

    Var(X) = E[X²] - (E[X])²

    Args:
        probs: Probability vector.
        values: Values assigned to each category.

    Returns:
        Variance.
    """
    probs = np.asarray(probs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(probs) != len(values):
        raise ValueError("probs and values must have same length")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")
    e_x = float(np.sum(probs * values))
    e_x2 = float(np.sum(probs * values**2))
    return e_x2 - e_x**2


# ---------------------------------------------------------------------------
# Uniform distribution (continuous)
# ---------------------------------------------------------------------------


def uniform_pdf(x: float, a: float, b: float) -> float:
    """Probability density function of Uniform(a, b).

    PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise.

    Args:
        x: Point at which to evaluate.
        a: Lower bound.
        b: Upper bound.

    Returns:
        Density at x.
    """
    if a >= b:
        raise ValueError(f"Lower bound a = {a} must be less than upper bound b = {b}")
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def uniform_expectation(a: float, b: float) -> float:
    """Expected value of Uniform(a, b): E[X] = (a + b) / 2.

    Args:
        a: Lower bound.
        b: Upper bound.

    Returns:
        Expected value.
    """
    if a >= b:
        raise ValueError(f"a = {a} must be less than b = {b}")
    return (a + b) / 2.0


def uniform_variance(a: float, b: float) -> float:
    """Variance of Uniform(a, b): Var(X) = (b - a)² / 12.

    Args:
        a: Lower bound.
        b: Upper bound.

    Returns:
        Variance.
    """
    if a >= b:
        raise ValueError(f"a = {a} must be less than b = {b}")
    return (b - a) ** 2 / 12.0


# ---------------------------------------------------------------------------
# Normal (Gaussian) distribution
# ---------------------------------------------------------------------------


def normal_pdf(x: float | NDArray, mu: float, sigma: float) -> float | NDArray:
    """Probability density function of Normal(mu, sigma²).

    PDF: f(x) = (1 / (sigma × sqrt(2π))) × exp(-(x - mu)² / (2σ²))

    Args:
        x: Point(s) at which to evaluate.
        mu: Mean of the distribution.
        sigma: Standard deviation (must be > 0).

    Returns:
        Density at x (scalar or array matching x).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    x_arr = np.asarray(x, dtype=np.float64)
    coefficient = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x_arr - mu) / sigma) ** 2
    result = coefficient * np.exp(exponent)
    return float(result) if result.ndim == 0 else result


def normal_expectation(mu: float, sigma: float) -> float:
    """Expected value of Normal(mu, sigma²): E[X] = mu.

    Args:
        mu: Mean.
        sigma: Standard deviation (must be > 0).

    Returns:
        Expected value (mu).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return mu


def normal_variance(mu: float, sigma: float) -> float:
    """Variance of Normal(mu, sigma²): Var(X) = sigma².

    Args:
        mu: Mean.
        sigma: Standard deviation (must be > 0).

    Returns:
        Variance (sigma²).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return sigma**2


def normal_cdf(x: float | NDArray, mu: float, sigma: float) -> float | NDArray:
    """Cumulative distribution function of Normal(mu, sigma²).

    Uses the error function: Φ(x) = 0.5 × (1 + erf((x - mu) / (sigma × √2)))

    Args:
        x: Point(s) at which to evaluate.
        mu: Mean.
        sigma: Standard deviation (must be > 0).

    Returns:
        CDF value at x.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    x_arr = np.asarray(x, dtype=np.float64)
    result = 0.5 * (1.0 + math.erf((x_arr - mu) / (sigma * math.sqrt(2.0))))
    return float(result) if np.ndim(result) == 0 else result
