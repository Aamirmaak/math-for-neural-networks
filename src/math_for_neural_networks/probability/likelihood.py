"""
Likelihood, Log-Likelihood, and Maximum Likelihood Estimation
=============================================================

Likelihood:
    L(θ | data) measures how plausible different parameter values θ
    are given the observed data. It is a function of θ, not of data.

    Key distinction: probability vs likelihood
    - Probability: P(data | θ) — given parameters, what is the data probability?
    - Likelihood: L(θ | data) — given data, how likely are the parameters?
    Mathematically they are the same function, but the interpretation differs.

Log-Likelihood:
    log L(θ | data) = Σ log P(x_i | θ)

    Products become sums, which are:
    1. Numerically more stable (avoid underflow)
    2. Easier to differentiate (for future optimization)
    3. Directly related to cross-entropy loss

Maximum Likelihood Estimation (MLE):
    θ_MLE = argmax_θ L(θ | data)
          = argmax_θ log L(θ | data)

    For many common distributions, MLE has a closed-form solution:
    - Bernoulli: p_MLE = (count of successes) / (total trials)
    - Gaussian: μ_MLE = sample mean, σ²_MLE = sample variance

ML Connection:
    - MLE is the foundation of supervised learning
    - Training a neural network with cross-entropy loss = MLE under categorical model
    - Log-likelihood = negative cross-entropy (up to a constant)
    - Gradient descent optimizes (approximately) log-likelihood
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from math_for_neural_networks.probability.fundamentals import validate_probability


def bernoulli_likelihood(data: NDArray, p: float) -> float:
    """Compute likelihood of Bernoulli data given parameter p.

    L(p | data) = Π p^x_i × (1-p)^(1-x_i)
                = p^(Σ x_i) × (1-p)^(n - Σ x_i)

    Args:
        data: Binary data (array of 0s and 1s).
        p: Parameter (probability of success).

    Returns:
        Likelihood value.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Bernoulli data must contain only 0s and 1s")
    validate_probability(p, "p")

    successes = float(np.sum(data))
    failures = len(data) - successes

    # Use log-likelihood to avoid underflow, then exponentiate
    log_lik = successes * math.log(p + 1e-300) + failures * math.log(1.0 - p + 1e-300)
    return math.exp(log_lik)


def bernoulli_log_likelihood(data: NDArray, p: float) -> float:
    """Compute log-likelihood of Bernoulli data given parameter p.

    log L(p | data) = Σ x_i × log(p) + (1-x_i) × log(1-p)

    Args:
        data: Binary data (array of 0s and 1s).
        p: Parameter (probability of success).

    Returns:
        Log-likelihood value.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Bernoulli data must contain only 0s and 1s")
    validate_probability(p, "p")

    successes = float(np.sum(data))
    failures = len(data) - successes

    return successes * math.log(p + 1e-300) + failures * math.log(1.0 - p + 1e-300)


def bernoulli_mle(data: NDArray) -> float:
    """Compute maximum likelihood estimate for Bernoulli parameter p.

    MLE: p_MLE = (number of successes) / (total trials)
              = (Σ x_i) / n

    This is the sample mean of binary data.

    Args:
        data: Binary data (array of 0s and 1s).

    Returns:
        MLE estimate of p.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Bernoulli data must contain only 0s and 1s")
    if len(data) == 0:
        raise ValueError("data must not be empty")

    return float(np.mean(data))


def gaussian_log_likelihood(data: NDArray, mu: float, sigma: float) -> float:
    """Compute log-likelihood of Gaussian data given parameters mu, sigma.

    log L(mu, sigma | data) = Σ log N(x_i | mu, sigma²)
    = -n/2 × log(2π) - n × log(sigma) - Σ(x_i - mu)² / (2σ²)

    Args:
        data: Observed data.
        mu: Mean parameter.
        sigma: Standard deviation parameter (must be > 0).

    Returns:
        Log-likelihood value.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    n = len(data)
    return (
        -n / 2.0 * math.log(2.0 * math.pi)
        - n * math.log(sigma)
        - float(np.sum((data - mu) ** 2)) / (2.0 * sigma**2)
    )


def gaussian_mle(data: NDArray) -> tuple[float, float]:
    """Compute maximum likelihood estimates for Gaussian parameters.

    MLE:
        μ_MLE = (1/n) × Σ x_i  (sample mean)
        σ²_MLE = (1/n) × Σ (x_i - μ_MLE)²  (population variance)

    Note: σ²_MLE uses n, not n-1 (biased estimator).
    For unbiased estimate, use n-1 (Bessel's correction).

    Args:
        data: Observed data.

    Returns:
        Tuple of (mu_MLE, sigma_MLE).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if len(data) == 0:
        raise ValueError("data must not be empty")

    mu_mle = float(np.mean(data))
    sigma_mle = float(np.sqrt(np.mean((data - mu_mle) ** 2)))
    return mu_mle, sigma_mle


def categorical_log_likelihood(data: NDArray, log_probs: NDArray) -> float:
    """Compute log-likelihood of categorical data.

    log L = Σ_i log P(x_i | probs)

    Args:
        data: Integer category labels.
        log_probs: Log-probability vector (log(softmax(logits))).

    Returns:
        Log-likelihood value.
    """
    data = np.asarray(data, dtype=np.int64)
    log_probs = np.asarray(log_probs, dtype=np.float64)

    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if log_probs.ndim != 1:
        raise ValueError(f"log_probs must be 1D, got {log_probs.ndim}D")
    if np.any(data < 0) or np.any(data >= len(log_probs)):
        raise ValueError(f"data values must be in [0, {len(log_probs) - 1}]")

    return float(np.sum(log_probs[data]))
