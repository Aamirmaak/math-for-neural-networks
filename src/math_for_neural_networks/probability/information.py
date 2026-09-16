"""
Information Theory — Entropy, Cross-Entropy, KL Divergence
==========================================================

Information-theoretic measures quantify uncertainty, information content,
and divergence between probability distributions.

Shannon Entropy: H(X) = -Σ p(x) × log p(x)
    Measures uncertainty. Higher entropy = more uncertain.
    Maximum for uniform distribution. Zero for deterministic outcomes.

Cross-Entropy: H(p, q) = -Σ p(x) × log q(x)
    Measures average bits needed to encode data from p using model q.
    Used as classification loss: target distribution p, predicted distribution q.

KL Divergence: D_KL(P || Q) = Σ p(x) × log(p(x) / q(x))
    Measures information lost when Q approximates P.
    NOT symmetric: D_KL(P || Q) ≠ D_KL(Q || P).
    NOT a metric (violates triangle inequality).
    Non-negative: D_KL(P || Q) ≥ 0 with equality iff P = Q.

Numerical Stability:
- Use log-probabilities to avoid underflow
- For p(x) = 0: 0 × log(0) = 0 by convention (limit)
- For q(x) = 0 but p(x) > 0: KL diverges to infinity

ML Connection:
- Cross-entropy is the standard classification loss
- KL divergence is used in variational inference (VAE), distillation
- Entropy measures model confidence/uncertainty
- Log-probabilities are used in language models
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def entropy(probs: NDArray, base: float = np.e) -> float:
    """Compute Shannon entropy H(X) = -Σ p(x) × log_b(p(x)).

    Uses the convention: 0 × log(0) = 0.

    Args:
        probs: Probability distribution p(x). Must sum to 1.
        base: Logarithm base. Default is natural log (nats).
            Use base=2 for bits.

    Returns:
        Entropy in the specified units (nats or bits).
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got {probs.ndim}D")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"probs must sum to 1, got {total}")

    # Use the convention: 0 log 0 = 0
    log_probs = np.zeros_like(probs)
    nonzero = probs > 0
    log_probs[nonzero] = np.log(probs[nonzero]) / np.log(base)

    return float(-np.sum(probs * log_probs))


def cross_entropy(p: NDArray, q: NDArray, base: float = np.e) -> float:
    """Compute cross-entropy H(p, q) = -Σ p(x) × log_b(q(x)).

    Measures the average number of bits needed to encode data from p
    when using a model q for encoding.

    Uses the convention: 0 × log(0) = 0.
    When p(x) > 0 and q(x) = 0, cross-entropy is infinity.

    Args:
        p: True/target distribution. Must sum to 1.
        q: Predicted/model distribution. Must sum to 1.
        base: Logarithm base. Default is natural log.

    Returns:
        Cross-entropy value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.ndim != 1 or q.ndim != 1:
        raise ValueError(f"p and q must be 1D, got {p.ndim}D and {q.ndim}D")
    if len(p) != len(q):
        raise ValueError(f"p and q must have same length, got {len(p)} and {len(q)}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("p and q must be non-negative")
    total_p = float(np.sum(p))
    total_q = float(np.sum(q))
    if abs(total_p - 1.0) > 1e-10:
        raise ValueError(f"p must sum to 1, got {total_p}")
    if abs(total_q - 1.0) > 1e-10:
        raise ValueError(f"q must sum to 1, got {total_q}")

    # Check for p(x) > 0 and q(x) = 0 (infinite cross-entropy)
    if np.any((p > 0) & (q == 0)):
        return float("inf")

    log_q = np.zeros_like(q)
    nonzero = q > 0
    log_q[nonzero] = np.log(q[nonzero]) / np.log(base)

    return float(-np.sum(p * log_q))


def kl_divergence(p: NDArray, q: NDArray, base: float = np.e) -> float:
    """Compute KL divergence D_KL(P || Q) = Σ p(x) × log(p(x) / q(x)).

    Measures information lost when Q approximates P.
    NOT symmetric: D_KL(P || Q) ≠ D_KL(Q || P).
    NOT a metric (does not satisfy triangle inequality).
    Non-negative: D_KL(P || Q) ≥ 0 with equality iff P = Q.

    When p(x) > 0 and q(x) = 0, D_KL diverges to infinity.

    Args:
        p: True distribution. Must sum to 1.
        q: Approximate distribution. Must sum to 1.
        base: Logarithm base. Default is natural log.

    Returns:
        KL divergence value.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.ndim != 1 or q.ndim != 1:
        raise ValueError(f"p and q must be 1D, got {p.ndim}D and {q.ndim}D")
    if len(p) != len(q):
        raise ValueError(f"p and q must have same length, got {len(p)} and {len(q)}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("p and q must be non-negative")
    total_p = float(np.sum(p))
    total_q = float(np.sum(q))
    if abs(total_p - 1.0) > 1e-10:
        raise ValueError(f"p must sum to 1, got {total_p}")
    if abs(total_q - 1.0) > 1e-10:
        raise ValueError(f"q must sum to 1, got {total_q}")

    # Check for p(x) > 0 and q(x) = 0 (infinite KL)
    if np.any((p > 0) & (q == 0)):
        return float("inf")

    # D_KL(P || Q) = Σ p(x) log(p(x)/q(x))
    # Only sum where p(x) > 0 (0 log 0 = 0 convention)
    nonzero = p > 0
    ratio = np.ones_like(p)
    ratio[nonzero] = p[nonzero] / q[nonzero]
    log_ratio = np.zeros_like(p)
    log_ratio[nonzero] = np.log(ratio[nonzero]) / np.log(base)

    return float(np.sum(p * log_ratio))


def log_probability(p: NDArray | float) -> NDArray | float:
    """Compute log probability with stable handling of p = 0.

    Uses the convention: log(0) = -inf.

    ML systems work with log probabilities to:
    1. Avoid numerical underflow (products of small numbers)
    2. Convert products to sums: log(P(x₁)×P(x₂)...) = log P(x₁) + log P(x₂) + ...
    3. Simplify optimization (log is monotonic)

    Args:
        p: Probability value(s) in [0, 1].

    Returns:
        Natural logarithm of the probability.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if np.any(p_arr < 0) or np.any(p_arr > 1):
        raise ValueError("log_probability requires p in [0, 1]")
    return np.where(p_arr > 0, np.log(p_arr), -np.inf)
