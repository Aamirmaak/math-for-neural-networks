"""
Probability Fundamentals
========================

Core probability concepts: sample spaces, events, probability axioms,
complement, union, intersection, independence, and conditional probability.

Mathematical Foundation:
- Sample space Ω: the set of all possible outcomes
- Event A ⊆ Ω: a subset of the sample space
- Probability P(A): a number in [0, 1] satisfying the Kolmogorov axioms

Axioms:
1. P(A) ≥ 0 for all events A
2. P(Ω) = 1
3. For mutually exclusive events A₁, A₂, ...: P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...

ML Connection:
- Probability is the foundation of statistical learning
- Model outputs are probability distributions
- Loss functions (cross-entropy) measure how well predicted probabilities match reality
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_probability(p: float, name: str = "probability") -> None:
    """Validate that a value is a valid probability in [0, 1].

    Args:
        p: Value to validate.
        name: Name for error messages.

    Raises:
        ValueError: If p is not in [0, 1].
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def validate_probabilities(probs: NDArray, name: str = "probabilities") -> None:
    """Validate that values are valid probabilities and sum approximately to 1.

    Args:
        probs: Array of probability values.
        name: Name for error messages.

    Raises:
        ValueError: If any value is outside [0, 1] or sum is not approximately 1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError(f"{name} must be in [0, 1], got {probs}")
    total = float(np.sum(probs))
    if abs(total - 1.0) > 1e-10:
        raise ValueError(f"{name} must sum to 1, got {total}")


def validate_sample_space_size(n: int) -> None:
    """Validate that a sample space size is a positive integer.

    Args:
        n: Size of the sample space.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Sample space size must be a positive integer, got {n}")


def probability_axiom_check(probs: NDArray) -> dict[str, bool | float]:
    """Check whether a set of probabilities satisfies the Kolmogorov axioms.

    Axioms:
        1. P(A) >= 0 for all events
        2. P(Ω) = 1 (total probability = 1)
        3. Additivity (assumed from axiom 2 for finite discrete)

    Args:
        probs: Array of probabilities for each outcome in the sample space.

    Returns:
        Dictionary with axiom checks and total probability.
    """
    probs = np.asarray(probs, dtype=np.float64)
    total = float(np.sum(probs))
    all_non_negative = bool(np.all(probs >= 0))
    total_is_one = abs(total - 1.0) < 1e-10
    return {
        "axiom_1_non_negative": all_non_negative,
        "axiom_2_total_one": total_is_one,
        "total_probability": total,
    }


def complement(p: float) -> float:
    """Compute the complement of a probability: P(not A) = 1 - P(A).

    Args:
        p: Probability of event A.

    Returns:
        Probability of the complement event.
    """
    validate_probability(p)
    return 1.0 - p


def union_exclusive(p_a: float, p_b: float) -> float:
    """Compute P(A ∪ B) for mutually exclusive events.

    Formula: P(A ∪ B) = P(A) + P(B)

    Args:
        p_a: Probability of event A.
        p_b: Probability of event B.

    Returns:
        P(A ∪ B).
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b, "P(B)")
    result = p_a + p_b
    if result > 1.0 + 1e-10:
        raise ValueError(f"Union probability exceeds 1: {result}")
    return min(result, 1.0)


def union_general(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """Compute P(A ∪ B) for general events (inclusion-exclusion).

    Formula: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

    Args:
        p_a: Probability of event A.
        p_b: Probability of event B.
        p_a_and_b: Probability of A ∩ B (intersection).

    Returns:
        P(A ∪ B).
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b, "P(B)")
    validate_probability(p_a_and_b, "P(A ∩ B)")
    if p_a_and_b > min(p_a, p_b) + 1e-10:
        raise ValueError(f"P(A ∩ B) = {p_a_and_b} cannot exceed min(P(A), P(B)) = {min(p_a, p_b)}")
    return p_a + p_b - p_a_and_b


def intersection_independent(p_a: float, p_b: float) -> float:
    """Compute P(A ∩ B) for independent events.

    Formula: P(A ∩ B) = P(A) × P(B)

    Args:
        p_a: Probability of event A.
        p_b: Probability of event B.

    Returns:
        P(A ∩ B).
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b, "P(B)")
    return p_a * p_b


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Compute P(A | B) = P(A ∩ B) / P(B).

    Args:
        p_a_and_b: Probability of A ∩ B.
        p_b: Probability of event B.

    Returns:
        P(A | B).

    Raises:
        ValueError: If P(B) = 0 (conditioning on impossible event).
    """
    validate_probability(p_a_and_b, "P(A ∩ B)")
    validate_probability(p_b, "P(B)")
    if p_b < 1e-15:
        raise ValueError(
            "Cannot compute conditional probability: P(B) = 0. "
            "Conditioning on an impossible event is undefined."
        )
    return p_a_and_b / p_b


def is_independent(p_a: float, p_b: float, p_a_and_b: float, tol: float = 1e-10) -> bool:
    """Check whether events A and B are independent.

    Independence means: P(A ∩ B) = P(A) × P(B)

    Args:
        p_a: Probability of event A.
        p_b: Probability of event B.
        p_a_and_b: Probability of A ∩ B.
        tol: Tolerance for floating-point comparison.

    Returns:
        True if events are independent.
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b, "P(B)")
    validate_probability(p_a_and_b, "P(A ∩ B)")
    return abs(p_a_and_b - p_a * p_b) < tol


def bayes_theorem(p_a: float, p_b_given_a: float, p_b: float) -> float:
    """Compute P(A | B) using Bayes' theorem.

    Formula: P(A | B) = P(B | A) × P(A) / P(B)

    Args:
        p_a: Prior probability P(A).
        p_b_given_a: Likelihood P(B | A).
        p_b: Evidence P(B) (marginal probability of B).

    Returns:
        Posterior probability P(A | B).
    """
    validate_probability(p_a, "P(A)")
    validate_probability(p_b_given_a, "P(B | A)")
    validate_probability(p_b, "P(B)")
    if p_b < 1e-15:
        raise ValueError("Cannot compute Bayes theorem: P(B) = 0")
    return (p_b_given_a * p_a) / p_b


def marginal_probability(joint_matrix: NDArray, axis: int = 0) -> NDArray:
    """Compute marginal probability by summing joint distribution.

    Given joint distribution P(X, Y), compute P(X) by summing over Y.

    Args:
        joint_matrix: 2D array where joint_matrix[i, j] = P(X=i, Y=j).
        axis: Axis to sum over (0 for marginalizing Y, 1 for marginalizing X).

    Returns:
        Marginal probability distribution.
    """
    joint_matrix = np.asarray(joint_matrix, dtype=np.float64)
    if joint_matrix.ndim != 2:
        raise ValueError(f"Expected 2D joint matrix, got {joint_matrix.ndim}D")
    result: NDArray = np.sum(joint_matrix, axis=axis)
    return result


def joint_from_conditional(p_y: NDArray, p_x_given_y: NDArray) -> NDArray:
    """Compute joint distribution P(X, Y) = P(X | Y) × P(Y).

    Args:
        p_y: Marginal probability P(Y) as 1D array.
        p_x_given_y: Conditional probability P(X | Y) as 2D array.
            p_x_given_y[i, j] = P(X=i | Y=j).

    Returns:
        Joint distribution P(X, Y) as 2D array.
    """
    p_y = np.asarray(p_y, dtype=np.float64)
    p_x_given_y = np.asarray(p_x_given_y, dtype=np.float64)

    if p_y.ndim != 1:
        raise ValueError(f"P(Y) must be 1D, got {p_y.ndim}D")
    if p_x_given_y.ndim != 2:
        raise ValueError(f"P(X|Y) must be 2D, got {p_x_given_y.ndim}D")
    if p_x_given_y.shape[1] != len(p_y):
        raise ValueError(
            f"P(X|Y) columns ({p_x_given_y.shape[1]}) must match P(Y) length ({len(p_y)})"
        )

    result: NDArray = p_x_given_y * p_y[np.newaxis, :]
    return result
