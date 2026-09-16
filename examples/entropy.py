"""
Entropy — Uncertainty in Probability Distributions
==================================================

Entropy H(X) = -Σ p(x) × log(p(x)) measures the average uncertainty
in a probability distribution.

Key insights:
- Uniform distribution: maximum entropy (maximum uncertainty)
- Deterministic distribution: zero entropy (no uncertainty)
- Binary (Bernoulli): H(p) = -p×log(p) - (1-p)×log(1-p)

ML Connection:
- High entropy: model is uncertain about predictions
- Low entropy: model is confident about predictions
- Cross-entropy loss measures how well predicted distribution matches target
"""

import numpy as np

from math_for_neural_networks.probability.information import entropy
from math_for_neural_networks.probability.distributions import bernoulli_pmf


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_uniform_entropy() -> None:
    """Uniform distribution has maximum entropy."""
    print_section("Example 1: Uniform Distribution Entropy")

    print("""
    For k equally likely outcomes, entropy H = log(k).
    This is the MAXIMUM possible entropy for k outcomes.
    """)

    for k in [2, 4, 8, 16]:
        uniform = np.ones(k) / k
        h = entropy(uniform)
        print(f"  k = {k:>2}: H = {h:.4f} nats (log({k}) = {np.log(k):.4f})")

    print("""
    As k increases, entropy grows logarithmically.
    More possible outcomes = more uncertainty = more bits needed to encode.
    """)


def example_2_bernoulli_entropy() -> None:
    """Bernoulli entropy as a function of p."""
    print_section("Example 2: Bernoulli Entropy")

    print("""
    For Bernoulli(p): H(p) = -p×log(p) - (1-p)×log(1-p)

    Maximum at p = 0.5 (maximum uncertainty)
    Minimum at p = 0 or p = 1 (certain outcome)
    """)

    print(f"  {'p':>6} | {'H(p)':>8} | {'Certainty':>10}")
    print("  " + "-" * 30)

    for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        probs = np.array([p, 1 - p])
        h = entropy(probs)
        certainty = "certain" if p in (0.0, 1.0) else "uncertain"
        print(f"  {p:>6.2f} | {h:>8.4f} | {certainty:>10}")

    print("""
    p = 0.5: maximum uncertainty (coin flip)
    p = 0.0 or 1.0: zero entropy (outcome is certain)
    """)


def example_3_entropy_and_ml() -> None:
    """How entropy relates to model confidence."""
    print_section("Example 3: Entropy and Model Confidence")

    print("""
    In ML, a model outputs a probability distribution over classes.
    The entropy of this distribution tells us about model confidence:

    - High entropy: model is uncertain (spreads probability across classes)
    - Low entropy: model is confident (concentrates probability on one class)
    """)

    # Model A: very confident
    model_a = np.array([0.95, 0.02, 0.02, 0.01])
    # Model B: uncertain
    model_b = np.array([0.25, 0.25, 0.25, 0.25])
    # Model C: moderately confident
    model_c = np.array([0.6, 0.2, 0.15, 0.05])

    print(f"  Model A (confident):  {model_a} -> H = {entropy(model_a):.4f}")
    print(f"  Model B (uncertain):  {model_b} -> H = {entropy(model_b):.4f}")
    print(f"  Model C (moderate):   {model_c} -> H = {entropy(model_c):.4f}")

    print("""
    Higher entropy means more uncertainty.
    A perfectly uniform model has H = log(k) where k is the number of classes.
    """)


if __name__ == "__main__":
    print("ENTROPY — UNCERTAINTY IN PROBABILITY DISTRIBUTIONS")
    print("=" * 55)

    example_1_uniform_entropy()
    example_2_bernoulli_entropy()
    example_3_entropy_and_ml()
