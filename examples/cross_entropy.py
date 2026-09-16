"""
Cross-Entropy and Classification Loss
======================================

Cross-entropy H(p, q) = -Σ p(x) × log(q(x)) measures the average
number of bits needed to encode data from distribution p using model q.

Key insight: For one-hot targets, cross-entropy = -log(predicted probability of true class).

This is THE standard loss function for classification in neural networks.
"""

import numpy as np

from math_for_neural_networks.probability.information import cross_entropy, entropy, kl_divergence
from math_for_neural_networks.probability.stability import softmax


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_one_hot_cross_entropy() -> None:
    """Cross-entropy with one-hot targets simplifies to -log(p_true)."""
    print_section("Example 1: One-Hot Cross-Entropy")

    print("""
    When target is one-hot (only one class is correct):
        H(p, q) = -log(q(correct_class))

    This is just the negative log probability of the correct class!
    """)

    target = np.array([0, 0, 1, 0])  # true class is index 2
    predicted_probs = np.array([0.1, 0.2, 0.5, 0.2])

    ce = cross_entropy(target, predicted_probs)
    ce_manual = -np.log(predicted_probs[2])

    print(f"  Target (one-hot): {target}")
    print(f"  Predicted probs:  {predicted_probs}")
    print(f"  Cross-entropy:    {ce:.4f}")
    print(f"  -log(p_true):     {ce_manual:.4f}")
    print(f"  Match:            {np.isclose(ce, ce_manual)}")


def example_2_relationship_to_entropy() -> None:
    """H(p, q) = H(p) + D_KL(p || q)"""
    print_section("Example 2: Cross-Entropy = Entropy + KL Divergence")

    print("""
    Cross-entropy decomposes as:
        H(p, q) = H(p) + D_KL(p || q)

    where:
        H(p) = entropy of the true distribution (constant during training)
        D_KL(p || q) = how much q diverges from p (what we minimize)

    Minimizing cross-entropy = minimizing KL divergence = making q close to p.
    """)

    p = np.array([0.3, 0.5, 0.2])
    q = np.array([0.4, 0.3, 0.3])

    h_p = entropy(p)
    kl = kl_divergence(p, q)
    ce = cross_entropy(p, q)

    print(f"  True distribution p:      {p}")
    print(f"  Predicted distribution q: {q}")
    print(f"  Entropy H(p):             {h_p:.4f}")
    print(f"  KL(p||q):                 {kl:.4f}")
    print(f"  Cross-entropy H(p,q):     {ce:.4f}")
    print(f"  H(p) + KL(p||q):          {h_p + kl:.4f}")
    print(f"  Match: {np.isclose(ce, h_p + kl)}")


def example_3_softmax_and_cross_entropy() -> None:
    """Softmax converts logits to probabilities; cross-entropy measures loss."""
    print_section("Example 3: Softmax -> Cross-Entropy Pipeline")

    print("""
    In neural network classification:
    1. Model outputs raw scores (logits): z = [2.0, 1.0, 0.5]
    2. Softmax converts to probabilities: q = softmax(z)
    3. Cross-entropy compares with target: loss = H(target, q)
    """)

    logits = np.array([2.0, 1.0, 0.5])
    target = np.array([1, 0, 0])  # true class is index 0

    probs = softmax(logits)
    ce = cross_entropy(target.astype(float), probs)

    print(f"  Logits:     {logits}")
    print(f"  Softmax:    {probs} (sum = {np.sum(probs):.4f})")
    print(f"  Target:     {target}")
    print(f"  Loss:       {ce:.4f}")
    print(f"  -log(p_0):  {-np.log(probs[0]):.4f}")

    print("""
    The loss is exactly -log(softmax(z)[true_class]).
    This is what neural network training minimizes!
    """)


if __name__ == "__main__":
    print("CROSS-ENTROPY AND CLASSIFICATION LOSS")
    print("=" * 55)

    example_1_one_hot_cross_entropy()
    example_2_relationship_to_entropy()
    example_3_softmax_and_cross_entropy()
