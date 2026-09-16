"""
Classification Forward Pass
===========================

Demonstrates the complete forward mathematical chain for classification:

    Input -> Affine -> Logits -> Softmax -> Probabilities -> Cross-Entropy Loss

This is a FORWARD-ONLY demonstration. No training is performed.
Every intermediate value is shown.
"""

import numpy as np

from math_for_neural_networks.neural_networks.activations import sigmoid
from math_for_neural_networks.neural_networks.attention import softmax
from math_for_neural_networks.neural_networks.layers import affine_transform
from math_for_neural_networks.neural_networks.losses import (
    categorical_cross_entropy,
    cross_entropy_with_logits,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def binary_classification() -> None:
    """Binary classification example."""
    print_section("Binary Classification")

    print("""
    Input -> Affine -> Sigmoid -> Probability -> Binary Cross-Entropy

    Example: Predicting if an email is spam (1) or not spam (0)
    """)

    x = np.array([0.5, -1.0, 2.0])
    W = np.array([[-0.3, 0.5, 0.8]])
    b = np.array([0.1])

    z = affine_transform(x, W, b)
    p_val = float(sigmoid(z[0]))
    target = np.array([1.0])
    loss = -(target * np.log(p_val) + (1 - target) * np.log(1 - p_val))

    print(f"  Input:       {x}")
    print(f"  Weights:     {W.flatten()}")
    print(f"  Bias:        {b[0]}")
    print(f"  Logit:       z = {z[0]:.4f}")
    print(f"  Probability: p = {p_val:.4f}")
    print(f"  Target:      {target[0]}")
    print(f"  Loss:        {loss[0]:.4f}")
    print(f"\n  Interpretation: Model predicts {p_val*100:.1f}% chance of spam")


def multi_class_classification() -> None:
    """Multi-class classification example."""
    print_section("Multi-Class Classification")

    print("""
    Input -> Affine -> Logits -> Softmax -> Probabilities -> Cross-Entropy

    Example: Classifying an image as cat, dog, or bird
    """)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    W = np.array([[0.1, -0.2, 0.3, 0.4],
                   [0.5, -0.6, 0.7, -0.8],
                   [-0.1, 0.2, -0.3, 0.4]])
    b = np.array([0.0, 0.0, 0.0])
    class_names = ["cat", "dog", "bird"]

    logits = affine_transform(x, W, b)
    probs = softmax(logits)
    target_idx = 0  # cat
    target = np.array([1.0, 0.0, 0.0])
    loss = categorical_cross_entropy(target, probs)

    print(f"  Input:  {x}")
    print(f"  Logits: {logits}")
    print(f"  Softmax probabilities:")
    for name, prob in zip(class_names, probs):
        print(f"    {name}: {prob:.4f}")
    print(f"  Sum of probabilities: {np.sum(probs):.6f}")
    print(f"  True class: {class_names[target_idx]}")
    print(f"  Loss: {loss:.4f}")
    print(f"\n  Interpretation: Model is {probs[target_idx]*100:.1f}% confident it's a {class_names[target_idx]}")


def logits_vs_probabilities() -> None:
    """Explain logits vs probabilities."""
    print_section("Logits vs Probabilities")

    print("""
    Logits: raw model outputs (unbounded real numbers)
    Probabilities: bounded [0,1], sum to 1

    Why operate on logits?
    - Softmax is numerically unstable for extreme values
    - Loss functions can be computed directly from logits
    - This is more stable than: softmax -> log -> loss
    """)

    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)

    print(f"  Logits:        {logits}")
    print(f"  Softmax:       {probs}")
    print(f"  Sum:           {np.sum(probs):.6f}")

    loss_softmax = categorical_cross_entropy(probs, probs)
    loss_logits = cross_entropy_with_logits(logits, np.array([0.0, 0.0, 1.0]))
    print(f"\n  Loss (from probs): {loss_softmax:.6f}")
    print(f"  Loss (from logits): {loss_logits:.6f}")
    print(f"  (Logits version is more numerically stable)")


if __name__ == "__main__":
    print("CLASSIFICATION FORWARD PASS")
    print("=" * 60)

    binary_classification()
    multi_class_classification()
    logits_vs_probabilities()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
    1. Binary: Input -> Affine -> Sigmoid -> BCE loss
    2. Multi-class: Input -> Affine -> Softmax -> CE loss
    3. Logits are raw outputs; probabilities are normalized
    4. Compute loss directly from logits for stability
    5. This is the FORWARD pass; training requires BACKWARD pass
    """)
