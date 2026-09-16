"""
Experiment 16: Attention
=========================

HYPOTHESIS: Scaled dot-product attention computes weighted averages of values
based on query-key similarity, producing context-aware representations.

OBJECTIVE: Demonstrate attention computation step by step with visualization.

METHOD: Create small Q, K, V matrices. Compute QK^T, scaled scores,
softmax, attention weights, and output. Visualize weights as heatmap.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.attention import (
    softmax,
    scaled_dot_product_attention,
    attention_weights,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the attention experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Attention",
        hypothesis="Attention computes query-key similarity weighted average",
        method="Manual Q, K, V computation with step-by-step visualization",
    )

    d_k = 4
    seq_len = 5

    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_k))

    # Step 1: QK^T
    scores = Q @ K.T
    print(f"\n  QK^T (raw scores):\n{np.array2string(scores, precision=3)}")

    # Step 2: Scaled
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\n  Scaled scores (QK^T/sqrt(d_k)):\n{np.array2string(scaled_scores, precision=3)}")

    # Step 3: Softmax
    weights = softmax(scaled_scores)
    print(f"\n  Attention weights:\n{np.array2string(weights, precision=3)}")

    # Step 4: Weighted sum
    output = weights @ V
    print(f"\n  Output (weights @ V):\n{np.array2string(output, precision=3)}")

    # Verify via library function
    lib_output, lib_weights = scaled_dot_product_attention(Q, K, V)
    result.metrics["output_match"] = bool(np.allclose(output, lib_output, atol=1e-6))
    result.metrics["weights_match"] = bool(np.allclose(weights, lib_weights, atol=1e-6))

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(scores, cmap="RdBu_r")
    axes[0].set_title("QK^T (Raw Scores)")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(scaled_scores, cmap="RdBu_r")
    axes[1].set_title(f"QK^T/sqrt({d_k}) (Scaled)")
    axes[1].set_xlabel("Key position")
    axes[1].set_ylabel("Query position")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(weights, cmap="YlOrRd")
    axes[2].set_title("Attention Weights (softmax)")
    axes[2].set_xlabel("Key position")
    axes[2].set_ylabel("Query position")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("Experiment 16: Scaled Dot-Product Attention", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "attention", experiment_name="16_attention")

    result.observations.append("Attention weights sum to 1 for each query (row)")
    result.observations.append("Scaling by 1/sqrt(d_k) prevents softmax saturation")
    result.conclusion = (
        "Attention computes weighted average of values based on query-key similarity. "
        "This is the core operation in Transformer architectures."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
