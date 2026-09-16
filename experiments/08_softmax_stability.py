"""
Experiment 8: Softmax Stability
================================

HYPOTHESIS: Naive softmax overflows for large logits, while numerically
stable softmax (subtracting max) produces correct probabilities.

OBJECTIVE: Demonstrate numerical stability of softmax implementation.

METHOD: Apply naive and stable softmax to increasingly large logits.
Show overflow behavior and stable output.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.probability.stability import softmax as stable_softmax
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def naive_softmax(x: np.ndarray) -> np.ndarray:
    """Naive softmax without numerical stability."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the softmax stability experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Softmax Stability",
        hypothesis="Naive softmax overflows for large logits",
        method="Compare naive vs stable softmax across logit magnitudes",
    )

    logit_scales = [1.0, 5.0, 10.0, 50.0, 100.0, 200.0]
    base_logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5])

    rows = []
    naive_results = []
    stable_results = []

    for scale in logit_scales:
        logits = base_logits * scale
        naive = naive_softmax(logits)
        stable = stable_softmax(logits)

        naive_valid = np.all(np.isfinite(naive))
        stable_valid = np.all(np.isfinite(stable))
        naive_sum = float(np.sum(naive))
        stable_sum = float(np.sum(stable))

        naive_results.append(naive)
        stable_results.append(stable)

        rows.append(
            [
                f"{scale}",
                f"{'valid' if naive_valid else 'INVALID'}",
                f"{naive_sum:.4f}",
                f"{'valid' if stable_valid else 'INVALID'}",
                f"{stable_sum:.4f}",
            ]
        )

    table = print_table(
        ["Scale", "Naive Status", "Naive Sum", "Stable Status", "Stable Sum"],
        rows,
        title="Softmax Stability Comparison",
    )
    print(table)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_idx = np.arange(len(base_logits))
    width = 0.15

    for i, scale in enumerate(logit_scales[:4]):
        axes[0].bar(x_idx + i * width, naive_results[i], width, label=f"scale={scale}", alpha=0.8)
    axes[0].set_xlabel("Class Index")
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Naive Softmax")
    axes[0].set_xticks(x_idx + width * 1.5)
    axes[0].set_xticklabels([f"c{i}" for i in range(len(base_logits))])
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, scale in enumerate(logit_scales[:4]):
        axes[1].bar(x_idx + i * width, stable_results[i], width, label=f"scale={scale}", alpha=0.8)
    axes[1].set_xlabel("Class Index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Stable Softmax (subtract max)")
    axes[1].set_xticks(x_idx + width * 1.5)
    axes[1].set_xticklabels([f"c{i}" for i in range(len(base_logits))])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Experiment 8: Softmax Numerical Stability", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "softmax_stability", experiment_name="08_softmax_stability")

    result.metrics["max_naive_valid_scale"] = max(
        s for s, r in zip(logit_scales, naive_results) if np.all(np.isfinite(r))
    )
    result.metrics["stable_always_valid"] = all(np.all(np.isfinite(r)) for r in stable_results)
    result.observations.append(
        "Subtracting max(logits) before exp() prevents overflow while "
        "preserving probabilities (softmax is shift-invariant)"
    )
    result.conclusion = (
        "Numerically stable softmax subtracts max(logits) to prevent overflow. "
        "This preserves softmax output because exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x)). "
        "All deep learning frameworks use this trick."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
