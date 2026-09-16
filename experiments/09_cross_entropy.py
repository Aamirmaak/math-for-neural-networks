"""
Experiment 9: Cross-Entropy
============================

HYPOTHESIS: Cross-entropy loss increases as the probability assigned to the
correct class decreases, providing strong gradients for incorrect predictions.

OBJECTIVE: Visualize cross-entropy as a function of predicted probability.

METHOD: For binary classification, plot CE loss vs p(y=1) for true label y=1.
Show the penalty for confident incorrect predictions.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_utils import ExperimentResult, save_fig, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the cross-entropy experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Cross-Entropy",
        hypothesis="CE penalizes confident incorrect predictions heavily",
        method="Plot CE loss vs predicted probability for fixed true label",
    )

    eps = 1e-15
    p = np.linspace(eps, 1 - eps, 200)

    # Binary CE for y=1: L = -log(p)
    ce_y1 = -np.log(p)
    # Binary CE for y=0: L = -log(1-p)
    ce_y0 = -np.log(1 - p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(p, ce_y1, linewidth=2, label="L = -log(p), y=1")
    axes[0].set_xlabel("Predicted probability p(y=1)")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Cross-Entropy when True Label = 1")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 10)

    axes[1].plot(p, ce_y0, linewidth=2, color="orange", label="L = -log(1-p), y=0")
    axes[1].set_xlabel("Predicted probability p(y=1)")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Cross-Entropy when True Label = 0")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, 10)

    fig.suptitle("Experiment 9: Cross-Entropy Loss", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "cross_entropy", experiment_name="09_cross_entropy")

    # Key values
    p_correct = 0.99
    p_wrong = 0.01
    ce_correct = -np.log(p_correct)
    ce_wrong = -np.log(p_wrong)

    result.metrics["ce_p_0.99"] = float(ce_correct)
    result.metrics["ce_p_0.50"] = float(-np.log(0.5))
    result.metrics["ce_p_0.01"] = float(ce_wrong)
    result.metrics["ce_p_0.001"] = float(-np.log(0.001))
    result.observations.append(f"CE when p(correct)=0.99: {ce_correct:.4f}")
    result.observations.append(f"CE when p(correct)=0.50: {-np.log(0.5):.4f}")
    result.observations.append(f"CE when p(correct)=0.01: {ce_wrong:.4f}")
    result.observations.append(f"CE when p(correct)=0.001: {-np.log(0.001):.4f}")
    result.observations.append("Loss grows logarithmically as predicted probability decreases")
    result.conclusion = (
        "Cross-entropy loss is small when the model assigns high probability to "
        "the correct class, and grows large when the model is confident but wrong. "
        "The gradient dL/dp = -1/p provides strong learning signal for wrong predictions."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
