"""
Experiment 1: Vector Geometry
=============================

HYPOTHESIS: Cosine similarity depends on angle between vectors, independent
of magnitude. Euclidean distance depends on both angle and magnitude.

OBJECTIVE: Demonstrate relationships between vector angle, dot product,
cosine similarity, and Euclidean distance.

METHOD: Define vector pairs with controlled angles and magnitudes.
Compute all four geometric quantities. Visualize.

CONNECTION: Embedding similarity in neural networks uses cosine similarity
because it is magnitude-invariant.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.linear_algebra import (
    dot_product,
    l2_norm,
    cosine_similarity,
    euclidean_distance,
)
from experiment_utils import ExperimentResult, save_fig, print_table, setup_seed


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the vector geometry experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Vector Geometry",
        hypothesis="Cosine similarity depends on angle, not magnitude",
        method="Define vectors with controlled angles and magnitudes",
    )

    angles_deg = [0, 30, 45, 60, 90, 120, 150, 180]
    v1 = np.array([1.0, 0.0])

    rows = []
    cos_sims = []
    euc_dists = []
    dot_vals = []

    for angle in angles_deg:
        rad = np.radians(angle)
        v2 = np.array([np.cos(rad), np.sin(rad)])
        dp = dot_product(v1, v2)
        cs = cosine_similarity(v1, v2)
        ed = euclidean_distance(v1, v2)
        rows.append([f"{angle}", f"{dp:.4f}", f"{cs:.4f}", f"{ed:.4f}"])
        cos_sims.append(cs)
        euc_dists.append(ed)
        dot_vals.append(dp)

    table = print_table(
        ["Angle", "Dot Product", "Cosine Sim", "Euclidean Dist"],
        rows,
        title="Vector Geometric Relationships",
    )
    print(table)

    # Verify magnitude independence
    v2_large = np.array([3.0, 0.0])
    cs_original = cosine_similarity(v1, np.array([1.0, 0.0]))
    cs_scaled = cosine_similarity(v1, v2_large)
    result.observations.append(
        f"Cosine similarity is magnitude-invariant: cos([1,0], [1,0])={cs_original:.4f}, "
        f"cos([1,0], [3,0])={cs_scaled:.4f}"
    )

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(angles_deg, cos_sims, "o-", label="Cosine Similarity")
    axes[0].set_xlabel("Angle (degrees)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Cosine Similarity vs Angle")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(angles_deg, euc_dists, "s-", color="orange", label="Euclidean Distance")
    axes[1].set_xlabel("Angle (degrees)")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].set_title("Euclidean Distance vs Angle")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(angles_deg, dot_vals, "^-", color="green", label="Dot Product")
    axes[2].set_xlabel("Angle (degrees)")
    axes[2].set_ylabel("Dot Product")
    axes[2].set_title("Dot Product vs Angle")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Experiment 1: Vector Geometric Relationships", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "vector_geometry", experiment_name="01_vector_geometry")

    result.metrics["angle_90_cosine"] = cos_sims[4]
    result.metrics["angle_0_euclidean"] = euc_dists[0]
    result.metrics["magnitude_invariance_verified"] = abs(cs_original - cs_scaled) < 1e-10
    result.observations.append(
        "Cosine similarity = cos(angle) for unit vectors, independent of magnitude"
    )
    result.observations.append("Euclidean distance grows with both angle and magnitude differences")
    result.conclusion = (
        "Cosine similarity measures direction alignment independent of magnitude. "
        "Euclidean distance measures overall proximity including magnitude. "
        "This is why embeddings use cosine similarity for semantic similarity."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
