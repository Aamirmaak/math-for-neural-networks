"""
Experiment 15: Embedding Geometry
===================================

HYPOTHESIS: Manually created toy vectors can demonstrate geometric properties
of embedding spaces (distance, similarity, clustering).

OBJECTIVE: Visualize vector positions, distances, and cosine similarities.

METHOD: Create toy embedding vectors for 4 categories. Compute pairwise
cosine similarities and Euclidean distances. Visualize in 2D.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.linear_algebra import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    l2_norm,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the embedding geometry experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Embedding Geometry",
        hypothesis="Cosine similarity reveals vector relationships",
        method="Create toy vectors, compute similarities and distances",
    )

    vectors = {
        "A (animal1)": np.array([1.0, 0.5, 0.3, 0.8]),
        "B (animal2)": np.array([0.9, 0.4, 0.2, 0.7]),
        "C (vehicle1)": np.array([0.1, 0.8, 0.9, 0.2]),
        "D (vehicle2)": np.array([0.2, 0.7, 0.8, 0.1]),
    }

    names = list(vectors.keys())
    n = len(names)

    cos_matrix = np.zeros((n, n))
    euc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = cosine_similarity(vectors[names[i]], vectors[names[j]])
            euc_matrix[i, j] = euclidean_distance(vectors[names[i]], vectors[names[j]])

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                [
                    f"{names[i][:8]}-{names[j][:8]}",
                    f"{cos_matrix[i, j]:.4f}",
                    f"{euc_matrix[i, j]:.4f}",
                ]
            )

    table = print_table(
        ["Pair", "Cosine Sim", "Euclidean Dist"],
        rows,
        title="Pairwise Similarities",
    )
    print(table)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(cos_matrix, cmap="YlOrRd", vmin=-1, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels([n[:8] for n in names], rotation=45, ha="right")
    axes[0].set_yticklabels([n[:8] for n in names])
    axes[0].set_title("Cosine Similarity")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(euc_matrix, cmap="YlOrRd")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels([n[:8] for n in names], rotation=45, ha="right")
    axes[1].set_yticklabels([n[:8] for n in names])
    axes[1].set_title("Euclidean Distance")
    fig.colorbar(im2, ax=axes[1])

    fig.suptitle("Experiment 15: Embedding Geometry", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "embedding_geometry", experiment_name="15_embedding_geometry")

    intra_sim = np.mean([cos_matrix[0, 1], cos_matrix[2, 3]])
    inter_sim = np.mean([cos_matrix[0, 2], cos_matrix[0, 3], cos_matrix[1, 2], cos_matrix[1, 3]])
    result.metrics["intra_group_similarity"] = intra_sim
    result.metrics["inter_group_similarity"] = inter_sim
    result.observations.append(f"Within-group similarity: {intra_sim:.4f}")
    result.observations.append(f"Between-group similarity: {inter_sim:.4f}")
    result.conclusion = (
        "Similar vectors (same category) have higher cosine similarity and lower "
        "Euclidean distance. This geometric structure is what embedding spaces capture."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
