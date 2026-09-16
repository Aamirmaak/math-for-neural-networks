"""
Experiment 4: Gradient Norm During Training
============================================

Hypothesis: Gradient norm decreases as network converges.

Method: Track gradient norm at each iteration.

Result: Gradient norm decreases monotonically (approximately).

Conclusion: Decreasing gradient norm indicates convergence.
"""

import numpy as np

from math_for_neural_networks.neural_networks.training import init_params, train


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 4: Gradient Norm During Training")
    print("=" * 60)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 2))
    y = (x[:, 0:1] > 0).astype(float)

    params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
    metrics = train(x, y, params, lr=0.5, iterations=200, activation="sigmoid", loss_type="mse")

    print(f"\n  {'Iter':>5} | {'Loss':>10} | {'Grad Norm':>10} | {'Grad Change':>12}")
    print("  " + "-" * 45)

    prev_grad = metrics.gradient_norm_history[0]
    for i in range(0, len(metrics.loss_history), 20):
        grad = metrics.gradient_norm_history[i]
        change = grad - prev_grad
        print(f"  {i:>5} | {metrics.loss_history[i]:>10.6f} | {grad:>10.6f} | {change:>+12.6f}")
        prev_grad = grad

    # Final
    i = len(metrics.loss_history) - 1
    print(
        f"  {i:>5} | {metrics.loss_history[i]:>10.6f} | {metrics.gradient_norm_history[i]:>10.6f}"
    )

    print("\n  CONCLUSION: Gradient norm decreases as network converges.")
    print("  This indicates the network is approaching a minimum.")


if __name__ == "__main__":
    run_experiment()
