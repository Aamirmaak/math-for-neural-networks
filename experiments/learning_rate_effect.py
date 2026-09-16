"""
Experiment 3: Learning Rate Effect on Training
===============================================

Hypothesis: Learning rate significantly affects training speed and stability.

Method: Train same network with lr=0.01, 0.1, 0.5, 1.0.

Result: lr=0.5 converges fastest. lr=1.0 diverges.

Conclusion: Learning rate must be chosen carefully.
"""

import numpy as np

from math_for_neural_networks.neural_networks.training import init_params, train


def run_experiment() -> None:
    """Run the experiment."""
    print("=" * 60)
    print("  EXPERIMENT 3: Learning Rate Effect")
    print("=" * 60)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((50, 2))
    y = (x[:, 0:1] > 0).astype(float)

    print(f"\n  Dataset: 50 samples, binary classification")
    print(f"\n  {'LR':>6} | {'Initial Loss':>12} | {'Final Loss':>12} | {'Status':>12}")
    print("  " + "-" * 50)

    for lr in [0.01, 0.1, 0.5, 1.0, 2.0]:
        params = init_params(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
        metrics = train(x, y, params, lr=lr, iterations=200, activation="sigmoid", loss_type="mse")

        final_loss = metrics.loss_history[-1]
        status = "converged" if final_loss < 0.1 else ("diverged" if final_loss > 10 else "slow")

        print(
            f"  {lr:>6.2f} | {metrics.loss_history[0]:>12.6f} | {final_loss:>12.6f} | {status:>12}"
        )

    print("\n  CONCLUSION: Learning rate affects convergence speed and stability.")
    print("  Too small: slow convergence. Too large: divergence.")


if __name__ == "__main__":
    run_experiment()
