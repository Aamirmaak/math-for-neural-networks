"""
Experiment 10: Backpropagation
===============================

HYPOTHESIS: Backpropagation correctly computes gradients by applying the
chain rule in reverse through a computational graph.

OBJECTIVE: Demonstrate forward and backward passes through a small graph.

METHOD: Build a small computational graph. Compute forward values and
backward gradients. Display node values and gradients.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.autograd.value import Value, get_topo_order
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the backpropagation experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Backpropagation",
        hypothesis="Chain rule in reverse computes correct gradients",
        method="Build computational graph, forward + backward, display values and gradients",
    )

    # Graph: z = w*x + b, loss = (z - target)^2
    w = Value(2.0, _op="w")
    x = Value(3.0, _op="x")
    b = Value(1.0, _op="b")
    target = Value(10.0, _op="target")

    z = w * x + b
    diff = z - target
    loss = diff**2

    topo = get_topo_order(loss)

    print("\n  Forward Pass:")
    print(f"  {'Node':>8} | {'Value':>10} | {'Op':>8}")
    print("  " + "-" * 32)
    for node in topo:
        print(f"  {str(node):>8} | {node.data:>10.4f} | {node._op:>8}")

    loss.backward()

    print("\n  Backward Pass (Gradients):")
    print(f"  {'Node':>8} | {'Value':>10} | {'Gradient':>10}")
    print("  " + "-" * 32)
    for node in topo:
        print(f"  {str(node):>8} | {node.data:>10.4f} | {node.grad:>10.4f}")

    # Manual verification
    w_val, x_val, b_val, target_val = 2.0, 3.0, 1.0, 10.0
    z_val = w_val * x_val + b_val
    loss_val = (z_val - target_val) ** 2
    dloss_dz = 2 * (z_val - target_val)
    dz_dw = x_val
    dz_dx = w_val
    dz_db = 1.0
    dloss_dw = dloss_dz * dz_dw
    dloss_dx = dloss_dz * dz_dx
    dloss_db = dloss_dz * dz_db

    rows = [
        ["w", f"{w_val:.4f}", f"{w.grad:.4f}", f"{dloss_dw:.4f}"],
        ["x", f"{x_val:.4f}", f"{x.grad:.4f}", f"{dloss_dx:.4f}"],
        ["b", f"{b_val:.4f}", f"{b.grad:.4f}", f"{dloss_db:.4f}"],
    ]
    table = print_table(
        ["Parameter", "Value", "Autograd Grad", "Manual Grad"],
        rows,
        title="Gradient Verification",
    )
    print(table)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    nodes = [
        ("w=2.0", 0.1, 0.7, "lightblue"),
        ("x=3.0", 0.1, 0.3, "lightgreen"),
        ("b=1.0", 0.4, 0.7, "lightyellow"),
        ("z=7.0", 0.4, 0.5, "lightyellow"),
        ("target=10.0", 0.7, 0.7, "lightyellow"),
        ("diff=-3.0", 0.7, 0.5, "lightyellow"),
        ("loss=9.0", 0.7, 0.3, "lightsalmon"),
    ]
    for label, x_pos, y_pos, color in nodes:
        ax.add_patch(
            plt.Rectangle(
                (x_pos - 0.08, y_pos - 0.08),
                0.2,
                0.15,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
        )
        ax.text(x_pos + 0.02, y_pos, label, ha="center", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Experiment 10: Computational Graph (Forward + Backward)", fontsize=14)

    save_fig(fig, "backpropagation", experiment_name="10_backpropagation")

    result.metrics["autograd_grad_w"] = float(w.grad)
    result.metrics["manual_grad_w"] = dloss_dw
    result.metrics["autograd_grad_x"] = float(x.grad)
    result.metrics["manual_grad_x"] = dloss_dx
    result.metrics["gradient_match"] = all(
        [
            abs(w.grad - dloss_dw) < 1e-10,
            abs(x.grad - dloss_dx) < 1e-10,
            abs(b.grad - dloss_db) < 1e-10,
        ]
    )
    result.observations.append("Forward: z = w*x + b = 2*3+1 = 7")
    result.observations.append("Loss = (z-target)^2 = (7-10)^2 = 9")
    result.observations.append("dL/dw = 2*(z-target)*x = 2*(-3)*3 = -18")
    result.observations.append("dL/dx = 2*(z-target)*w = 2*(-3)*2 = -12")
    result.observations.append("dL/db = 2*(z-target)*1 = -6")
    result.conclusion = (
        "Backpropagation correctly computes all parameter gradients by applying "
        "the chain rule in reverse through the computational graph. "
        "The autograd engine produces gradients matching manual computation."
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
