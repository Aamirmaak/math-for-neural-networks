"""
Experiment 11: Gradient Checking
==================================

HYPOTHESIS: Analytical gradients from backpropagation match numerical
gradients computed via finite differences.

OBJECTIVE: Verify all Stage 6 neural network gradient implementations.

METHOD: Use gradient_check for each component: affine, sigmoid, relu,
tanh, mse, softmax+CE, sigmoid+BCE.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_for_neural_networks.neural_networks.backprop import (
    gradient_check,
    affine_backward,
    sigmoid_backward,
    relu_backward,
    tanh_backward,
    mse_backward,
    softmax_backward,
    sigmoid_bce_backward,
)
from experiment_utils import ExperimentResult, save_fig, setup_seed, print_table


def run_experiment(seed: int = 42) -> ExperimentResult:
    """Run the gradient checking experiment."""
    rng = setup_seed(seed)
    result = ExperimentResult(
        name="Gradient Checking",
        hypothesis="Analytical gradients match numerical gradients",
        method="Compare backprop gradients vs finite-difference gradients",
    )

    checks = []

    # 1. Affine backward
    x = rng.standard_normal((5, 3))
    W = rng.standard_normal((4, 3))
    b = rng.standard_normal(4)
    dout = rng.standard_normal((5, 4))

    def affine_func(params_flat):
        W_r = params_flat[:12].reshape(4, 3)
        b_r = params_flat[12:]
        z = x @ W_r.T + b_r
        return float(np.sum(z * dout))

    def affine_grad(params_flat):
        W_r = params_flat[:12].reshape(4, 3)
        b_r = params_flat[12:]
        _, dW, db = affine_backward(x, W_r, b_r, dout)
        return np.concatenate([dW.ravel(), db])

    params_flat = np.concatenate([W.ravel(), b])
    res = gradient_check(affine_func, affine_grad, params_flat)
    checks.append(("Affine", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 2. Sigmoid backward
    x_sig = rng.standard_normal(5)
    dout_sig = rng.standard_normal(5)

    def sig_func(x_val):
        a = 1.0 / (1.0 + np.exp(-x_val))
        return float(np.sum(a * dout_sig))

    def sig_grad(x_val):
        return sigmoid_backward(x_val, dout_sig)

    res = gradient_check(sig_func, sig_grad, x_sig)
    checks.append(("Sigmoid", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 3. ReLU backward
    def relu_func(x_val):
        a = np.maximum(0, x_val)
        return float(np.sum(a * dout_sig))

    def relu_grad(x_val):
        return relu_backward(x_val, dout_sig)

    res = gradient_check(relu_func, relu_grad, x_sig)
    checks.append(("ReLU", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 4. Tanh backward
    def tanh_func(x_val):
        a = np.tanh(x_val)
        return float(np.sum(a * dout_sig))

    def tanh_grad(x_val):
        return tanh_backward(x_val, dout_sig)

    res = gradient_check(tanh_func, tanh_grad, x_sig)
    checks.append(("Tanh", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 5. MSE backward
    y_true = rng.standard_normal(5)
    y_pred = rng.standard_normal(5)

    def mse_func(pred_flat):
        return float(np.sum((pred_flat - y_true) ** 2) / y_true.size)

    def mse_grad(pred_flat):
        return mse_backward(y_true, pred_flat)

    res = gradient_check(mse_func, mse_grad, y_pred)
    checks.append(("MSE", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 6. Softmax + CE backward
    logits = rng.standard_normal((3,))
    targets = np.array([1.0, 0.0, 0.0])

    def softmax_ce_func(logits_flat):
        shifted = logits_flat - np.max(logits_flat)
        exp_l = np.exp(shifted)
        probs = exp_l / np.sum(exp_l)
        eps = 1e-15
        return float(-np.sum(targets * np.log(np.clip(probs, eps, 1))))

    def softmax_ce_grad(logits_flat):
        return softmax_backward(logits_flat, targets)

    res = gradient_check(softmax_ce_func, softmax_ce_grad, logits)
    checks.append(("Softmax+CE", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    # 7. Sigmoid + BCE backward
    logits_bce = rng.standard_normal(5)
    targets_bce = rng.choice([0.0, 1.0], size=5)

    def sigmoid_bce_func(z):
        a = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-15
        n = targets_bce.size
        return float(
            -np.mean(
                targets_bce * np.log(np.clip(a, eps, 1))
                + (1 - targets_bce) * np.log(np.clip(1 - a, eps, 1))
            )
        )

    def sigmoid_bce_grad(z):
        return sigmoid_bce_backward(z, targets_bce)

    res = gradient_check(sigmoid_bce_func, sigmoid_bce_grad, logits_bce)
    checks.append(("Sigmoid+BCE", res["passed"], res["max_abs_error"], res["max_rel_error"]))

    rows = [
        [name, "PASS" if passed else "FAIL", f"{abs_err:.4e}", f"{rel_err:.4e}"]
        for name, passed, abs_err, rel_err in checks
    ]
    table = print_table(
        ["Component", "Status", "Max Abs Error", "Max Rel Error"],
        rows,
        title="Gradient Checking Results",
    )
    print(table)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c[0] for c in checks]
    abs_errors = [c[2] for c in checks]
    colors = ["green" if c[1] else "red" for c in checks]
    ax.bar(names, abs_errors, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Max Absolute Error")
    ax.set_title("Experiment 11: Gradient Checking - Analytical vs Numerical")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_fig(fig, "gradient_checking", experiment_name="11_gradient_checking")

    all_passed = all(c[1] for c in checks)
    result.metrics["all_checks_passed"] = all_passed
    result.metrics["num_checks"] = len(checks)
    result.metrics["num_passed"] = sum(1 for c in checks if c[1])
    result.observations.append(
        f"Gradient checks passed: {sum(1 for c in checks if c[1])}/{len(checks)}"
    )
    result.conclusion = (
        (
            "All gradient implementations match numerical approximations. "
            "This validates the correctness of backpropagation through each component."
        )
        if all_passed
        else ("Some gradient checks failed. Review implementations.")
    )
    return result


if __name__ == "__main__":
    result = run_experiment()
    print("\n" + result.summary())
