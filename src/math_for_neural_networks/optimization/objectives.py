"""
Objective Functions
===================

Common test functions used for evaluating optimization algorithms.

Each function provides:
- The function value f(x) or f(x, y)
- The analytical gradient (for verification)
- ML connection explanation
- Mathematical properties (convexity, minima)

These are NOT neural network losses. They are mathematical functions
chosen to test optimizer behavior in controlled settings.

ML Connection:
Optimizing these functions is mathematically equivalent to optimizing
neural network loss functions. The same gradient descent algorithms
that minimize f(x) = x^2 are used to minimize L(theta) in neural networks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Scalar functions (1D)
# ---------------------------------------------------------------------------


def quadratic(x: NDArray | float) -> float:
    """f(x) = x^2

    Properties:
        - Convex
        - Global minimum at x = 0, f(0) = 0
        - Gradient: f'(x) = 2x
        - Simplest optimization test case

    ML Connection:
        This is the simplest possible optimization problem.
        If your optimizer cannot minimize x^2, it cannot train a neural network.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return float(np.sum(x_arr**2))


def quadratic_gradient(x: NDArray | float) -> NDArray:
    """Gradient of f(x) = x^2: f'(x) = 2x."""
    x_arr = np.asarray(x, dtype=np.float64)
    return (2.0 * x_arr).astype(np.float64)


def quartic(x: NDArray | float) -> float:
    """f(x) = x^4 - 2x^2

    Properties:
        - Non-convex
        - Two global minima at x = +/- 1, f(+/-1) = -1
        - Local maximum at x = 0, f(0) = 0
        - Tests ability to escape saddle points

    ML Connection:
        Non-convex functions demonstrate that optimizer behavior
        depends on initialization and learning rate.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return float(np.sum(x_arr**4 - 2.0 * x_arr**2))


def quartic_gradient(x: NDArray | float) -> NDArray:
    """Gradient of f(x) = x^4 - 2x^2: f'(x) = 4x^3 - 4x."""
    x_arr = np.asarray(x, dtype=np.float64)
    return (4.0 * x_arr**3 - 4.0 * x_arr).astype(np.float64)


def rosenbrock_1d(x: float, a: float = 1.0, b: float = 100.0) -> float:
    """f(x) = (a - x)^2  (1D Rosenbrock-like)

    Properties:
        - Convex
        - Minimum at x = a
        - Parameter b controls curvature (not used here but documented)

    ML Connection:
        Tests convergence speed on functions with different curvatures.
    """
    return (a - x) ** 2


def rosenbrock_1d_gradient(x: float, a: float = 1.0, b: float = 100.0) -> float:
    """Gradient of 1D Rosenbrock-like: f'(x) = -2(a - x)."""
    return -2.0 * (a - x)


# ---------------------------------------------------------------------------
# Multivariate functions (2D)
# ---------------------------------------------------------------------------


def sphere(x: NDArray) -> float:
    """f(x, y) = x^2 + y^2

    Properties:
        - Convex
        - Global minimum at origin (0, 0), f(0,0) = 0
        - Gradient: nabla f = [2x, 2y]
        - Isotropic (same curvature in all directions)

    ML Connection:
        The simplest multivariate optimization test.
        All reasonable optimizers should converge quickly.
    """
    return float(np.sum(x**2))


def sphere_gradient(x: NDArray) -> NDArray:
    """Gradient of f(x,y) = x^2 + y^2: nabla f = [2x, 2y]."""
    return 2.0 * x.copy()


def rosenbrock(x: NDArray) -> float:
    """f(x, y) = (1 - x)^2 + 100(y - x^2)^2

    Properties:
        - Non-convex (but has single global minimum)
        - Global minimum at (1, 1), f(1,1) = 0
        - Narrow curved valley makes optimization challenging
        - Classic optimization test function

    ML Connection:
        The curved valley is analogous to the loss landscapes
        encountered in neural network training, where parameters
        are coupled and curvatures differ across directions.
    """
    x_val = float(x[0])
    y_val = float(x[1])
    return (1.0 - x_val) ** 2 + 100.0 * (y_val - x_val**2) ** 2


def rosenbrock_gradient(x: NDArray) -> NDArray:
    """Gradient of Rosenbrock function.

    nabla f = [-2(1-x) - 400x(y-x^2), 200(y-x^2)]
    """
    x_val = float(x[0])
    y_val = float(x[1])
    dfdx = -2.0 * (1.0 - x_val) - 400.0 * x_val * (y_val - x_val**2)
    dfdy = 200.0 * (y_val - x_val**2)
    return np.array([dfdx, dfdy], dtype=np.float64)


def beale(x: NDArray) -> float:
    """f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Properties:
        - Non-convex
        - Global minimum at (3, 0.5), f(3, 0.5) = 0
        - Multiple local minima
        - Tests ability to navigate complex landscapes

    ML Connection:
        Demonstrates that initialization matters — different starting
        points can lead to different local minima.
    """
    x_val = float(x[0])
    y_val = float(x[1])
    return (
        (1.5 - x_val + x_val * y_val) ** 2
        + (2.25 - x_val + x_val * y_val**2) ** 2
        + (2.625 - x_val + x_val * y_val**3) ** 2
    )


def beale_gradient(x: NDArray) -> NDArray:
    """Numerical gradient of Beale function (analytical is complex)."""
    h = 1e-7
    grad = np.zeros(2, dtype=np.float64)
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (beale(x_plus) - beale(x_minus)) / (2.0 * h)
    return grad


def ackley(x: NDArray) -> float:
    """f(x, y) = -20*exp(-0.2*sqrt(0.5*(x^2+y^2))) - exp(0.5*(cos(2*pi*x)+cos(2*pi*y))) + e + 20

    Properties:
        - Non-convex with many local minima
        - Global minimum at (0, 0), f(0,0) = 0
        - Nearly flat outer region with large number of local minima
        - Tests exploration vs exploitation

    ML Connection:
        The many local minima resemble the loss landscape of
        deep neural networks, where optimizers must navigate
        past poor local minima.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x_arr**2)))
    term2 = -np.exp(0.5 * np.sum(np.cos(2.0 * np.pi * x_arr)))
    return float(term1 + term2 + np.e + 20.0)


def ackley_gradient(x: NDArray) -> NDArray:
    """Numerical gradient of Ackley function."""
    h = 1e-7
    grad = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (ackley(x_plus) - ackley(x_minus)) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Data-fitting objectives (bridge to ML)
# ---------------------------------------------------------------------------


def linear_regression_loss(params: NDArray, X: NDArray, y: NDArray) -> float:
    """MSE loss for linear regression: L(w, b) = (1/n) * sum((y_i - (w*x_i + b))^2)

    Args:
        params: [w, b] — weight and bias.
        X: Input features (1D).
        y: Target values (1D).

    Returns:
        Mean squared error.
    """
    w = float(params[0])
    b = float(params[1])
    predictions = w * X + b
    return float(np.mean((y - predictions) ** 2))


def linear_regression_gradient(params: NDArray, X: NDArray, y: NDArray) -> NDArray:
    """Gradient of MSE loss for linear regression.

    dL/dw = -(2/n) * sum(x_i * (y_i - (w*x_i + b)))
    dL/db = -(2/n) * sum(y_i - (w*x_i + b))
    """
    w = float(params[0])
    b = float(params[1])
    n = len(X)
    residuals = y - (w * X + b)
    dLdw = float(-2.0 * np.sum(X * residuals) / n)
    dLdb = float(-2.0 * np.sum(residuals) / n)
    return np.array([dLdw, dLdb], dtype=np.float64)


def logistic_loss(params: NDArray, X: NDArray, y: NDArray) -> float:
    """Binary cross-entropy loss for logistic regression: L(w, b) = -(1/n) * sum(y*log(sigma) + (1-y)*log(1-sigma))

    where sigma = sigmoid(w*x + b)

    Args:
        params: [w, b] — weight and bias.
        X: Input features (1D).
        y: Binary targets (0 or 1).

    Returns:
        Mean binary cross-entropy loss.
    """
    w = float(params[0])
    b = float(params[1])
    logits = w * X + b
    # Numerically stable sigmoid
    sigmoid_vals = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    eps = 1e-12
    sigmoid_vals = np.clip(sigmoid_vals, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(sigmoid_vals) + (1.0 - y) * np.log(1.0 - sigmoid_vals)))


def logistic_gradient(params: NDArray, X: NDArray, y: NDArray) -> NDArray:
    """Gradient of binary cross-entropy loss for logistic regression.

    dL/dw = (1/n) * sum((sigma - y) * x)
    dL/db = (1/n) * sum(sigma - y)
    """
    w = float(params[0])
    b = float(params[1])
    logits = w * X + b
    sigmoid_vals = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    n = len(X)
    errors = sigmoid_vals - y
    dLdw = float(np.sum(errors * X) / n)
    dLdb = float(np.sum(errors) / n)
    return np.array([dLdw, dLdb], dtype=np.float64)
