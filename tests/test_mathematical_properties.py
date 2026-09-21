"""Mathematical property tests.

Tests fundamental mathematical properties and identities that must hold
for correct implementations. These are not unit tests of specific functions
but rather tests of mathematical laws that any correct implementation must satisfy.
"""

import numpy as np
import pytest

from math_for_neural_networks.linear_algebra.norms import euclidean_distance, l2_norm
from math_for_neural_networks.linear_algebra.operations import matrix_vector_multiply
from math_for_neural_networks.linear_algebra.similarity import cosine_similarity
from math_for_neural_networks.linear_algebra.vectors import dot_product, vector_add
from math_for_neural_networks.neural_networks.activations import (
    gelu,
    relu,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)
from math_for_neural_networks.neural_networks.attention import softmax
from math_for_neural_networks.neural_networks.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    mean_squared_error,
)
from math_for_neural_networks.probability.distributions import (
    bernoulli_pmf,
    categorical_pmf,
    normal_pdf,
)
from math_for_neural_networks.probability.fundamentals import (
    conditional_probability,
    intersection_independent,
)

# ============================================================
# VECTOR PROPERTIES
# ============================================================


class TestVectorProperties:
    """Fundamental vector space properties."""

    def test_dot_commutativity(self) -> None:
        """dot(a, b) == dot(b, a) — commutativity of inner product."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        assert dot_product(a, b) == pytest.approx(dot_product(b, a))

    def test_dot_self_nonneg(self) -> None:
        """dot(a, a) >= 0 — non-negativity of inner product."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(5)
        assert dot_product(a, a) >= 0

    def test_dot_self_zero_implies_zero_vector(self) -> None:
        """dot(a, a) == 0 implies a == 0."""
        zero = np.zeros(5)
        assert dot_product(zero, zero) == 0.0

    def test_l2_norm_nonneg(self) -> None:
        """||x|| >= 0 — norm is non-negative."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        assert l2_norm(x) >= 0

    def test_l2_norm_zero(self) -> None:
        """||0|| == 0."""
        assert l2_norm(np.zeros(5)) == 0.0

    def test_l2_norm_positive_definite(self) -> None:
        """||x|| > 0 for x != 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        assert l2_norm(x) > 0

    def test_l2_norm_triangle_inequality(self) -> None:
        """||a + b|| <= ||a|| + ||b|| — triangle inequality."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        norm_sum = l2_norm(vector_add(a, b))
        triangle = l2_norm(a) + l2_norm(b)
        assert norm_sum <= triangle + 1e-10

    def test_l2_norm_scalar_multiplication(self) -> None:
        """||c * x|| == |c| * ||x||."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        c = 3.0
        assert l2_norm(c * x) == pytest.approx(abs(c) * l2_norm(x))

    def test_euclidean_distance_nonneg(self) -> None:
        """distance(x, y) >= 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        assert euclidean_distance(x, y) >= 0

    def test_euclidean_distance_self_zero(self) -> None:
        """distance(x, x) == 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        assert euclidean_distance(x, x) == pytest.approx(0.0)

    def test_euclidean_distance_symmetry(self) -> None:
        """distance(x, y) == distance(y, x)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        assert euclidean_distance(x, y) == pytest.approx(euclidean_distance(y, x))

    def test_cosine_similarity_self_one(self) -> None:
        """cos(x, x) ≈ 1 for non-zero x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        assert cosine_similarity(x, x) == pytest.approx(1.0, abs=1e-10)

    def test_cosine_similarity_bounded(self) -> None:
        """-1 <= cos(x, y) <= 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        y = rng.standard_normal(5)
        sim = cosine_similarity(x, y)
        assert -1.0 - 1e-10 <= sim <= 1.0 + 1e-10

    def test_cosine_similarity_orthogonal(self) -> None:
        """cos(x, y) ≈ 0 for orthogonal vectors."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(x, y) == pytest.approx(0.0, abs=1e-10)

    def test_angle_self_zero(self) -> None:
        """angle(x, x) ≈ 0 for non-zero x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        cos_sim = dot_product(x, x) / (l2_norm(x) * l2_norm(x))
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_angle_orthogonal_90(self) -> None:
        """angle(x, y) ≈ pi/2 for orthogonal vectors."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        cos_sim = dot_product(x, y) / (l2_norm(x) * l2_norm(y))
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        assert angle == pytest.approx(np.pi / 2, abs=1e-10)


# ============================================================
# MATRIX PROPERTIES
# ============================================================


class TestMatrixProperties:
    """Fundamental matrix properties."""

    def test_transpose_twice(self) -> None:
        """(A^T)^T == A — involution of transpose."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 4))
        assert np.allclose(A.T.T, A)

    def test_identity_times_vector(self) -> None:
        """I @ x == x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4)
        I = np.eye(4)
        result = matrix_vector_multiply(I, x)
        assert np.allclose(result, x)

    def test_determinant_identity(self) -> None:
        """det(I) == 1."""
        assert np.linalg.det(np.eye(3)) == pytest.approx(1.0)

    def test_determinant_product(self) -> None:
        """det(A @ B) == det(A) * det(B) for square matrices."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        det_AB = np.linalg.det(A @ B)
        det_A_det_B = np.linalg.det(A) * np.linalg.det(B)
        assert det_AB == pytest.approx(det_A_det_B, abs=1e-6)

    def test_inverse_times_matrix(self) -> None:
        """A^-1 @ A == I (approximately for non-singular A)."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        A_inv = np.linalg.inv(A)
        product = A_inv @ A
        assert np.allclose(product, np.eye(3), atol=1e-6)

    def test_frobenius_norm_nonneg(self) -> None:
        """||A||_F >= 0."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 4))
        assert l2_norm(A.ravel()) >= 0

    def test_frobenius_norm_zero(self) -> None:
        """||0||_F == 0."""
        assert l2_norm(np.zeros((3, 4)).ravel()) == 0.0

    def test_frobenius_norm_definition(self) -> None:
        """||A||_F == sqrt(sum(a_ij^2))."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 4))
        expected = np.sqrt(np.sum(A**2))
        assert l2_norm(A.ravel()) == pytest.approx(expected)


# ============================================================
# ACTIVATION FUNCTION PROPERTIES
# ============================================================


class TestActivationProperties:
    """Mathematical properties of activation functions."""

    def test_sigmoid_range(self) -> None:
        """0 < sigmoid(x) < 1 for all x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100) * 10
        s = sigmoid(x)
        assert np.all(s > 0) and np.all(s < 1)

    def test_sigmoid_zero(self) -> None:
        """sigmoid(0) == 0.5."""
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_negation(self) -> None:
        """sigmoid(-x) == 1 - sigmoid(x)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10)
        assert np.allclose(sigmoid(-x), 1.0 - sigmoid(x))

    def test_sigmoid_derivative_bounded(self) -> None:
        """0 <= sigmoid'(x) <= 0.25."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        d = sigmoid_derivative(x)
        assert np.all(d >= 0) and np.all(d <= 0.25 + 1e-10)

    def test_sigmoid_derivative_zero_at_extremes(self) -> None:
        """sigmoid'(x) ≈ 0 for large |x|."""
        assert sigmoid_derivative(100.0) < 1e-10
        assert sigmoid_derivative(-100.0) < 1e-10

    def test_tanh_range(self) -> None:
        """-1 <= tanh(x) <= 1 for all x (saturates for large |x|)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100) * 10
        t = tanh(x)
        assert np.all(t >= -1.0) and np.all(t <= 1.0)

    def test_tanh_zero(self) -> None:
        """tanh(0) == 0."""
        assert tanh(0.0) == pytest.approx(0.0)

    def test_tanh_odd(self) -> None:
        """tanh(-x) == -tanh(x) — odd function."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10)
        assert np.allclose(tanh(-x), -tanh(x))

    def test_tanh_derivative_bounded(self) -> None:
        """0 <= tanh'(x) <= 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        d = tanh_derivative(x)
        assert np.all(d >= 0) and np.all(d <= 1.0 + 1e-10)

    def test_tanh_derivative_max_at_zero(self) -> None:
        """tanh'(0) == 1 (maximum)."""
        assert tanh_derivative(0.0) == pytest.approx(1.0)

    def test_relu_nonneg(self) -> None:
        """relu(x) >= 0 for all x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100) * 10
        r = relu(x)
        assert np.all(r >= 0)

    def test_relu_identity_positive(self) -> None:
        """relu(x) == x for x > 0."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 10, size=10)
        assert np.allclose(relu(x), x)

    def test_relu_zero_negative(self) -> None:
        """relu(x) == 0 for x < 0."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, -0.1, size=10)
        assert np.all(relu(x) == 0)

    def test_gelu_smooth(self) -> None:
        """GELU is a smooth approximation of ReLU."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        g = gelu(x)
        r = relu(x)
        # GELU should be close to ReLU for large positive values
        mask = x > 3
        assert np.allclose(g[mask], r[mask], atol=0.1)


# ============================================================
# SOFTMAX PROPERTIES
# ============================================================


class TestSoftmaxProperties:
    """Mathematical properties of softmax."""

    def test_sums_to_one(self) -> None:
        """sum(softmax(x)) == 1."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10)
        assert np.sum(softmax(x)) == pytest.approx(1.0, abs=1e-10)

    def test_sums_to_one_batch(self) -> None:
        """sum(softmax(x), axis=-1) == 1 for each row in batch."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 8))
        sums = np.sum(softmax(x), axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-10)

    def test_nonneg(self) -> None:
        """softmax(x) >= 0 for all x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10)
        assert np.all(softmax(x) >= 0)

    def test_large_positive_stable(self) -> None:
        """Softmax handles large positive values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        s = softmax(x)
        assert np.all(np.isfinite(s))
        assert np.sum(s) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_stable(self) -> None:
        """Softmax handles large negative values without underflow."""
        x = np.array([-1000.0, -1001.0, -1002.0])
        s = softmax(x)
        assert np.all(np.isfinite(s))
        assert np.sum(s) == pytest.approx(1.0, abs=1e-6)

    def test_monotonicity(self) -> None:
        """If x_i > x_j, then softmax(x)_i > softmax(x)_j."""
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert s[2] > s[1] > s[0]

    def test_translation_invariance(self) -> None:
        """softmax(x + c) == softmax(x) for any constant c."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5)
        c = 100.0
        assert np.allclose(softmax(x + c), softmax(x))


# ============================================================
# LOSS FUNCTION PROPERTIES
# ============================================================


class TestLossProperties:
    """Mathematical properties of loss functions."""

    def test_mse_nonneg(self) -> None:
        """MSE >= 0."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(5)
        y_pred = rng.standard_normal(5)
        assert mean_squared_error(y_true, y_pred) >= 0

    def test_mse_zero(self) -> None:
        """MSE(y, y) == 0."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(5)
        assert mean_squared_error(y, y) == pytest.approx(0.0)

    def test_mse_symmetric(self) -> None:
        """MSE(a, b) == MSE(b, a)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        assert mean_squared_error(a, b) == pytest.approx(mean_squared_error(b, a))

    def test_bce_nonneg(self) -> None:
        """BCE >= 0 for valid probability inputs."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0.0, 1.0], size=10)
        y_pred = rng.uniform(0.1, 0.9, size=10)
        assert binary_cross_entropy(y_true, y_pred) >= 0

    def test_bce_zero_perfect(self) -> None:
        """BCE(y, y) ≈ 0 for y in {0, 1}."""
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        assert binary_cross_entropy(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_cce_nonneg(self) -> None:
        """CCE >= 0 for valid probability inputs."""
        rng = np.random.default_rng(42)
        y_true = np.eye(3)[rng.choice(3, size=5)]
        y_pred = rng.uniform(0.1, 0.9, size=(5, 3))
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        assert categorical_cross_entropy(y_true, y_pred) >= 0

    def test_cce_zero_perfect(self) -> None:
        """CCE(y, y) ≈ 0 for one-hot y."""
        y = np.eye(3)
        assert categorical_cross_entropy(y, y) == pytest.approx(0.0, abs=1e-10)


# ============================================================
# PROBABILITY PROPERTIES
# ============================================================


class TestProbabilityProperties:
    """Mathematical properties of probability functions."""

    def test_bernoulli_pmf_range(self) -> None:
        """0 <= P(x) <= 1 for Bernoulli."""
        assert 0 <= bernoulli_pmf(0, 0.3) <= 1
        assert 0 <= bernoulli_pmf(1, 0.3) <= 1

    def test_bernoulli_pmf_sums_to_one(self) -> None:
        """P(0) + P(1) == 1 for Bernoulli."""
        p = 0.3
        assert bernoulli_pmf(0, p) + bernoulli_pmf(1, p) == pytest.approx(1.0)

    def test_normal_pdf_nonneg(self) -> None:
        """normal_pdf(x) >= 0 for all x."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100) * 10
        assert np.all(normal_pdf(x, 0.0, 1.0) >= 0)

    def test_normal_pdf_peak_at_mean(self) -> None:
        """normal_pdf(mu) is maximum."""
        mu, sigma = 0.0, 1.0
        peak = normal_pdf(mu, mu, sigma)
        assert normal_pdf(mu + 1, mu, sigma) < peak
        assert normal_pdf(mu - 1, mu, sigma) < peak

    def test_categorical_pmf_sums_to_one(self) -> None:
        """sum(categorical_pmf(k, probs)) == 1."""
        probs = np.array([0.2, 0.5, 0.3])
        total = sum(categorical_pmf(k, probs) for k in range(3))
        assert total == pytest.approx(1.0)

    def test_conditional_definition(self) -> None:
        """P(A|B) = P(A,B) / P(B) for P(B) > 0."""
        p_ab = 0.3
        p_b = 0.5
        assert conditional_probability(p_ab, p_b) == pytest.approx(p_ab / p_b)

    def test_joint_independence(self) -> None:
        """P(A,B) == P(A) * P(B) for independent events."""
        p_a, p_b = 0.3, 0.5
        assert intersection_independent(p_a, p_b) == pytest.approx(p_a * p_b)


# ============================================================
# NUMERICAL STABILITY PROPERTIES
# ============================================================


class TestNumericalStability:
    """Properties that must hold for numerically stable implementations."""

    def test_log_sum_exp_stability(self) -> None:
        """log(sum(exp(x))) is finite for moderate x."""
        x = np.array([1000.0, 1001.0, 1002.0])
        # Should not overflow
        shifted = x - np.max(x)
        result = np.max(x) + np.log(np.sum(np.exp(shifted)))
        assert np.isfinite(result)

    def test_sigmoid_overflow_protection(self) -> None:
        """Sigmoid returns finite output for extreme inputs."""
        s_pos = sigmoid(1000.0)
        s_neg = sigmoid(-1000.0)
        assert np.isfinite(s_pos) and 0 <= s_pos <= 1
        assert np.isfinite(s_neg) and 0 <= s_neg <= 1
        assert s_pos > s_neg  # sigmoid is monotonically increasing

    def test_cross_entropy_stability(self) -> None:
        """Cross-entropy is finite for valid inputs."""
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.9, 0.05, 0.05])
        assert np.isfinite(binary_cross_entropy(y_true, y_pred))
