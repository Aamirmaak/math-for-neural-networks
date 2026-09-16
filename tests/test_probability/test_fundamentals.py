"""Tests for probability fundamentals."""

import numpy as np
import pytest

from math_for_neural_networks.probability.fundamentals import (
    bayes_theorem,
    complement,
    conditional_probability,
    intersection_independent,
    is_independent,
    joint_from_conditional,
    marginal_probability,
    probability_axiom_check,
    union_exclusive,
    union_general,
    validate_probabilities,
    validate_probability,
)


class TestValidateProbability:
    def test_valid(self) -> None:
        validate_probability(0.0)
        validate_probability(0.5)
        validate_probability(1.0)

    def test_invalid_below(self) -> None:
        with pytest.raises(ValueError):
            validate_probability(-0.1)

    def test_invalid_above(self) -> None:
        with pytest.raises(ValueError):
            validate_probability(1.1)


class TestValidateProbabilities:
    def test_valid(self) -> None:
        validate_probabilities(np.array([0.2, 0.3, 0.5]))

    def test_invalid_negative(self) -> None:
        with pytest.raises(ValueError):
            validate_probabilities(np.array([-0.1, 0.5, 0.6]))

    def test_does_not_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            validate_probabilities(np.array([0.2, 0.2, 0.2]))


class TestAxiomCheck:
    def test_valid(self) -> None:
        result = probability_axiom_check(np.array([0.25, 0.25, 0.5]))
        assert result["axiom_1_non_negative"] is True
        assert result["axiom_2_total_one"] is True

    def test_negative(self) -> None:
        result = probability_axiom_check(np.array([-0.1, 0.6, 0.5]))
        assert result["axiom_1_non_negative"] is False

    def test_not_summing_to_one(self) -> None:
        result = probability_axiom_check(np.array([0.2, 0.2, 0.2]))
        assert result["axiom_2_total_one"] is False


class TestComplement:
    def test_basic(self) -> None:
        assert complement(0.3) == pytest.approx(0.7)
        assert complement(0.0) == pytest.approx(1.0)
        assert complement(1.0) == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        p = 0.7
        assert complement(complement(p)) == pytest.approx(p)


class TestUnionExclusive:
    def test_basic(self) -> None:
        assert union_exclusive(0.3, 0.4) == pytest.approx(0.7)

    def test_zero(self) -> None:
        assert union_exclusive(0.0, 0.0) == pytest.approx(0.0)

    def test_exceeds_one(self) -> None:
        with pytest.raises(ValueError):
            union_exclusive(0.7, 0.5)


class TestUnionGeneral:
    def test_basic(self) -> None:
        assert union_general(0.5, 0.4, 0.2) == pytest.approx(0.7)

    def test_disjoint(self) -> None:
        assert union_general(0.3, 0.4, 0.0) == pytest.approx(0.7)

    def test_subset(self) -> None:
        assert union_general(0.5, 0.3, 0.3) == pytest.approx(0.5)


class TestIntersectionIndependent:
    def test_independent(self) -> None:
        assert intersection_independent(0.5, 0.4) == pytest.approx(0.2)

    def test_zero(self) -> None:
        assert intersection_independent(0.0, 0.5) == pytest.approx(0.0)

    def test_one(self) -> None:
        assert intersection_independent(1.0, 1.0) == pytest.approx(1.0)


class TestConditionalProbability:
    def test_basic(self) -> None:
        assert conditional_probability(0.2, 0.5) == pytest.approx(0.4)

    def test_zero_denominator(self) -> None:
        with pytest.raises(ValueError, match="P\\(B\\) = 0"):
            conditional_probability(0.0, 0.0)


class TestIndependence:
    def test_independent(self) -> None:
        assert is_independent(0.5, 0.4, 0.2) is True

    def test_dependent(self) -> None:
        assert is_independent(0.5, 0.4, 0.3) is False


class TestBayesTheorem:
    def test_basic(self) -> None:
        result = bayes_theorem(p_a=0.01, p_b_given_a=0.9, p_b=0.059)
        assert result == pytest.approx(0.01 * 0.9 / 0.059, rel=1e-3)

    def test_zero_evidence(self) -> None:
        with pytest.raises(ValueError, match="P\\(B\\) = 0"):
            bayes_theorem(0.5, 0.5, 0.0)


class TestMarginalProbability:
    def test_sum_rows(self) -> None:
        joint = np.array([[0.1, 0.2], [0.3, 0.4]])
        marginal = marginal_probability(joint, axis=0)
        np.testing.assert_allclose(marginal, [0.4, 0.6])

    def test_sum_cols(self) -> None:
        joint = np.array([[0.1, 0.2], [0.3, 0.4]])
        marginal = marginal_probability(joint, axis=1)
        np.testing.assert_allclose(marginal, [0.3, 0.7])

    def test_not_2d(self) -> None:
        with pytest.raises(ValueError, match="Expected 2D"):
            marginal_probability(np.array([1, 2, 3]))


class TestJointFromConditional:
    def test_basic(self) -> None:
        p_y = np.array([0.3, 0.7])
        p_x_given_y = np.array([[0.8, 0.2], [0.2, 0.8]])
        joint = joint_from_conditional(p_y, p_x_given_y)
        assert joint.shape == (2, 2)
        np.testing.assert_allclose(joint.sum(axis=0), p_y)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError):
            joint_from_conditional(np.array([0.5, 0.5]), np.array([[0.5]]))
