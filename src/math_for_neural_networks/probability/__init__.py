"""
Probability & Statistics Module
================================

Educational probability toolkit for understanding the mathematical foundations
of machine learning, neural networks, and statistical modeling.

Modules:
    fundamentals: Probability axioms, events, conditional probability, Bayes' theorem
    distributions: Bernoulli, Binomial, Categorical, Uniform, Normal
    moments: Expectation, variance, standard deviation
    information: Entropy, cross-entropy, KL divergence, log probability
    likelihood: Likelihood, log-likelihood, MLE
    stability: Numerically stable softmax, log-sum-exp, cross-entropy with logits
"""

from math_for_neural_networks.probability.distributions import (
    bernoulli_expectation,
    bernoulli_pmf,
    bernoulli_variance,
    binomial_expectation,
    binomial_pmf,
    binomial_variance,
    categorical_expectation,
    categorical_pmf,
    categorical_variance,
    normal_cdf,
    normal_expectation,
    normal_pdf,
    normal_variance,
    uniform_expectation,
    uniform_pdf,
    uniform_variance,
)
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
from math_for_neural_networks.probability.information import (
    cross_entropy,
    entropy,
    kl_divergence,
    log_probability,
)
from math_for_neural_networks.probability.likelihood import (
    bernoulli_likelihood,
    bernoulli_log_likelihood,
    bernoulli_mle,
    categorical_log_likelihood,
    gaussian_log_likelihood,
    gaussian_mle,
)
from math_for_neural_networks.probability.moments import (
    expectation,
    standard_deviation,
    variance,
)
from math_for_neural_networks.probability.stability import (
    binary_cross_entropy_with_logits,
    cross_entropy_with_logits,
    log_softmax,
    log_sum_exp,
    sigmoid,
    softmax,
)

__all__ = [
    # Fundamentals
    "validate_probability",
    "validate_probabilities",
    "probability_axiom_check",
    "complement",
    "union_exclusive",
    "union_general",
    "intersection_independent",
    "conditional_probability",
    "is_independent",
    "bayes_theorem",
    "marginal_probability",
    "joint_from_conditional",
    # Distributions
    "bernoulli_pmf",
    "bernoulli_expectation",
    "bernoulli_variance",
    "binomial_pmf",
    "binomial_expectation",
    "binomial_variance",
    "categorical_pmf",
    "categorical_expectation",
    "categorical_variance",
    "uniform_pdf",
    "uniform_expectation",
    "uniform_variance",
    "normal_pdf",
    "normal_expectation",
    "normal_variance",
    "normal_cdf",
    # Moments
    "expectation",
    "variance",
    "standard_deviation",
    # Information
    "entropy",
    "cross_entropy",
    "kl_divergence",
    "log_probability",
    # Likelihood
    "bernoulli_likelihood",
    "bernoulli_log_likelihood",
    "bernoulli_mle",
    "gaussian_log_likelihood",
    "gaussian_mle",
    "categorical_log_likelihood",
    # Stability
    "log_sum_exp",
    "softmax",
    "log_softmax",
    "sigmoid",
    "binary_cross_entropy_with_logits",
    "cross_entropy_with_logits",
]
