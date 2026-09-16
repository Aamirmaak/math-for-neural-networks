"""
Maximum Likelihood Estimation (MLE)
=====================================

MLE finds the parameter theta that maximizes the likelihood of observed data.

For Bernoulli: p_MLE = (count of successes) / (total trials)
For Gaussian: mu_MLE = sample mean, sigma_MLE = sample standard deviation

This connects probability to learning from data:
    Data -> Likelihood -> Optimization -> Learned Parameters
"""

import numpy as np

from math_for_neural_networks.probability.likelihood import (
    bernoulli_log_likelihood,
    bernoulli_mle,
    gaussian_log_likelihood,
    gaussian_mle,
)


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def example_1_bernoulli_mle() -> None:
    """MLE for Bernoulli: simple coin flip estimation."""
    print_section("Example 1: Bernoulli MLE - Coin Flip")

    print("""
    We observe n coin flips and want to estimate the probability p of heads.
    MLE: p_MLE = (number of heads) / (total flips)
    """)

    # Generate data
    np.random.seed(42)
    true_p = 0.7
    data = np.random.binomial(1, true_p, size=100)

    mle = bernoulli_mle(data)
    print(f"  True p:      {true_p}")
    print(f"  Observed:    {np.sum(data)} heads in {len(data)} flips")
    print(f"  MLE estimate: {mle:.4f}")
    print(f"  Error:       {abs(mle - true_p):.4f}")

    print("\n  As sample size increases, MLE converges to true p:")

    for n in [10, 50, 100, 500, 1000]:
        data = np.random.binomial(1, true_p, size=n)
        mle = bernoulli_mle(data)
        print(f"    n = {n:>4}: p_MLE = {mle:.4f} (error = {abs(mle - true_p):.4f})")


def example_2_gaussian_mle() -> None:
    """MLE for Gaussian: mean and variance estimation."""
    print_section("Example 2: Gaussian MLE - Height Estimation")

    print("""
    We observe data and want to estimate mu and sigma of a Gaussian distribution.
    MLE: mu_MLE = sample mean, sigma_MLE = sqrt(sample variance)
    """)

    np.random.seed(42)
    true_mu = 170.0  # cm
    true_sigma = 10.0
    data = np.random.normal(true_mu, true_sigma, size=200)

    mu_mle, sigma_mle = gaussian_mle(data)
    print(f"  True mu = {true_mu}, sigma = {true_sigma}")
    print(f"  MLE mu  = {mu_mle:.2f}, sigma = {sigma_mle:.2f}")
    print(f"  Error in mu:  {abs(mu_mle - true_mu):.2f}")
    print(f"  Error in sigma:  {abs(sigma_mle - true_sigma):.2f}")


def example_3_likelihood_curve() -> None:
    """How likelihood changes with parameter value."""
    print_section("Example 3: Likelihood vs Parameter")

    print("""
    For observed data, we can plot how the likelihood changes as we vary p.
    The MLE is the peak of this curve.
    """)

    # Observed: 7 heads in 10 flips
    data = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    mle = bernoulli_mle(data)

    print(f"  Data: 7 heads in 10 flips")
    print(f"  MLE: p = {mle:.1f}")

    print(f"\n  {'p':>6} | {'log L(p)':>10}")
    print("  " + "-" * 20)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        ll = bernoulli_log_likelihood(data, p)
        marker = " <- MLE" if abs(p - mle) < 0.01 else ""
        print(f"  {p:>6.1f} | {ll:>10.4f}{marker}")

    print(f"\n  The log-likelihood is maximized at the MLE.")


if __name__ == "__main__":
    print("MAXIMUM LIKELIHOOD ESTIMATION")
    print("=" * 55)

    example_1_bernoulli_mle()
    example_2_gaussian_mle()
    example_3_likelihood_curve()
