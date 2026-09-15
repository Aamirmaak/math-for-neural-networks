# 05_RESEARCH_PLAN.md — Research Plan

## Mathematical Topics to Study

### Linear Algebra (Foundation)
- **Core:** Vector spaces, linear transformations, basis, dimension
- **Operations:** Addition, scalar multiplication, dot product, matrix multiplication
- **Properties:** Associativity, distributivity, transpose properties
- **Norms:** L1, L2, L∞, p-norms; equivalence in finite dimensions
- **Special matrices:** Identity, diagonal, orthogonal, symmetric, positive definite
- **Decompositions (future):** LU, QR, SVD, eigendecomposition, Cholesky
- **Geometric interpretation:** Projections, angles, orthogonality, subspaces

### Calculus (Foundation)
- **Single-variable:** Limits, continuity, differentiability, mean value theorem
- **Derivatives:** Definition, rules (sum, product, quotient, chain), higher-order
- **Multivariable:** Partial derivatives, gradient, Jacobian, Hessian
- **Chain rule:** General form, computational graph interpretation
- **Optimization theory:** Critical points, convexity, gradient descent convergence
- **Numerical differentiation:** Finite differences (forward, central, complex-step), error analysis
- **Automatic differentiation:** Forward mode, reverse mode (future)

### Probability & Statistics (Foundation)
- **Probability spaces:** Sample space, events, probability axioms
- **Random variables:** Discrete, continuous, mixed; PMF, PDF, CDF
- **Expectation:** Linearity, law of unconscious statistician
- **Variance, covariance, correlation:** Properties, matrix form
- **Key distributions:** Bernoulli, Binomial, Categorical, Gaussian, Exponential, Gamma, Beta, Dirichlet
- **Information theory:** Entropy, cross-entropy, KL divergence, mutual information
- **Maximum likelihood:** Principle, asymptotic properties, Fisher information
- **Bayesian perspective:** Prior, likelihood, posterior, conjugate priors (future)

### Optimization (Core)
- **Gradient descent:** Step size, convergence rates, convex vs. non-convex
- **Stochastic optimization:** SGD, variance reduction, minibatching
- **Momentum methods:** Heavy ball, Nesterov acceleration
- **Adaptive methods:** AdaGrad, RMSProp, Adam, AdamW
- **Second-order:** Newton, quasi-Newton (BFGS, L-BFGS) — future
- **Constrained optimization:** Lagrange multipliers, KKT — future
- **Learning rate schedules:** Constant, step, cosine, warmup, cyclical

### Neural-Network Mathematics (Application)
- **Linear layers:** Affine transformations, weight initialization
- **Activation functions:** Sigmoid, tanh, ReLU, GELU, Swish, softmax
- **Loss functions:** MSE, cross-entropy, BCE, hinge, contrastive
- **Backpropagation:** Chain rule on computational graphs, vectorized
- **Embeddings:** Lookup tables, gradient flow, positional encodings
- **Normalization:** BatchNorm, LayerNorm, RMSNorm, GroupNorm
- **Attention:** Scaled dot-product, multi-head, causal masking
- **Architecture patterns:** Residual connections, MLPs, Transformers

## Recommended Progression

```
Phase 1: Foundations (Stages 1-3)
├── Linear Algebra → Vector/Matrix ops, geometry
├── Calculus → Derivatives, gradients, chain rule
└── Probability → Distributions, entropy, MLE

Phase 2: Optimization (Stage 4)
└── GD, SGD, Momentum, Adam on mathematical functions

Phase 3: Neural Network Mathematics (Stages 5-6)
├── Layers, activations, losses
├── Backpropagation, training loops
└── Embeddings, attention, normalization

Phase 4: Validation & Experimentation (Stages 7-8)
├── Structured experiments
├── Visualization suite
└── PyTorch numerical parity
```

## Sources / Reference Categories

### Primary References (Textbooks)
- **Linear Algebra:** Strang "Introduction to Linear Algebra", Axler "Linear Algebra Done Right"
- **Calculus:** Stewart "Calculus", Spivak "Calculus" (rigorous)
- **Multivariable Calculus:** Marsden & Tromba "Vector Calculus"
- **Probability:** Blitzstein & Hwang "Introduction to Probability", Bishop "Pattern Recognition" (Ch. 1-2)
- **Information Theory:** Cover & Thomas "Elements of Information Theory"
- **Optimization:** Nocedal & Wright "Numerical Optimization", Boyd & Vandenberghe "Convex Optimization"
- **Deep Learning:** Goodfellow, Bengio, Courville "Deep Learning" (Ch. 4-8)
- **Neural Networks:** Nielsen "Neural Networks and Deep Learning" (free online)

### Technical References (Papers / Docs)
- **Automatic Differentiation:** Baydin et al. "Automatic differentiation in ML" (2018)
- **Adam:** Kingma & Ba "Adam: A Method for Stochastic Optimization" (2015)
- **LayerNorm:** Ba et al. "Layer Normalization" (2016)
- **Attention:** Vaswani et al. "Attention Is All You Need" (2017)
- **Transformers:** Multiple — focus on mathematical formulation
- **Numerical Recipes:** Press et al. (for finite difference methods)

### Implementation References (Code)
- **NumPy docs:** Array manipulation, linear algebra, random
- **Micrograd:** Karpathy's minimal autograd (educational)
- **PyTorch source:** For comparison/validation only
- **JAX docs:** For autodiff design patterns (reference only)

## Questions to Investigate

### Numerical Verification
1. What tolerances are appropriate for float64 vs float32 comparisons?
2. When does complex-step differentiation outperform central differences?
3. How to detect and handle catastrophic cancellation in gradient verification?
4. What are good test functions for gradient verification (beyond polynomials)?

### Optimization
1. How do learning rate schedules interact with momentum on pathological landscapes?
2. What are minimal examples where Adam fails but SGD+momentum succeeds?
3. How to visualize optimization trajectories in high dimensions meaningfully?

### Neural Network Mathematics
1. What is the exact gradient flow through LayerNorm vs BatchNorm?
2. How does attention gradient scale with sequence length?
3. What initialization schemes preserve gradient variance at depth?

### Educational Effectiveness
1. Which visualizations most improve understanding of gradient descent?
2. What minimal examples best illustrate the chain rule in backprop?
3. How to design exercises that reveal misconceptions (e.g., gradient ≠ derivative)?

## How Research Findings Influence Implementation

| Finding Type | Implementation Impact |
|--------------|----------------------|
| Numerical stability issue | Add verification test, document workaround, adjust tolerance |
| Better algorithm variant | Implement as alternative, add comparison experiment |
| Mathematical correction | Fix implementation, add regression test, update docs |
| Pedagogical insight | Add notebook example, improve visualization, add exercise |
| Performance bottleneck | Profile, optimize hot path, document complexity |

**Process:** Research → Document in `13_LEARNINGS.md` → Implement fix/improvement → Add verification test → Update documentation status

## Note

This plan identifies topics and sources for **future study**. No research has been performed yet. As each stage is implemented, relevant research will be conducted and findings recorded in `13_LEARNINGS.md`.