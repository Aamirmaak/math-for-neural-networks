# 13_LEARNINGS.md — Intended Learning Outcomes

## Purpose

Document what this project is designed to teach through implementation. These are **intended** learning outcomes — not yet achieved. As each stage is completed, specific learnings will be recorded with evidence.

## Meta-Learning: How This Project Teaches

**Pedagogical Approach:** Implementation → Verification → Visualization → Connection

1. **Implement** the mathematics from scratch (no black boxes)
2. **Verify** numerically against analytical results (catch errors, understand precision)
3. **Visualize** to build geometric intuition (see what equations mean)
4. **Connect** to neural network operations (see why the math matters)

---

## Mathematical Foundations

### Linear Algebra
**Intended Learning:**
- Vectors as geometric objects (direction + magnitude), not just arrays
- Matrix multiplication as linear transformation composition
- Dot product as projection measure → cosine similarity
- Norms as distance measures → geometry of high-dimensional spaces
- Transpose as adjoint → connection to gradient computation
- **Why it matters for NN:** Weight matrices transform input space; embeddings are vectors; attention is dot products

### Calculus
**Intended Learning:**
- Derivative as local linear approximation (Jacobian)
- Gradient as direction of steepest ascent
- Chain rule as gradient flow through computation graphs
- Numerical differentiation: finite differences (tradeoffs: h too small → cancellation, too large → truncation)
- Complex-step differentiation: no cancellation, but requires complex support
- **Why it matters for NN:** Backpropagation IS the chain rule; gradients flow backward through layers

### Probability & Statistics
**Intended Learning:**
- Distributions as mathematical objects with PMF/PDF/CDF
- Expectation as weighted average; variance as spread
- Entropy as uncertainty measure; cross-entropy as "surprise" of Q given P
- KL divergence as asymmetry → not a metric
- MLE as principle: find parameters that make data most likely
- **Why it matters for NN:** Softmax outputs probabilities; cross-entropy loss; language modeling is MLE

### Optimization
**Intended Learning:**
- Gradient descent as following local slope
- Learning rate as step size: too large → divergence, too small → slow
- Momentum as velocity: smooths oscillations, accelerates in consistent directions
- Adam as per-parameter adaptive learning rates + momentum
- Convergence theory: convex → global optimum; non-convex → critical points
- **Why it matters for NN:** Training IS optimization; architecture choices affect landscape

---

## Neural Network Mathematics

### Layers & Activations
**Intended Learning:**
- Linear layer: affine transformation y = xW^T + b
- Activations as non-linearities: without them, deep = shallow
- Sigmoid: saturates → vanishing gradients
- ReLU: sparse gradients, dead neurons
- GELU: smooth ReLU variant, used in Transformers
- Softmax: converts logits to probabilities (temperature controls sharpness)
- **Why it matters:** Architecture design = choosing transformations

### Loss Functions
**Intended Learning:**
- MSE: L2 distance → regression
- Cross-entropy: -log P(true class) → classification
- BCE: per-class independent probabilities → multi-label
- Gradient of loss w.r.t. logits: simple, elegant forms
- **Why it matters:** Loss defines what "good" means; gradient drives learning

### Backpropagation
**Intended Learning:**
- Forward pass: compute values, store for backward
- Backward pass: compute gradients via chain rule (reverse mode autodiff)
- Gradient accumulation at shared parameters (e.g., embedding lookup)
- Vectorized backprop: batch gradients = sum/mean of per-sample gradients
- **Why it matters:** This IS how neural networks learn

### Embeddings & Attention
**Intended Learning:**
- Embeddings: discrete → continuous; gradients only flow to accessed indices
- Attention: weighted sum of values, weights = softmax(QK^T/√d)
- Multi-head: parallel attention → concatenate → project
- Causal masking: prevent future token attention
- **Why it matters:** Foundation of Transformers, LLMs

### Normalization
**Intended Learning:**
- BatchNorm: normalize per feature across batch → stabilizes training
- LayerNorm: normalize per sample across features → works for variable length
- RMSNorm: simpler, no mean subtraction → used in LLMs
- Gradient flow through normalization: non-trivial, affects scale
- **Why it matters:** Enables deep networks, stable training

---

## Numerical & Computational Skills

### Numerical Verification
**Intended Learning:**
- Analytical vs. numerical gradients: the ultimate correctness check
- Tolerance selection: float64 precision, FD error, problem conditioning
- Catastrophic cancellation: subtracting nearby numbers loses precision
- Condition number: how input errors amplify in output
- **Skill:** Writing verification tests that catch real bugs, not floating-point noise

### Numerical Stability
**Intended Learning:**
- Log-sum-exp trick for softmax stability
- Gradient clipping for explosion prevention
- Weight initialization: Xavier/He for variance preservation
- Mixed precision: when float32 is enough, when float64 needed
- **Skill:** Implementing math that works in practice, not just theory

### Experimentation Methodology
**Intended Learning:**
- Hypothesis-driven experimentation (not random tweaking)
- Baselines: know what you're comparing against
- Reproducibility: seeds, configs, environment capture
- Analysis: separate observation from interpretation
- **Skill:** Scientific approach to ML engineering

---

## Software Engineering for ML

### Clean Architecture
**Intended Learning:**
- Separation of concerns: primitives → operations → NN math → verification → viz → experiments
- Dependency direction: inward toward primitives
- No circular dependencies
- **Skill:** Building maintainable, extensible ML codebases

### Testing Mathematical Code
**Intended Learning:**
- Property-based testing: mathematical laws as tests (commutativity, triangle inequality)
- Numerical verification tests: analytical vs. finite difference
- Edge case coverage: zeros, extremes, NaN/Inf
- Regression tests: prevent re-introducing fixed bugs
- **Skill:** Confidence in correctness through automated testing

### Type-Driven Development
**Intended Learning:**
- Types as documentation: function signatures encode mathematical contracts
- Generics for shape-safe operations (future: shape typing)
- Protocol-based extensibility (Optimizer, Distribution, Module)
- **Skill:** Using type system to prevent bugs and clarify design

---

## Research Engineering Mindset

### From Math to Code to Insight
**Intended Learning:**
- Reading a paper → identifying the mathematical core → implementing minimal version
- Verifying implementation matches paper equations
- Experimenting with variations to understand design choices
- Documenting findings for future reference

### Bridging to Production Frameworks
**Intended Learning:**
- Understanding what PyTorch/JAX do under the hood
- Knowing when to use framework vs. custom implementation
- Debugging framework models by understanding the math
- Contributing to frameworks with mathematical rigor

---

## Stage-by-Stage Learning Checkpoints

| Stage | Key Learning Milestone |
|-------|------------------------|
| 1 | Implement vector/matrix ops; verify against NumPy; understand broadcasting |
| 2 | Implement gradients; verify with finite differences; understand chain rule mechanically |
| 3 | Implement distributions, entropy, MLE; verify integrals/sums = 1 |
| 4 | Implement optimizers; visualize trajectories; understand momentum/Adam mechanics |
| 5 | Implement layers, activations, losses; verify gradients; understand NN building blocks |
| 6 | Implement backprop; train on XOR/MNIST; understand gradient flow |
| 7 | Design experiments; create visualizations; communicate results |
| 8 | Compare with PyTorch; understand framework internals; validate correctness |

---

## How Learnings Will Be Recorded

As each stage completes, specific entries will be added to this document:

```markdown
## Stage N: Stage Name — Learnings (YYYY-MM-DD)

### Confirmed
- Learning 1 (with evidence: experiment ID, verification result)
- Learning 2 (with evidence: ...)

### Surprises
- Unexpected behavior discovered
- Numerical issue encountered

### Corrections
- Prior misconception corrected
- Implementation approach changed

### Open Questions
- Things not yet understood
- Areas for deeper study
```

**Current Status:** All learnings are INTENDED. No learnings confirmed yet.

---

## Stage 1: Linear Algebra — Learnings (2026-09-16)

### Confirmed
- Dot product implementation (triple-loop) produces identical results to NumPy within 1e-10
- Matrix multiplication triple-loop correctly implements C_ij = sum_k A_ik * B_kj
- Projection formula proj_b(a) = ((a · b) / (b · b)) * b decomposes vectors correctly
- Cosine similarity range is guaranteed [-1, 1] for non-zero vectors
- L2 norm squared equals self dot product: ||v||^2 = v · v
- Eigenvalue relationship A v = lambda v verified numerically for random symmetric matrices
- Matrix multiplication is NOT commutative: AB != BA (verified with concrete examples)

### Surprises
- The educational triple-loop matrix multiplication is ~1000x slower than NumPy for large matrices, but that's expected and fine for educational purposes

### Corrections
- None yet

### Open Questions
- How do conditioning and numerical stability affect very large matrix operations?
- When does from-scratch eigenvalue implementation become educational rather than tedious?

---

## Stage 2: Calculus — Learnings (2026-09-16)

### Confirmed
- Central differences are O(h²) accurate — typically 100x better than forward differences for smooth functions
- Optimal step size h ≈ 1e-5 for float64 precision; too small causes roundoff errors (floating-point subtraction cancellation)
- Too large h causes truncation error; too small h causes roundoff error — there's a sweet spot
- Sigmoid derivative σ'(x) = σ(x)(1-σ(x)) — elegant formula that avoids recomputing exp
- Sigmoid saturates for |x| > 5 → derivatives vanish → no learning (vanishing gradient problem)
- ReLU derivative is piecewise constant (0 or 1) — no saturation for positive inputs
- The chain rule IS backpropagation — each layer computes local derivatives, backward pass multiplies them
- Deep sigmoid networks suffer from vanishing gradients: product of many small derivatives → zero
- Gradient magnitude tells us steepness; gradient direction tells us steepest ascent
- Gradient descent follows negative gradient to find minima

### Surprises
- ReLU derivative at x=0 is undefined mathematically, but we define it as 0 in practice (convention)
- The chain rule for a single neuron (z = wx + b, loss = (z-target)²) produces clean, interpretable gradients
- Product of 10 sigmoid derivatives can be as small as ~1e-4 — gradient essentially vanishes

### Corrections
- Fixed relu_derivative to handle scalar inputs (bool.has no .astype method) — need np.asarray first
- Fixed log_func to validate domain (x > 0) rather than relying on NumPy's warning behavior

### Open Questions
- How do second-order derivatives (Hessian) affect optimization landscape?
- What is the relationship between condition number and gradient descent convergence rate?
- How do adaptive learning rates (Adam) compensate for varying gradient magnitudes?