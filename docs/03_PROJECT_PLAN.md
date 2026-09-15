# 03_PROJECT_PLAN.md — Project Plan

## Stages Overview

| Stage | Name | Focus | Dependencies | Status |
|-------|------|-------|--------------|--------|
| 0 | Project Definition & Documentation | Repo structure, docs, tooling, planning | None | ✅ Complete |
| 1 | Linear Algebra Primitives | Scalars, vectors, matrices, core ops | Stage 0 | ✅ Implemented |
| 2 | Calculus & Numerical Differentiation | Derivatives, gradients, chain rule, verification | Stage 1 | 📋 Planned |
| 3 | Probability & Statistics Foundations | Distributions, entropy, cross-entropy, MLE | Stage 1 | 📋 Planned |
| 4 | Optimization Algorithms | GD, SGD, momentum, Adam | Stage 2 | 📋 Planned |
| 5 | Neural-Network Mathematics | Layers, activations, losses, embeddings | Stage 1, 3 | 📋 Planned |
| 6 | Backpropagation & Training Loops | Autograd, training, validation | Stage 2, 5 | 📋 Planned |
| 7 | Experiments & Visualization Suite | Structured experiments, plots, notebooks | Stage 4, 6 | 📋 Planned |
| 8 | PyTorch Comparison/Validation | Numerical parity checks | Stage 6 | 📋 Planned |

## Milestones & Implementation Order

### Stage 1: Linear Algebra Primitives
**Checkpoint:** All core ops tested, numerical verification framework established

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 1.1 | `Scalar`, `Vector`, `Matrix` classes with `__init__`, validation | Unit tests for construction, shape |
| 1.2 | Vector addition, scalar multiplication | Property tests (commutativity, associativity) |
| 1.3 | Dot product, cosine similarity | Numerical verification vs. NumPy |
| 1.4 | Matrix multiplication, transpose | Property tests (associativity, distributivity), numerical vs. NumPy |
| 1.5 | Norms (L1, L2, Linf), normalization | Edge cases (zero vector), numerical verification |
| 1.6 | Linear algebra submodule API, documentation | Import tests, docstring coverage |

### Stage 2: Calculus & Numerical Differentiation
**Checkpoint:** Analytical vs. finite-difference verification working

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 2.1 | Finite difference utilities (forward, central, complex-step) | Test against known derivatives |
| 2.2 | `derivative(f, x)`, `gradient(f, x)` functions | Numerical verification on polynomials, trig |
| 2.3 | Partial derivatives for multivariate functions | Test on known functions (e.g., f(x,y)=x²+y²) |
| 2.4 | Chain rule implementation (manual) | Compose functions, verify gradients |
| 2.5 | Analytical derivatives for key functions (sigmoid, ReLU, softmax, log, exp) | Numerical verification with tolerances |
| 2.6 | Calculus submodule API, documentation | Import tests, docstring coverage |

### Stage 3: Probability & Statistics Foundations
**Checkpoint:** Distributions, entropy, cross-entropy working and verified

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 3.1 | Probability distributions (Bernoulli, Categorical, Gaussian) | PMF/PDF integration = 1, sampling |
| 3.2 | Expectation, variance, covariance | Analytical vs. Monte Carlo |
| 3.3 | Log probability, entropy, cross-entropy | Numerical verification on known distributions |
| 3.4 | Likelihood, MLE for simple models | Compare with closed-form solutions |
| 3.5 | Probability submodule API, documentation | Import tests, docstring coverage |

### Stage 4: Optimization Algorithms
**Checkpoint:** Optimizers converge on test functions, trajectories visualizable

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 4.1 | Gradient descent with fixed/adaptive LR | Convergence on convex quadratics |
| 4.2 | SGD with minibatches (conceptual) | Noise behavior on simple loss |
| 4.3 | Momentum, Nesterov | Faster convergence vs. plain GD |
| 4.4 | Adam (conceptual implementation) | Standard benchmarks (Rosenbrock, etc.) |
| 4.5 | Learning rate schedules (step, cosine, warmup) | Visualization of LR curves |
| 4.6 | Optimization submodule API, documentation | Import tests, docstring coverage |

### Stage 5: Neural-Network Mathematics
**Checkpoint:** Linear layer + activations + loss compose correctly

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 5.1 | `Linear` layer (weight matrix + bias) | Shape correctness, forward pass |
| 5.2 | Activations: Sigmoid, ReLU, Tanh, GELU, Softmax | Numerical verification of derivatives |
| 5.3 | Loss functions: MSE, CrossEntropy, BCE | Gradient verification |
| 5.4 | Embeddings (lookup + gradient) | Gradient flows to correct indices |
| 5.5 | LayerNorm, RMSNorm | Forward/backward shape correctness |
| 5.6 | Attention (scaled dot-product) | Numerical verification |
| 5.7 | Neural networks submodule API, documentation | Import tests, docstring coverage |

### Stage 6: Backpropagation & Training Loops
**Checkpoint:** End-to-end training on toy problem (XOR, MNIST subset)

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 6.1 | Computation graph / autograd (minimal) | Gradient check vs. finite differences |
| 6.2 | `Sequential` container, `Module` base | Composition works |
| 6.3 | Training loop (forward, loss, backward, step) | Loss decreases on toy problem |
| 6.4 | Validation, checkpointing, logging | Reproducible runs with seeds |
| 6.5 | Backpropagation submodule API, documentation | Import tests, docstring coverage |

### Stage 7: Experiments & Visualization Suite
**Checkpoint:** Reproducible experiment reports with plots

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 7.1 | Experiment runner (config → run → log → plot) | Deterministic with fixed seed |
| 7.2 | Visualization: function plots, gradient fields, optimization paths | Publication-quality outputs |
| 7.3 | Loss landscape visualization (1D, 2D slices) | Correct contours |
| 7.4 | Notebook examples for each major concept | Executable, documented |
| 7.5 | Experiment log format, analysis templates | Consistent structure |

### Stage 8: PyTorch Comparison/Validation
**Checkpoint:** Numerical parity on core operations

| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| 8.1 | Linear layer parity (forward + backward) | Max relative error < 1e-5 |
| 8.2 | Activation parity | Max relative error < 1e-5 |
| 8.3 | Loss parity | Max relative error < 1e-5 |
| 8.4 | Optimizer step parity (SGD, Adam) | Parameter update matching |
| 8.5 | Small network training parity | Loss curves match within tolerance |

## Dependencies Between Milestones

```
Stage 0
  └── Stage 1 (Linear Algebra)
        ├── Stage 2 (Calculus) ← requires Vector/Matrix from Stage 1
        ├── Stage 3 (Probability) ← requires Vector/Matrix from Stage 1
        │       └── Stage 5 (NN Math) ← requires Probability (softmax, CE)
        └── Stage 5 (NN Math) ← requires Linear Algebra
Stage 2 (Calculus)
        ├── Stage 4 (Optimization) ← requires gradients
        └── Stage 6 (Backprop) ← requires chain rule, autograd
Stage 4 (Optimization)
        └── Stage 6 (Training Loops) ← requires optimizers
Stage 5 (NN Math)
        └── Stage 6 (Backprop) ← requires layers, activations, losses
Stage 6 (Backprop)
        ├── Stage 7 (Experiments) ← requires training capability
        └── Stage 8 (PyTorch Comparison) ← requires working implementation
```

## Verification Checkpoints

Each stage must pass before proceeding:

1. **Code Quality:** `ruff check .`, `ruff format . --check`, `mypy src/` — zero errors
2. **Tests:** `pytest -xvs` — all tests pass
3. **Numerical Verification:** Designated tests pass with defined tolerances
4. **Documentation:** All public APIs have docstrings; status markers updated (PLANNED→IMPLEMENTED→VERIFIED)
5. **Example:** At least one working example/notebook per stage

## Timeline Estimate (Rough)

| Stage | Estimated Effort |
|-------|------------------|
| 0 | 1 session (this work) |
| 1 | 2-3 sessions |
| 2 | 2-3 sessions |
| 3 | 2-3 sessions |
| 4 | 2-3 sessions |
| 5 | 3-4 sessions |
| 6 | 3-4 sessions |
| 7 | 2-3 sessions |
| 8 | 2-3 sessions |

**Total:** ~20-25 sessions for full MVP. Adjust based on depth and verification rigor.