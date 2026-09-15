# 01_PRD.md — Product Requirements Document

## Problem Statement

Many beginners learn neural networks by memorizing mathematical operations (matrix multiplication, derivatives, gradients, chain rule, probability, loss functions, optimization) without developing an operational understanding of what the mathematics is actually doing. This creates a fragile foundation that breaks down when debugging, extending, or researching novel architectures.

## Users

**Primary:** AI/ML learners who want deep mathematical understanding through implementation

**Secondary:**
- ML/DL students
- Aspiring AI engineers
- Researchers learning mathematical foundations
- Developers implementing neural networks from scratch
- Educators who want computational demonstrations

## Goals

1. **Mathematical Understanding Through Implementation** — Teach by building, not memorizing
2. **Numerical Verification as Core Principle** — Compare analytical results vs. numerical approximations
3. **Clean, Modular Architecture** — Separate primitives, operations, NN math, verification, visualization, experiments
4. **Reproducible Experiments** — Structured lifecycle with clear hypothesis/result separation
5. **Progressive Learning Path** — Linear Algebra → Calculus → Probability → Optimization → NN Mathematics → Backpropagation
6. **Open-Source Ready** — Professional structure, documentation, testing, licensing

## Non-Goals

- Production ML framework (use PyTorch/JAX/TensorFlow)
- GPU acceleration or distributed training
- Automatic differentiation engine (manual for learning)
- Web UI / dashboard (CLI, notebooks, scripts only)
- Comprehensive coverage of all mathematical topics
- Replacement for textbooks or courses
- Commercial deployment infrastructure

## Requirements

### Functional Requirements (Planned)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Implement scalar, vector, matrix primitives with basic operations | MVP |
| FR-02 | Implement dot product, matrix multiplication, transpose, norms | MVP |
| FR-03 | Implement numerical differentiation (finite differences) | MVP |
| FR-04 | Implement analytical derivatives for key functions | MVP |
| FR-05 | Implement gradient computation and verification | MVP |
| FR-06 | Implement probability distributions, entropy, cross-entropy | MVP |
| FR-07 | Implement gradient descent, SGD, momentum, Adam | MVP |
| FR-08 | Implement linear layers, activations (sigmoid, ReLU, softmax) | MVP |
| FR-09 | Implement loss functions (MSE, cross-entropy) | MVP |
| FR-10 | Implement backpropagation for simple networks | MVP |
| FR-11 | Numerical verification framework (analytical vs. numerical) | MVP |
| FR-12 | Visualization utilities for functions, gradients, optimization paths | MVP |
| FR-13 | Experiment runner with structured logging | MVP |
| FR-14 | CLI for common operations and demonstrations | Future |
| FR-15 | Jupyter notebook examples for each major concept | Future |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Python 3.10+ |
| NFR-02 | NumPy as primary numerical dependency |
| NFR-03 | Matplotlib for visualization |
| NFR-04 | pytest for testing |
| NFR-05 | Type hints throughout (mypy strict) |
| NFR-06 | Ruff for linting/formatting |
| NFR-07 | MIT license |
| NFR-08 | Reproducible installation via pyproject.toml |
| NFR-09 | No mandatory heavy ML frameworks |
| NFR-10 | Clear PLANNED/IMPLEMENTED/VERIFIED/DEFERRED status tracking |

## MVP Scope

The Minimum Viable Product focuses on a carefully selected core:

**Linear Algebra:** Scalars, vectors, matrices, addition, dot product, matrix multiplication, transpose, norms, cosine similarity

**Calculus:** Derivative, partial derivative, gradient, chain rule, numerical differentiation, selected analytical derivatives

**Probability/Statistics:** Probability, conditional probability, expectation, variance, distributions, log probability, entropy, cross-entropy, likelihood, MLE

**Optimization:** Gradient descent, learning rate, SGD concept, momentum, Adam concept

**Neural-Network Mathematics:** Linear layers, activations (sigmoid, softmax), loss functions, backpropagation, embeddings, attention, normalization

**Verification:** Analytical vs. numerical comparison framework with tolerances

**Visualization:** Function plots, gradient fields, optimization trajectories, loss landscapes

## Future Scope (Deferred)

- Deeper tensor concepts (beyond 2D)
- Eigenvalues/eigenvectors, matrix decompositions (SVD, QR, Cholesky)
- Projections, orthogonalization
- More numerical methods (Newton, quasi-Newton)
- Richer optimization experiments (learning rate schedules, adaptive methods)
- Advanced loss functions (focal, contrastive, triplet)
- Attention mathematics (scaled dot-product, multi-head)
- Transformer-related mathematical demonstrations
- PyTorch comparison/validation suite
- Interactive visualizations (optional, if educational value proven)
- Educational CLI experiences
- Benchmark/experiment suites
- Additional notebooks
- Research-oriented experiments

## Success Criteria

1. **Stage 0 Complete** — All documentation, structure, tooling established ✅
2. **Stage 1 Complete** — Linear algebra primitives implemented, tested, verified
3. **Stage 2 Complete** — Calculus primitives implemented, numerical verification working
4. **Stage 3 Complete** — Probability/statistics primitives implemented
5. **Stage 4 Complete** — Optimization algorithms implemented and visualized
6. **Stage 5 Complete** — Neural-network mathematics implemented
7. **Stage 6 Complete** — Backpropagation working on simple networks
8. **Stage 7 Complete** — Experiment suite and visualizations documented
9. **Stage 8 Complete** — PyTorch comparison validation passing

Each stage must have:
- All code typed, linted, tested
- Numerical verification passing defined tolerances
- Documentation updated with IMPLEMENTED/VERIFIED status
- At least one educational notebook/example