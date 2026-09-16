# 14_CHANGELOG.md — Changelog

## Format
Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) with project-specific conventions.

### Entry Types
- **Added** — New functionality
- **Changed** — Changes in existing functionality
- **Deprecated** — Soon-to-be removed functionality
- **Removed** — Now removed functionality
- **Fixed** — Bug fixes
- **Security** — Vulnerability fixes
- **Documentation** — Documentation only changes
- **Infrastructure** — Tooling, CI, packaging changes

### Status Markers
Each entry may include status: `[PLANNED]`, `[IMPLEMENTED]`, `[VERIFIED]`, `[DEFERRED]`

---

## [Unreleased]

---

## [0.4.0] — 2026-09-16 (Stage 3: Probability & Statistics Foundations)

### Added
- **probability/fundamentals.py**: Probability axioms, complement, union, intersection, conditional probability, Bayes theorem, marginal/joint distributions
- **probability/distributions.py**: Bernoulli, Binomial, Categorical, Uniform, Normal distributions (PMF/PDF/CDF/E/V)
- **probability/moments.py**: Expectation, variance, standard deviation
- **probability/information.py**: Entropy, cross-entropy, KL divergence, log probability
- **probability/likelihood.py**: Likelihood, log-likelihood, Bernoulli MLE, Gaussian MLE, categorical log-likelihood
- **probability/stability.py**: Log-sum-exp, softmax, log-softmax, sigmoid, binary/cross-entropy with logits

### Tests
- 149 tests across 7 test files (all passing)
- Unit tests for all probability operations
- Numerical verification tests comparing against NumPy/scipy
- Edge case tests (empty distributions, zero probabilities, extreme values)
- Properties tests (entropy bounds, KL divergence non-negativity)

### Examples
- examples/entropy.py — Entropy intuition, uniform/Bernoulli entropy, model confidence
- examples/cross_entropy.py — One-hot cross-entropy, KL decomposition, softmax pipeline
- examples/mle.py — Bernoulli/Gaussian MLE, likelihood curves

### Experiments
- experiments/entropy_categorical.py — Entropy vs concentration
- experiments/cross_entropy_loss.py — Cross-entropy vs predicted probability
- experiments/mle_convergence.py — MLE convergence with sample size

### Changed
- Updated README.md with Stage 3 status, probability usage examples
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.4.0, added probability import
- Updated docs/03_PROJECT_PLAN.md — Stage 3 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 3 completion entry
- Updated docs/14_CHANGELOG.md — This entry

### Design Decisions
- 0 × log(0) = 0 convention for entropy calculations
- Log-sum-exp trick for numerical stability in softmax
- KL divergence explicitly labeled as divergence, not metric
- Central differences as default numerical method (O(h²) accuracy)
- Step-size sensitivity: optimal h ≈ 1e-5 for float64

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.3.0] — 2026-09-16 (Stage 2: Calculus)

### Added
- **calculus/derivatives.py**: Analytical derivatives — quadratic, cubic, polynomial, sin, cos, exp, log, sigmoid, tanh, relu (with domain validation)
- **calculus/finite_differences.py**: Numerical differentiation — forward_difference, central_difference, numerical_derivative, step_size_analysis
- **calculus/partials.py**: Partial derivatives — central and forward difference methods
- **calculus/gradients.py**: Gradient computation — numerical_gradient, gradient_magnitude, gradient_direction
- **calculus/chain_rule.py**: Chain rule — scalar chain_rule_scalar, multivariable chain_rule_multi, demonstrate_chain_rule, neural_network_chain_rule_demo

### Tests
- 102 tests across 6 test files (all passing)
- Unit tests for all analytical derivatives and numerical methods
- Numerical verification tests comparing analytical vs central differences
- Activation function derivative verification (sigmoid, tanh, ReLU)
- Gradient verification against analytical gradients
- Edge case tests (zero step size, negative step size, domain errors)

### Examples
- examples/derivatives.py — Derivative intuition, activation functions, NN connection
- examples/gradients.py — Gradient computation, direction, gradient descent simulation
- examples/chain_rule.py — Chain rule, backpropagation, vanishing gradients

### Experiments
- experiments/step_size_sensitivity.py — Step-size analysis, floating-point limits, forward vs central comparison

### Changed
- Updated README.md with Stage 2 status, calculus usage examples
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.3.0, added calculus import
- Updated docs/11_PROGRESS_LOG.md — Stage 2 completion entry
- Updated docs/14_CHANGELOG.md — This entry

### Design Decisions
- Central differences as default (O(h²) accuracy vs O(h) for forward)
- Step-size analysis reveals optimal h ≈ 1e-5 for float64
- Sigmoid derivative computed as σ(x)(1-σ(x)) — avoids recomputing exp
- ReLU derivative handles scalar inputs via np.asarray conversion
- Chain rule implemented as both scalar and multivariable for pedagogical completeness

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.2.0] — 2026-09-16 (Stage 1: Linear Algebra)

### Added
- **linear_algebra/vectors.py**: Vector operations — creation, validation, addition, subtraction, scalar multiplication, dot product
- **linear_algebra/matrices.py**: Matrix operations — creation, validation, addition, subtraction, scalar multiplication, transpose, identity matrix
- **linear_algebra/operations.py**: Core operations — matrix-vector multiplication, matrix-matrix multiplication (educational triple-loop implementation)
- **linear_algebra/norms.py**: Norms and distance — L1 norm, L2 norm, Euclidean distance
- **linear_algebra/similarity.py**: Cosine similarity with explicit zero-vector handling
- **linear_algebra/geometry.py**: Vector projection, orthogonal decomposition, verification, linear transformations (scaling, rotation, reflection, shear)
- **linear_algebra/eigen.py**: Eigenvalue/eigenvector decomposition (NumPy reference), eigenvector verification, condition number

### Tests
- 127 tests across 8 test files (all passing)
- Unit tests for all operations
- Mathematical property tests (commutativity, associativity, distributivity, triangle inequality)
- Numerical verification tests comparing all implementations against NumPy
- Edge case tests (empty inputs, mismatched dimensions, zero vectors, scalars)

### Examples
- examples/vector_operations.py — Vectors, dot product, norms, cosine similarity, neuron computation
- examples/matrix_operations.py — Matrices, linear layers, batch processing, non-commutativity
- examples/vector_projection.py — Projections, orthogonal decomposition, linear transformations

### Changed
- Updated README.md with Stage 1 status, usage examples, roadmap
- Updated docs/03_PROJECT_PLAN.md — Stage 1 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 1 completion entry
- Updated docs/13_LEARNINGS.md — Stage 1 learnings
- Updated docs/14_CHANGELOG.md — This entry
- Updated pyproject.toml — Added test ignores for mathematical notation (N806, E741, B905)
- Version bumped to 0.2.0

### Design Decisions
- Functions over classes for mathematical operations
- Educational triple-loop implementations alongside NumPy for verification
- Explicit zero-vector error handling for cosine similarity
- NumPy reference for eigenvalue decomposition (from-scratch solver too complex for this stage)
- Projection decomposition: a = proj_b(a) + orth_b(a)

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.1.0] — 2026-09-15 (Stage 0 Complete)

### Added
- Initial repository structure with `src/`, `tests/`, `docs/`, `examples/`, `notebooks/`, `experiments/`, `scripts/`, `results/` directories
- `pyproject.toml` with complete build configuration, metadata, and tool settings (pytest, ruff, mypy, coverage)
- `requirements.txt` with core dependencies (numpy, matplotlib) and documented optional extras (dev, notebook, experiment)
- `LICENSE` with MIT license and third-party dependency license notice
- `.gitignore` with comprehensive Python ignores
- Package initialization: `src/math_for_neural_networks/__init__.py` with version 0.1.0

### Documentation
- **01_PRD.md**: Product Requirements Document — problem, users, goals, non-goals, MVP, future scope, success criteria
- **02_TRD.md**: Technical Requirements Document — Python 3.10+, dependencies, packaging, testing, visualization, numerical approach
- **03_PROJECT_PLAN.md**: Project Plan — 9 stages, detailed milestones, dependencies, verification checkpoints, timeline estimates
- **04_ARCHITECTURE.md**: Architecture — principles, package boundaries, responsibilities, dependency direction, testing/experiment/visualization boundaries
- **05_RESEARCH_PLAN.md**: Research Plan — mathematical topics, progression, sources, questions, influence on implementation
- **06_EXPERIMENT_PLAN.md**: Experiment Plan — categories, lifecycle, hypotheses, baselines, metrics, reproducibility, report format
- **07_TESTING_STRATEGY.md**: Testing Strategy — unit, property-based, numerical verification, edge cases, stability, CI direction
- **08_DEPLOYMENT_PLAN.md**: Deployment Plan — local dev, PyPI, CLI, reproducibility, release strategy, GitHub setup
- **09_SECURITY.md**: Security — dependency hygiene, input validation, supply chain, secrets, malicious contributions
- **10_DECISIONS.md**: Architectural Decisions — 20 decisions with rationale, alternatives, consequences (DEC-001 through DEC-020)
- **11_PROGRESS_LOG.md**: Progress Log — Stage 0 completion entry with artifacts summary
- **12_EXPERIMENT_LOG.md**: Experiment Log — format definition, planned experiments per stage, registry structure
- **13_LEARNINGS.md**: Learnings — intended learning outcomes by mathematical area and stage
- **14_CHANGELOG.md**: This file

### Infrastructure
- Ruff configuration: linting + formatting rules in `pyproject.toml`
- MyPy strict mode configuration in `pyproject.toml`
- Pytest configuration with markers (unit, integration, numerical, slow) in `pyproject.toml`
- Coverage configuration in `pyproject.toml`
- Optional dependency groups: `[dev]`, `[notebook]`, `[experiment]`

### Architectural Decisions (20)
| ID | Decision |
|----|----------|
| DEC-001 | Python >= 3.10 minimum |
| DEC-002 | NumPy as primary numerical backend |
| DEC-003 | Source layout (`src/math_for_neural_networks`) |
| DEC-004 | MIT License |
| DEC-005 | No web UI / dashboard |
| DEC-006 | Verification as cross-cutting module |
| DEC-007 | Strict type checking (mypy strict) |
| DEC-008 | Ruff for linting + formatting |
| DEC-009 | Experiment reports as JSON + Markdown |
| DEC-010 | Tolerance policy for numerical verification |
| DEC-011 | Randomness via numpy.random.Generator (PCG64) |
| DEC-012 | Stage-gated implementation with verification checkpoints |
| DEC-013 | Status markers (PLANNED/IMPLEMENTED/VERIFIED/DEFERRED) |
| DEC-014 | No fabricated results in documentation |
| DEC-015 | Optional dependencies via extras |
| DEC-016 | Experiment configs as YAML/JSON files |
| DEC-017 | Colorblind-safe visualization defaults (viridis/cividis) |
| DEC-018 | No automatic differentiation in core |
| DEC-019 | Float64 default precision |
| DEC-020 | Documentation in `docs/` with numbered prefixes |

---

## [0.1.0] — 2026-09-15 (Stage 0 Complete)

### Added
- Complete Stage 0 foundation: all documentation, structure, tooling, licensing
- Project ready for Stage 1 implementation

### Status Markers
All Stage 0 items: `[DOCUMENTED]` (infrastructure validated: install, import, lint, type-check)

---

## Upcoming Releases (Planned)

### [0.5.0] — Stage 4: Optimization Algorithms (Planned)
**Target:** GD, SGD, momentum, Adam, convergence verification

### [0.6.0] — Stage 5: Neural-Network Mathematics (Planned)
**Target:** Layers, activations, losses, embeddings, attention, normalization

### [0.7.0] — Stage 6: Backpropagation & Training Loops (Planned)
**Target:** Autograd, training loops, XOR/MNIST validation

### [0.8.0] — Stage 7: Experiments & Visualization Suite (Planned)
**Target:** Experiment runner, visualizations, notebooks

### [0.9.0] — Stage 8: PyTorch Comparison/Validation (Planned)
**Target:** Numerical parity on core operations

### [1.0.0] — MVP Complete (Planned)
**Target:** Stable API, full documentation, PyPI release

---

## Legend
- `[PLANNED]` — Designed but not started
- `[IMPLEMENTED]` — Code written, not yet verified
- `[VERIFIED]` — Implemented, tested, numerical verification passed
- `[DEFERRED]` — Moved to future scope