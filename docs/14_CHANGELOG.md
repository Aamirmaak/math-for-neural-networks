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

## [0.8.0] — 2026-09-16 (Stage 7: Integrated Experiments & Visualization)

### Added
- **experiments/experiment_utils.py**: Shared experiment utilities (setup_seed, ExperimentResult, save_fig, print_table, ensure_results_dir)
- **experiments/01_vector_geometry.py**: Vector geometry relationships (angle, dot product, cosine similarity, Euclidean distance)
- **experiments/02_matrix_transformations.py**: 2D linear transformations (scaling, rotation, reflection, shear)
- **experiments/03_derivative_accuracy.py**: Forward vs central finite differences across step sizes
- **experiments/04_gradient_field.py**: Contour lines and gradient vectors for f(x,y) = x^2+y^2
- **experiments/05_learning_rate.py**: Gradient descent with different learning rates
- **experiments/06_optimizer_comparison.py**: GD vs Momentum vs Adam on Rosenbrock function
- **experiments/07_activation_functions.py**: Sigmoid, tanh, ReLU, GELU comparison
- **experiments/08_softmax_stability.py**: Naive vs stable softmax numerical stability
- **experiments/09_cross_entropy.py**: Cross-entropy loss vs predicted probability
- **experiments/10_backpropagation.py**: Computational graph forward + backward pass
- **experiments/11_gradient_checking.py**: Analytical vs numerical gradient verification
- **experiments/12_toy_training.py**: Small neural network training on toy dataset
- **experiments/13_gradient_flow.py**: Gradient magnitudes through layers during training
- **experiments/14_vanishing_exploding.py**: Gradient magnitude through sigmoid layers
- **experiments/15_embedding_geometry.py**: Toy embedding vectors, similarities, distances
- **experiments/16_attention.py**: Scaled dot-product attention step by step
- **experiments/17_attention_scaling.py**: Effect of 1/sqrt(d_k) on attention weights
- **experiments/18_integrated_demo.py**: Complete training pipeline with intermediate math

### Tests
- 10 new tests (703 total)
- Experiment utility tests (seed, result, directory, table, figure)
- Integration test: complete training pipeline (forward, loss, backward, update)

### Changed
- Updated README.md with Stage 7 status
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.8.0
- Updated docs/03_PROJECT_PLAN.md — Stage 7 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 7 completion entry
- Updated docs/14_CHANGELOG.md — This entry
- Updated pyproject.toml — Added per-file-ignores for experiments directory

### Design Decisions
- Flat experiment structure (individual scripts) over nested directories
- ExperimentResult dataclass for structured results
- Reproducible seeds for all stochastic experiments
- Matplotlib Agg backend for headless environments
- Per-file ruff ignores for experiments (T201 print statements allowed)
- Integration test covers complete mathematical pipeline

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.7.0] — 2026-09-16 (Stage 6: Backpropagation & Training)

### Added
- **autograd/__init__.py**: Public API exports for autograd module
- **autograd/value.py**: Scalar Value class with backward() for automatic differentiation, get_topo_order() for graph inspection
- **neural_networks/backprop.py**: gradient_check (analytical vs numerical), gradient_check_scalar, affine_backward (dX, dW, db), sigmoid_backward, relu_backward, tanh_backward, mse_backward, softmax_backward, sigmoid_bce_backward
- **neural_networks/training.py**: NetworkParams dataclass, TrainingMetrics dataclass, init_params, forward (2-layer network), backward (manual backprop), sgd_step, train (full training loop), compute_numerical_gradients

### Tests
- 87 tests across 4 test files (all passing)
- 33 autograd Value tests (ops, gradients, accumulation, topo sort, edge cases)
- 12 numerical verification tests (autograd vs numerical differentiation)
- 19 backprop tests (gradient_check, affine_backward, activation backward, loss backward, softmax+CE, sigmoid+BCE)
- 23 training tests (params, forward, backward, sgd, numerical gradients, training loop)

### Examples
- examples/computational_graph.py — Graph construction, chain rule, activation, gradient accumulation, topo sort
- examples/manual_backpropagation.py — Single neuron, numerical verification, two-layer network, matrix backprop
- examples/autograd_demo.py — Basic ops, single neuron, gradient accumulation, deep network, loss landscape
- examples/training_demo.py — Forward/backward, training loop, activation comparison, LR effect, gradient norm

### Experiments
- experiments/manual_vs_numerical.py — Manual backprop vs numerical gradients
- experiments/toy_network_training.py — Toy binary classification training
- experiments/learning_rate_effect.py — LR comparison
- experiments/gradient_norm_training.py — Gradient norm during training
- experiments/activation_gradients.py — Activation gradient behavior
- experiments/vanishing_gradient.py — Vanishing gradient intuition
- experiments/softmax_ce_gradient.py — Softmax + CE gradient identity

### Changed
- Updated README.md with Stage 6 status, backpropagation usage examples
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.7.0, added autograd import
- Updated docs/03_PROJECT_PLAN.md — Stage 6 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 6 completion entry
- Updated docs/14_CHANGELOG.md — This entry

### Design Decisions
- Scalar autograd for educational clarity (understanding computational graphs)
- Manual backpropagation in NumPy arrays for practical neural network training
- Gradient checking via central differences for verification
- Softmax+CE gradient identity (p-y) for simplified backpropagation
- Sigmoid+BCE gradient identity with 1/n normalization
- Functions over classes for all implementations (except dataclasses for state)

### Key Mathematical Identities
- Softmax + CE: dL/dz = softmax(z) - y
- Sigmoid + BCE: dL/dz = (1/n)(sigma(z) - y)
- Affine: dL/dX = dL/dZ @ W, dL/dW = dL/dZ^T @ X, dL/db = sum(dL/dZ)
- Sigmoid: da/dx = a(1-a)
- Tanh: da/dx = 1-a^2
- ReLU: da/dx = 1 if x>0 else 0
- MSE: dL/dy_pred = (2/n)(y_pred - y_true)

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.6.0] — 2026-09-16 (Stage 5: Neural Network Mathematics)

### Added
- **neural_networks/layers.py**: affine_transform (single and batch, shape validation)
- **neural_networks/activations.py**: sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, gelu, gelu_derivative (all numerically stable)
- **neural_networks/losses.py**: mean_squared_error, binary_cross_entropy, categorical_cross_entropy, cross_entropy_with_logits
- **neural_networks/attention.py**: softmax (axis-stable), scaled_dot_product_attention (with optional mask), attention_weights
- **neural_networks/normalization.py**: layer_norm (1D + 2D, gamma/beta), layer_norm_stats

### Tests
- 137 tests across 6 test files (all passing)
- Unit tests for all neural network operations
- Numerical verification tests comparing analytical vs numerical derivatives
- Edge case tests (empty inputs, mismatched dimensions, extreme values)
- Property tests (softmax sums to 1, attention weights sum to 1, normalization zero mean)

### Examples
- examples/affine_transformation.py — Affine transform, bias, batching, NN connection
- examples/activation_functions.py — Sigmoid, tanh, ReLU, GELU comparison
- examples/classification_example.py — Binary/multi-class classification forward pass
- examples/attention_example.py — Self-attention, shape flow, scaling, masking
- examples/normalization_example.py — Layer norm, gamma/beta, Transformers connection

### Experiments
- experiments/activation_comparison.py — Activation function comparison (ranges, derivatives, saturation)
- experiments/softmax_stability.py — Naive vs stable softmax numerical stability
- experiments/attention_scaling.py — Effect of 1/sqrt(d_k) scaling on attention weights

### Changed
- Updated README.md with Stage 5 status, neural network usage examples
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.6.0, added neural_networks import
- Updated docs/03_PROJECT_PLAN.md — Stage 5 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 5 completion entry
- Updated docs/14_CHANGELOG.md — This entry

### Design Decisions
- Layer normalization uses population variance (ddof=0), not sample variance (ddof=1)
- Cross-entropy with logits for numerical stability (avoids softmax->log->CE pipeline)
- Attention scaling by 1/sqrt(d_k) prevents softmax saturation
- Functions over classes for all implementations
- GELU implemented via tanh approximation (standard formula)

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

---

## [0.5.0] — 2026-09-16 (Stage 4: Optimization Algorithms)

### Added
- **optimization/diagnostics.py**: OptResult dataclass, check_convergence, has_finite_values
- **optimization/objectives.py**: Quadratic, quartic, sphere, Rosenbrock, Beale, Ackley objective functions with analytical gradients; linear regression loss, logistic loss with gradients
- **optimization/gradient_descent.py**: Gradient descent with configurable learning rate, convergence criteria (gradient norm, parameter change, objective change), history tracking
- **optimization/momentum.py**: SGD with momentum (velocity accumulation, beta coefficient)
- **optimization/adam.py**: Adam optimizer (first/second moments, bias correction, epsilon) + adam_step function for single-step updates

### Tests
- 91 tests across 6 test files (all passing)
- Unit tests for all objective functions and their gradients
- Gradient descent convergence tests with various learning rates
- Momentum tests with different beta values
- Adam tests including bias correction verification
- Numerical gradient verification against analytical forms
- Edge case tests (zero gradients, extreme parameters, invalid inputs)

### Examples
- examples/gradient_descent.py — 1D/2D GD, learning rate comparison, module usage
- examples/gradient_descent_2d.py — Contour visualization, Rosenbrock, starting point effect
- examples/learning_rate.py — LR grid search, scheduling concept, curvature interaction
- examples/momentum.py — Narrow valley problem, momentum solution, beta comparison
- examples/adam.py — Adam step-by-step, bias correction, three-way optimizer comparison

### Experiments
- experiments/lr_sensitivity.py — Learning rate sensitivity analysis
- experiments/contour_path.py — 2D GD path visualization (quadratic + Rosenbrock)
- experiments/momentum_vs_gd.py — Momentum vs GD comparison with convergence history
- experiments/optimizer_comparison.py — Adam vs Momentum vs GD on quadratic and Rosenbrock
- experiments/init_effect.py — Initialization effect on convergence (convex vs non-convex)

### Changed
- Updated README.md with Stage 4 status, optimization usage examples
- Updated src/math_for_neural_networks/__init__.py — Version bump to 0.5.0, added optimization import
- Updated docs/03_PROJECT_PLAN.md — Stage 4 marked Implemented
- Updated docs/11_PROGRESS_LOG.md — Stage 4 completion entry
- Updated docs/14_CHANGELOG.md — This entry

### Design Decisions
- Adam convergence check AFTER update step (not before) to avoid false convergence at t=0
- Scalar objective functions use np.asarray to handle both scalar and array inputs
- Mathematical notation (X, dLdw) requires per-file ruff ignores (N803, N806)
- Functions over classes for optimizer implementations
- Convergence criteria: gradient norm, parameter change, objective change (all configurable)
- Bias correction: m_hat = m/(1-beta1^t), v_hat = v/(1-beta2^t)

### Status: [IMPLEMENTED] [TESTED] [VERIFIED]

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