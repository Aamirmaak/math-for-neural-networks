# Math for Neural Networks

**Status: Stage 6 — Backpropagation & Training (Pre-Alpha)**

An educational/research-oriented Python toolkit for learning the mathematics behind neural networks through implementation, visualization, and numerical verification.

---

## Problem

Many beginners learn neural networks by memorizing mathematical operations—matrix multiplication, derivatives, gradients, chain rule, probability, loss functions, optimization—without developing an operational understanding of what the mathematics is actually doing.

This project addresses that educational gap by providing a programmable mathematical learning and experimentation toolkit where important mathematical concepts can be:

- **Implemented** from first principles
- **Tested** with rigorous numerical verification
- **Visualized** to build geometric intuition
- **Numerically verified** against analytical results
- **Connected** directly to neural-network operations

## Goals

- Build deep mathematical understanding through implementation rather than memorization
- Create a clean, modular Python architecture for mathematical primitives
- Establish numerical verification as a core principle (analytical vs. numerical comparison)
- Provide reproducible experiments and visualizations
- Prepare the developer for: Mathematics → Neural Networks → Backpropagation → PyTorch → Transformers → LLMs → Agents → RL → Research Engineering → AGI/Frontier AI

## Intended Users

**Primary:** AI/ML learners who want deep mathematical understanding through implementation

**Secondary:**
- ML/DL students
- Aspiring AI engineers
- Researchers learning mathematical foundations
- Developers implementing neural networks from scratch
- Educators who want computational demonstrations

## Current Status

**Stage 6 Implemented:** Backpropagation & Training — scalar autograd engine, gradient checking, affine/activation/loss backpropagation, simple neural network forward/backward, training loop with SGD. All verified with 87 new tests.

**Stage 5 Implemented:** Neural Network Mathematics — affine transformation, activation functions (sigmoid, tanh, ReLU, GELU), softmax, loss functions (MSE, BCE, CCE, logits+CE), attention mathematics, layer normalization. All verified with 137 tests.

**Stage 4 Implemented:** Optimization — gradient descent, SGD with momentum, Adam optimizer, convergence diagnostics, objective functions, numerical verification.

**Stage 3 Implemented:** Probability & Statistics — distributions, entropy, cross-entropy, KL divergence, MLE, softmax, log-sum-exp. All verified.

**Stage 2 Implemented:** Calculus — analytical derivatives, numerical differentiation, partial derivatives, gradients, and chain rule. All verified.

**Stage 1 Implemented:** Linear algebra primitives — vectors, matrices, operations, norms, cosine similarity, projections, linear transformations, eigenvalues/eigenvectors. All verified against NumPy.

**Stage 0 Complete:** Project foundation established — documentation, architecture, repository structure, testing strategy, experiment strategy, and distribution plan.

### Implemented (Stage 5) — Neural Network Mathematics
- Affine transformation (single and batch)
- Activation functions: sigmoid, tanh, ReLU, GELU with derivatives
- Softmax (numerically stable with axis support)
- Loss functions: MSE, binary cross-entropy, categorical cross-entropy, cross-entropy with logits
- Attention mathematics: scaled dot-product attention with masking
- Layer normalization (1D and 2D, gamma/beta parameters)
- Numerical verification of all derivatives against analytical forms
- 137 passing tests (unit + numerical verification)
- Educational examples: affine transformation, activations, classification, attention, normalization
- Experiments: activation comparison, softmax stability, attention scaling

### Implemented (Stage 4) — Optimization
- Gradient descent with configurable learning rate, convergence criteria
- SGD with momentum (velocity accumulation, beta coefficient)
- Adam optimizer (first/second moments, bias correction, epsilon)
- Adam single-step function for educational clarity
- Convergence diagnostics: gradient norm, parameter change, objective change
- Objective functions: quadratic, quartic, sphere, Rosenbrock, Beale, Ackley
- Machine learning objectives: linear regression loss, logistic loss + gradients
- Numerical gradient verification against analytical gradients
- 91 passing tests (unit + numerical verification)
- Educational examples: gradient descent, learning rate, momentum, Adam
- Experiments: LR sensitivity, 2D contour path, momentum vs GD, Adam comparison

### Implemented (Stage 3) — Probability & Statistics
- Probability fundamentals: axioms, complement, union, intersection, conditional, Bayes
- Distributions: Bernoulli, Binomial, Categorical, Uniform, Normal
- Moments: expectation, variance, standard deviation
- Information theory: entropy, cross-entropy, KL divergence
- Likelihood: MLE for Bernoulli, Gaussian, categorical models
- Numerical stability: log-sum-exp, softmax, log-softmax, sigmoid
- 149 passing tests (unit + numerical verification)
- Educational examples: entropy, cross-entropy, MLE
- Experiments: entropy vs concentration, cross-entropy loss, MLE convergence

### Implemented (Stage 2) — Calculus
- Analytical derivatives: quadratic, cubic, polynomial, sin, cos, exp, log
- Neural network activations: sigmoid, tanh, ReLU (with derivatives)
- Finite differences: forward, central, numerical_derivative
- Step-size sensitivity analysis
- Partial derivatives (central and forward difference)
- Numerical gradient computation
- Gradient magnitude and direction
- Chain rule: scalar, multivariable, neural network demo
- 102 passing tests (unit + numerical verification)
- Educational examples and step-size experiment

### Implemented (Stage 1) — Linear Algebra
- Vector operations: addition, subtraction, scalar multiplication, dot product
- Matrix operations: addition, subtraction, scalar multiplication, transpose, identity
- Matrix multiplication (educational triple-loop implementation)
- Matrix-vector multiplication
- Norms: L1 (Manhattan), L2 (Euclidean), Euclidean distance
- Cosine similarity with zero-vector handling
- Vector projection and orthogonal decomposition
- Linear transformations: scaling, rotation, reflection, shear
- Eigenvalue/eigenvector decomposition and verification
- Condition number computation
- Numerical verification against NumPy for all operations
- 127 passing tests (unit + numerical verification)

### Implemented (Stage 6) — Backpropagation & Training
- Scalar autograd engine (Value class with backward pass)
- Computational graph construction and topological ordering
- Gradient checking (analytical vs numerical finite differences)
- Affine layer backpropagation (dX, dW, db for Z=XW^T+b)
- Activation backpropagation (sigmoid, tanh, ReLU)
- Loss backpropagation (MSE, softmax+CE identity, sigmoid+BCE identity)
- Simple 2-layer neural network forward/backward
- Training loop with SGD optimizer
- Numerical gradient verification for all components
- 87 passing tests (unit + numerical verification)
- Educational examples: computational graph, manual backprop, autograd demo, training demo
- Experiments: manual vs numerical, toy training, learning rate effect, gradient norm, activation gradients, vanishing gradient, softmax+CE identity

## Planned Capabilities (MVP Roadmap)

### Linear Algebra (Planned)
- Scalars, vectors, matrices
- Vector addition, dot product, matrix multiplication
- Transpose, norms, cosine similarity

### Calculus (Planned)
- Derivative, partial derivative, gradient
- Chain rule
- Numerical differentiation
- Selected analytical derivatives

### Probability & Statistics (Planned)
- Probability, conditional probability
- Expectation, variance, distributions
- Log probability, entropy, cross-entropy
- Likelihood, maximum likelihood

### Optimization (Implemented)
- Gradient descent with learning rate
- SGD with momentum
- Adam optimizer with bias correction

### Neural-Network Mathematics (Implemented)
- Linear layers, activation functions (sigmoid, tanh, ReLU, GELU)
- Softmax (numerically stable)
- Loss functions (MSE, binary CE, categorical CE, logits+CE)
- Attention mathematics (scaled dot-product, masking)
- Layer normalization

### Backpropagation & Training (Implemented)
- Scalar autograd engine (Value class with backward pass)
- Gradient checking (analytical vs numerical)
- Affine layer backpropagation
- Activation backpropagation (sigmoid, tanh, ReLU)
- Loss backpropagation (MSE, softmax+CE, sigmoid+BCE)
- Simple neural network forward/backward
- Training loop with SGD

## Installation

```bash
# Clone the repository
git clone https://github.com/Aamirmaak/math-for-neural-networks.git
cd math-for-neural-networks

# Install in development mode with core dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev]"

# Or with notebook support
pip install -e ".[notebook]"

# Or with experiment dependencies
pip install -e ".[experiment]"
```

## Usage

```python
# Linear Algebra
from math_for_neural_networks.linear_algebra import (
    dot_product,
    vector_add,
    l2_norm,
    cosine_similarity,
    matrix_multiply,
    matrix_vector_multiply,
    project_vector,
)

a = [1, 2, 3]
b = [4, 5, 6]
print(dot_product(a, b))          # 32.0
print(vector_add(a, b))           # [5, 7, 9]
print(l2_norm(a))                 # 3.7416...
print(cosine_similarity(a, b))    # 0.9746...

# Calculus
from math_for_neural_networks.calculus import (
    quadratic, quadratic_derivative,
    central_difference, numerical_gradient,
    sigmoid, sigmoid_derivative,
    chain_rule_scalar,
)

# Analytical derivatives
print(quadratic(3.0))             # 9.0
print(quadratic_derivative(3.0))  # 6.0

# Numerical verification
numerical = central_difference(quadratic, 3.0, h=1e-5)
print(numerical)                  # ~6.0

# Gradient computation
def f(point):
    return point[0]**2 + point[1]**2

grad = numerical_gradient(f, [3.0, 4.0])
print(grad)                       # [6.0, 8.0]

# Neural network activation derivatives
print(sigmoid(0.0))               # 0.5
print(sigmoid_derivative(0.0))    # 0.25
```

## Repository Structure

```
math-for-neural-networks/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── docs/
├── src/
│   └── math_for_neural_networks/
├── tests/
├── examples/
├── notebooks/
├── experiments/
├── scripts/
└── results/
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest

# Run linting
ruff check .
ruff format .

# Type checking
mypy src/
```

## Testing Strategy

- **Framework:** pytest
- **Principles:** Unit tests, property-based mathematical testing, numerical verification (analytical vs. finite-difference), edge cases, stability, regression testing
- **Markers:** `unit`, `integration`, `numerical`, `slow`

## Experiment Strategy

Structured experiment lifecycle:
```
Hypothesis → Method → Baseline → Experiment → Result → Analysis → Conclusion
```

All experiments must clearly separate hypothesis, methodology, measured results, interpretation, and limitations. No fabricated results.

## Visualization Strategy

Visualizations exist only when they improve mathematical understanding. Tools: Matplotlib. No decorative charts.

## Architecture Principles

- Clean separation: mathematical primitives → higher-level operations → neural-network mathematics → numerical verification → visualization → experiments → CLI/interfaces → tests
- NumPy as primary numerical dependency
- PyTorch only for comparison/validation (future)
- Modular, inspectable, understandable implementations
- No deep-learning frameworks for core mathematics

## License

MIT License — see [LICENSE](LICENSE) for details.

**Important:** Third-party dependencies have their own licenses (NumPy: BSD-3, Matplotlib: PSF, etc.). Verify compatibility before distribution. This notice is not legal advice.

## Limitations

- **Pre-alpha:** Stages 1-6 implemented; advanced training and experiments pending
- **Educational focus:** Not a production ML framework
- **No GPU acceleration:** Pure CPU/NumPy implementation
- **No automatic differentiation:** Manual implementation for learning purposes
- **No web UI:** CLI, notebooks, and scripts only

## Roadmap

| Stage | Focus | Status |
|-------|-------|--------|
| 0 | Project definition & documentation | ✅ Complete |
| 1 | Linear algebra primitives | ✅ Implemented |
| 2 | Calculus & numerical differentiation | ✅ Implemented |
| 3 | Probability & statistics foundations | ✅ Implemented |
| 4 | Optimization algorithms | ✅ Implemented |
| 5 | Neural-network mathematics | ✅ Implemented |
| 6 | Backpropagation & training loops | ✅ Implemented |
| 7 | Experiments & visualization suite | 📋 Planned |
| 8 | PyTorch comparison/validation | 📋 Planned |

Future scope (deferred): tensor decompositions, eigenvalues/eigenvectors, advanced optimization, attention mathematics, transformer demonstrations, interactive visualizations, benchmark suites.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (to be created) for guidelines. The project welcomes:
- Mathematical corrections and improvements
- Numerical verification enhancements
- Educational examples and notebooks
- Documentation improvements
- Test coverage expansions

## Documentation

- [PRD](docs/01_PRD.md) — Product Requirements Document
- [TRD](docs/02_TRD.md) — Technical Requirements Document
- [Project Plan](docs/03_PROJECT_PLAN.md) — Stages, milestones, dependencies
- [Architecture](docs/04_ARCHITECTURE.md) — Architectural principles and boundaries
- [Research Plan](docs/05_RESEARCH_PLAN.md) — Mathematical topics and progression
- [Experiment Plan](docs/06_EXPERIMENT_PLAN.md) — Experiment categories and methodology
- [Testing Strategy](docs/07_TESTING_STRATEGY.md) — Testing approach and standards
- [Deployment Plan](docs/08_DEPLOYMENT_PLAN.md) — Distribution and release strategy
- [Security](docs/09_SECURITY.md) — Security considerations
- [Decisions](docs/10_DECISIONS.md) — Architectural decisions and rationale
- [Progress Log](docs/11_PROGRESS_LOG.md) — Stage-by-stage progress
- [Experiment Log](docs/12_EXPERIMENT_LOG.md) — Experiment record format
- [Learnings](docs/13_LEARNINGS.md) — Intended learning outcomes
- [Changelog](docs/14_CHANGELOG.md) — Version history