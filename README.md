# Math for Neural Networks

**Status: Stage 1 — Linear Algebra Primitives (Pre-Alpha)**

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

**Stage 1 Implemented:** Linear algebra primitives — vectors, matrices, operations, norms, cosine similarity, projections, linear transformations, eigenvalues/eigenvectors. All operations tested and numerically verified against NumPy.

**Stage 0 Complete:** Project foundation established — documentation, architecture, repository structure, testing strategy, experiment strategy, and distribution plan.

### Implemented (Stage 1)
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

### Planned
- Calculus & numerical differentiation
- Probability & statistics
- Optimization algorithms
- Neural-network mathematics
- Backpropagation & training loops
- PyTorch comparison

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

### Optimization (Planned)
- Gradient descent, learning rate
- Stochastic gradient descent (conceptual)
- Momentum, Adam (conceptual)

### Neural-Network Mathematics (Planned)
- Linear layers, activation functions
- Sigmoid, softmax
- Loss functions
- Backpropagation
- Embeddings, attention, normalization

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
from math_for_neural_networks.linear_algebra import (
    dot_product,
    vector_add,
    l2_norm,
    cosine_similarity,
    matrix_multiply,
    matrix_vector_multiply,
    project_vector,
)

# Vectors
a = [1, 2, 3]
b = [4, 5, 6]
print(dot_product(a, b))          # 32.0
print(vector_add(a, b))           # [5, 7, 9]
print(l2_norm(a))                 # 3.7416...

# Cosine similarity
print(cosine_similarity(a, b))    # 0.9746...

# Matrices
W = [[1, 2], [3, 4]]
x = [5, 6]
print(matrix_vector_multiply(W, x))  # [17, 39]

# Projection
proj = project_vector([3, 4], [1, 0])
print(proj)                       # [3, 0]
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

- **Pre-alpha:** Only linear algebra implemented so far
- **Educational focus:** Not a production ML framework
- **No GPU acceleration:** Pure CPU/NumPy implementation
- **No automatic differentiation:** Manual implementation for learning purposes
- **No web UI:** CLI, notebooks, and scripts only

## Roadmap

| Stage | Focus | Status |
|-------|-------|--------|
| 0 | Project definition & documentation | ✅ Complete |
| 1 | Linear algebra primitives | ✅ Implemented |
| 2 | Calculus & numerical differentiation | 📋 Planned |
| 3 | Probability & statistics foundations | 📋 Planned |
| 4 | Optimization algorithms | 📋 Planned |
| 5 | Neural-network mathematics | 📋 Planned |
| 6 | Backpropagation & training loops | 📋 Planned |
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