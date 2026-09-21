# Math for Neural Networks

An educational Python toolkit for learning the mathematics behind neural networks through implementation, testing, and numerical verification.

## Why This Project Exists

Many learners memorize matrix multiplication, derivatives, gradients, chain rule, probability, and loss functions without understanding what the mathematics is actually doing numerically.

This project turns those concepts into executable, testable, visualizable code. You implement each concept from first principles, verify it numerically against analytical results, and connect it directly to neural network operations.

## What It Covers

| Area | What You Learn |
|------|---------------|
| **Linear Algebra** | Vectors, matrices, dot products, norms, cosine similarity, projections, eigenvalues |
| **Calculus** | Derivatives, partial derivatives, gradients, chain rule, finite differences |
| **Probability** | Distributions, entropy, cross-entropy, KL divergence, MLE |
| **Optimization** | Gradient descent, momentum, Adam, convergence diagnostics |
| **Neural Networks** | Affine layers, activations, softmax, loss functions, attention, normalization |
| **Backpropagation** | Autograd, computational graphs, gradient checking, training loops |
| **PyTorch Parity** | Side-by-side comparison with PyTorch (optional dependency) |

## Mathematical Roadmap

```
Linear Algebra
  ├── vectors, matrices, dot products, matrix multiplication
  ├── norms, cosine similarity, projections
  └── eigenvalues/eigenvectors

Calculus
  ├── derivatives, partial derivatives, gradients
  ├── directional derivatives, chain rule
  └── finite differences (numerical verification)

Probability
  ├── random variables, distributions (Bernoulli, Normal, Categorical)
  ├── expectation, variance, entropy
  └── cross-entropy, KL divergence, likelihood, MLE

Optimization
  ├── gradient descent, SGD
  ├── momentum, Adam
  └── convergence diagnostics, objective functions

Neural Network Mathematics
  ├── affine transformations, activation functions
  ├── softmax (numerically stable)
  ├── loss functions (MSE, BCE, CCE, logits+CE)
  ├── embeddings, attention, layer normalization
  └── backpropagation, autograd, training loops

Framework Parity
  └── PyTorch comparison (optional)
```

## Neural Network Connections

Every module connects directly to neural network operations:

- **Dot product** → Neuron computes `z = w · x + b`
- **Matrix multiplication** → Batch processing, linear layers
- **Softmax** → Classification probability output
- **Cross-entropy** → Classification loss function
- **Gradient descent** → Training optimizer
- **Chain rule** → Backpropagation
- **Attention** → Transformer self-attention mechanism

## Installation

```bash
git clone https://github.com/Aamirmaak/math-for-neural-networks.git
cd math-for-neural-networks

# Core (NumPy + Matplotlib only)
pip install -e .

# With development tools
pip install -e ".[dev]"

# With notebook support
pip install -e ".[notebook]"

# With experiment dependencies
pip install -e ".[experiment]"

# With PyTorch for comparison tests
pip install -e ".[comparison]"
```

**Requirements:** Python >= 3.10

## Quick Start

```python
import numpy as np
from math_for_neural_networks.linear_algebra import dot_product, l2_norm, cosine_similarity
from math_for_neural_networks.calculus import central_difference, quadratic_derivative
from math_for_neural_networks.neural_networks.activations import sigmoid

# Linear algebra: a neuron computes dot product + bias
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, 0.3, 0.2])
z = dot_product(w, x) + 0.1
print(f"Neuron output: {z:.4f}")  # 1.7000

# Calculus: verify derivative numerically
numerical = central_difference(lambda t: t**2, 3.0, h=1e-5)
analytical = quadratic_derivative(3.0)
print(f"Numerical: {numerical:.6f}, Analytical: {analytical:.6f}")  # ~6.0

# Activation function
print(f"sigmoid(0) = {sigmoid(0.0)}")  # 0.5
```

## Example

```python
from math_for_neural_networks.neural_networks.training import init_params, forward, backward, sgd_step, mean_squared_error

# Simple 2-layer network
params = init_params(input_dim=3, hidden_dim=4, output_dim=2, seed=42)
x = np.random.randn(8, 3)
y = np.random.randn(8, 2)

# Forward pass
y_pred, cache = forward(x, params, "sigmoid")
loss = mean_squared_error(y, y_pred)

# Backward pass
grads = backward(y, y_pred, params, cache, "sigmoid", "mse")

# Update parameters
sgd_step(params, grads, learning_rate=0.01)
```

## Testing

```bash
# Run all 814 tests
pytest

# Run with coverage
pytest --cov=math_for_neural_networks --cov-report=term-missing

# Run specific test categories
pytest -m unit
pytest -m numerical
pytest -m integration

# Run PyTorch comparison tests (requires PyTorch)
pytest tests/comparison/ -v
```

## Experiments

18 structured experiments covering all mathematical topics:

```bash
# Run all experiments
for f in experiments/[0-1]*.py; do python "$f"; done

# Run a specific experiment
python experiments/05_learning_rate.py
```

## Numerical Verification

Every mathematical function with derivatives includes numerical verification:

```python
# Analytical vs. finite-difference comparison
from math_for_neural_networks.calculus import central_difference
from math_for_neural_networks.neural_networks.activations import sigmoid, sigmoid_derivative

x = 0.5
analytical = sigmoid_derivative(x)
numerical = central_difference(sigmoid, x, h=1e-7)
print(f"Error: {abs(analytical - numerical):.2e}")  # < 1e-6
```

## PyTorch Comparison

Optional side-by-side verification against PyTorch:

```bash
pip install -e ".[comparison]"
pytest tests/comparison/ -v
```

Compares: affine transforms, activations (sigmoid, tanh, ReLU, GELU), softmax, losses (MSE, BCE, CCE), gradients, small network forward/backward, and optimizer updates (SGD, Momentum, Adam).

## Repository Structure

```
math-for-neural-networks/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
├── docs/
│   ├── 01_PRD.md              # Product Requirements
│   ├── 02_TRD.md              # Technical Requirements
│   ├── 03_PROJECT_PLAN.md     # Stages & Milestones
│   ├── 04_ARCHITECTURE.md     # Architecture Principles
│   ├── 05_RESEARCH_PLAN.md    # Mathematical Topics
│   ├── 06_EXPERIMENT_PLAN.md  # Experiment Strategy
│   ├── 07_TESTING_STRATEGY.md # Testing Approach
│   ├── 08_DEPLOYMENT_PLAN.md  # Distribution Plan
│   ├── 09_SECURITY.md         # Security Considerations
│   ├── 10_DECISIONS.md        # Architectural Decisions
│   ├── 11_PROGRESS_LOG.md     # Stage Progress
│   ├── 12_EXPERIMENT_LOG.md   # Experiment Records
│   ├── 13_LEARNINGS.md        # Learning Outcomes
│   ├── 14_CHANGELOG.md        # Version History
│   ├── 15_PYTORCH_COMPARISON.md
│   ├── 16_EXPERIMENT_REGISTRY.md
│   ├── FINAL_VALIDATION_REPORT.md
│   └── RELEASE_CHECKLIST.md
├── src/
│   └── math_for_neural_networks/
│       ├── linear_algebra/    # Vectors, matrices, norms, similarity
│       ├── calculus/          # Derivatives, gradients, chain rule
│       ├── probability/       # Distributions, entropy, likelihood
│       ├── optimization/      # GD, momentum, Adam
│       ├── neural_networks/   # Activations, losses, attention, training
│       └── autograd/          # Scalar autograd engine
├── tests/
├── examples/
├── experiments/
└── results/
```

## Documentation

- [PRD](docs/01_PRD.md) — Product Requirements
- [TRD](docs/02_TRD.md) — Technical Requirements
- [Project Plan](docs/03_PROJECT_PLAN.md) — Stages & Milestones
- [Architecture](docs/04_ARCHITECTURE.md) — Architecture Principles
- [Testing Strategy](docs/07_TESTING_STRATEGY.md) — Testing Approach
- [Changelog](docs/14_CHANGELOG.md) — Version History
- [Final Validation](docs/FINAL_VALIDATION_REPORT.md) — Verification Summary
- [Release Checklist](docs/RELEASE_CHECKLIST.md) — Release Process

## Project Status

| Area | Status |
|------|--------|
| Linear Algebra | Implemented, tested |
| Calculus | Implemented, tested |
| Probability | Implemented, tested |
| Optimization | Implemented, tested |
| Neural Network Math | Implemented, tested |
| Backpropagation | Implemented, tested |
| Experiments | 18 executed |
| PyTorch Comparison | Implemented, tested (optional) |
| Mathematical Properties | 62 tests |
| Total Tests | 814 passing |
| Coverage | 91% |
| CI | GitHub Actions |
| PyPI Release | Not released |

## Limitations

- **Pre-alpha:** No public release yet
- **Educational focus:** Not a production ML framework
- **No GPU:** Pure CPU/NumPy implementation
- **No autograd:** Manual backpropagation for learning
- **No web UI:** CLI, notebooks, and scripts only

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key expectations:
- All new functionality must include tests
- Numerical verification required for mathematical functions
- Do not claim experiment results without execution
- Do not introduce unnecessary dependencies

## License

MIT License — see [LICENSE](LICENSE).

Third-party dependencies have their own licenses (NumPy: BSD-3, Matplotlib: PSF, etc.). Verify compatibility before distribution.

## How to Cite

If this toolkit is useful in your research or learning:

```bibtex
@software{math_for_neural_networks,
  title = {Math for Neural Networks},
  url = {https://github.com/Aamirmaak/math-for-neural-networks},
  year = {2026},
  license = {MIT}
}
```
