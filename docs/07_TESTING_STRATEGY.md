# 07_TESTING_STRATEGY.md — Testing Strategy

## Testing Philosophy

Tests are **executable specifications** of mathematical behavior. They document expected behavior, catch regressions, and enable confident refactoring. Every mathematical function must have tests before or alongside implementation.

## Test Framework

- **Primary:** pytest
- **Configuration:** `pyproject.toml` `[tool.pytest.ini_options]`
- **Markers:** `unit`, `integration`, `numerical`, `slow`
- **Coverage:** Target >90% for implemented code (branch coverage)

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures, seeds
├── test_linear_algebra/
│   ├── test_scalars.py
│   ├── test_vectors.py
│   ├── test_matrices.py
│   ├── test_norms.py
│   └── test_verification.py       # Analytical vs. numerical
├── test_calculus/
│   ├── test_finite_differences.py
│   ├── test_derivatives.py
│   ├── test_gradients.py
│   ├── test_chain_rule.py
│   ├── test_analytical.py
│   └── test_verification.py
├── test_probability/
│   ├── test_distributions.py
│   ├── test_expectations.py
│   ├── test_entropy.py
│   ├── test_likelihood.py
│   └── test_verification.py
├── test_optimization/
│   ├── test_gradient_descent.py
│   ├── test_sgd.py
│   ├── test_momentum.py
│   ├── test_adam.py
│   └── test_verification.py
├── test_neural_networks/
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_attention.py
│   ├── test_containers.py
│   └── test_verification.py
├── test_verification/             # Cross-cutting verification utilities
│   ├── test_numerical.py
│   ├── test_tolerances.py
│   └── test_comparison.py
├── test_visualization/            # Smoke tests for plotting
│   ├── test_functions.py
│   └── test_gradients.py
├── test_experiments/              # Experiment runner tests
│   ├── test_runner.py
│   └── test_config.py
└── integration/
    ├── test_la_calculus.py        # Vector + gradient
    ├── test_calc_opt.py           # Gradient + optimizer
    ├── test_nn_training.py        # Full training loop
    └── test_pytorch_parity.py     # Comparison tests (optional)
```

## Test Categories

### 1. Unit Tests (Marker: `unit`)

**Scope:** Single function/class in isolation

**Patterns:**
- Known input → expected output (exact or with tolerance)
- Edge cases: empty, zero, negative, extreme values
- Invalid inputs: wrong shapes, types, NaN/Inf
- Mathematical properties: commutativity, associativity, distributivity

**Example:**
```python
def test_vector_addition_commutativity():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    assert (v1 + v2).data == pytest.approx((v2 + v1).data)


def test_matrix_multiply_shape_error():
    A = Matrix([[1, 2], [3, 4]])  # 2x2
    B = Matrix([[1, 2, 3]])  # 1x3
    with pytest.raises(ValueError, match="shape"):
        A @ B
```

### 2. Property-Based Tests (Marker: `unit`)

**Scope:** Mathematical properties that hold for all valid inputs

**Tool:** `hypothesis` (add to dev deps when needed)

**Properties to test:**
- `dot(v, w) == dot(w, v)` (commutativity)
- `norm(v) >= 0` (non-negativity)
- `norm(v) == 0 iff v == 0` (definiteness)
- `norm(a * v) == |a| * norm(v)` (homogeneity)
- `norm(v + w) <= norm(v) + norm(w)` (triangle inequality)
- `derivative(f, x) ≈ (f(x+h) - f(x-h)) / (2h)` (finite difference)

### 3. Numerical Verification Tests (Marker: `numerical`)

**Scope:** Analytical implementation vs. numerical approximation

**Core Principle:** For every function `f` with analytical derivative `f'`:
```
numerical = finite_difference(f, x)
analytical = f'(x)
assert relative_error(numerical, analytical) < TOLERANCE
```

**Tolerances (float64):**
- Polynomials, smooth functions: `1e-10` (central), `1e-14` (complex-step)
- Sigmoid, tanh, softmax: `1e-7` (central)
- ReLU (non-smooth at 0): `1e-7` away from 0, skip at 0
- Matrix operations vs. NumPy: `1e-12`

**Test Template:**
```python
@pytest.mark.numerical
def test_sigmoid_gradient_verification():
    analytical = sigmoid_gradient
    numerical = central_difference(sigmoid, h=1e-6)
    x_vals = np.linspace(-10, 10, 1000)
    for x in x_vals:
        a = analytical(x)
        n = numerical(x)
        rel_err = abs(a - n) / max(abs(a), abs(n), 1e-12)
        assert rel_err < 1e-7, f"x={x}: analytical={a}, numerical={n}, rel_err={rel_err}"
```

### 4. Edge Case Tests (Marker: `unit`)

**Required edge cases per function:**
- Zero inputs (vectors, matrices, scalars)
- Single-element inputs
- Very large/small values (overflow/underflow)
- NaN/Inf inputs (should raise or propagate)
- Non-contiguous arrays (if applicable)
- Wrong dtypes (should cast or raise)

### 5. Stability Tests (Marker: `numerical`)

**Focus:** Numerical stability, not just correctness
- Condition number sensitivity
- Catastrophic cancellation detection
- Gradient explosion/vanishing on deep compositions
- Accumulation error in iterative algorithms

### 6. Regression Tests (Marker: `unit`)

**Trigger:** Bug fixes, numerical issues discovered
- Minimal reproduction case
- Added to prevent re-occurrence
- Linked to issue/PR in comments

### 7. Integration Tests (Marker: `integration`)

**Scope:** Cross-module workflows
- Linear algebra → Calculus (gradient of matrix function)
- Calculus → Optimization (GD on function)
- Neural networks → Training loop (forward + backward + step)
- Full pipeline: data → model → loss → backward → update

## Test Data Management

- **Fixtures:** `conftest.py` provides seeded generators, common test vectors/matrices
- **No external test data files** — Generate programmatically with fixed seeds
- **Parametrized tests:** Use `@pytest.mark.parametrize` for multiple input sets

## Tolerance Policy

| Comparison | Default Tolerance | Rationale |
|------------|-------------------|-----------|
| Analytical vs. Central Difference | `1e-7` | O(h²) error, floating point |
| Analytical vs. Complex Step | `1e-14` | No subtraction cancellation |
| Custom vs. NumPy (LA) | `1e-12` | Same algorithm, diff implementation |
| Custom vs. PyTorch (NN) | `1e-5` | Different operation order, fusion |
| Monte Carlo expectation | `1e-2` / `1e-3` | Statistical variance |

**Override:** Specific tests may use tighter/looser tolerances with justification in comment.

## CI Direction (Future)

When CI is enabled (GitHub Actions):
```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Format check
        run: ruff format . --check
      - name: Type check
        run: mypy src/
      - name: Unit tests
        run: pytest -m "not slow and not numerical" -x
      - name: Numerical tests
        run: pytest -m "numerical" -x
      - name: Integration tests
        run: pytest -m "integration" -x
```

## Running Tests Locally

```bash
# All tests
pytest

# Unit only (fast)
pytest -m "unit"

# Numerical verification (may be slower)
pytest -m "numerical"

# Integration
pytest -m "integration"

# With coverage
pytest --cov=src/math_for_neural_networks --cov-report=term-missing

# Verbose
pytest -xvs
```

## Test Naming Convention

```
test_<function>_<scenario>_<expected>
test_dot_product_orthogonal_vectors_returns_zero
test_gradient_descent_converges_on_convex_quadratic
test_sigmoid_gradient_matches_central_difference
```

## What NOT to Test (Yet)

- Performance benchmarks (separate concern)
- Property tests requiring `hypothesis` (add when needed)
- Visual inspection tests (manual)
- PyTorch comparison (Stage 8, optional dependency)