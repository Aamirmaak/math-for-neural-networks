# 04_ARCHITECTURE.md — Architecture Document

## Architectural Principles

1. **Separation of Concerns** — Mathematical primitives, higher-level operations, neural-network mathematics, numerical verification, visualization, experiments, and interfaces are separate modules with clear boundaries.

2. **Dependency Direction** — Dependencies flow inward toward primitives:
   ```
   experiments → visualization → neural_networks → optimization → probability → calculus → linear_algebra
                    ↓
               verification (cross-cutting)
   ```
   No circular dependencies. Verification is a cross-cutting concern used by all mathematical modules.

3. **Minimal Abstraction** — Implement mathematics directly with NumPy. Avoid unnecessary wrapper classes, factories, or patterns that obscure the math. Code should be readable as "executable mathematics."

4. **Inspectability** — All implementations must be understandable without debugger. Prefer functions over classes where appropriate. Expose internal state for verification.

5. **Numerical Verification First** — Every mathematical function that has an analytical form should have a verification test comparing analytical vs. numerical (finite-difference) results.

6. **Reproducibility** — All randomness uses explicit `numpy.random.Generator` with seeded PCG64. Experiment configs are deterministic.

7. **No Deep Learning Frameworks in Core** — PyTorch/JAX/TensorFlow are optional dependencies used ONLY for comparison/validation in experiments, never for core mathematical implementation.

## Package Boundaries

```
src/math_for_neural_networks/
├── __init__.py                    # Public API re-exports
├── linear_algebra/                # PRIMITIVES — No internal dependencies
│   ├── __init__.py
│   ├── scalars.py                 # Scalar operations
│   ├── vectors.py                 # Vector class, operations
│   ├── matrices.py                # Matrix class, operations
│   ├── norms.py                   # L1, L2, Linf, cosine similarity
│   └── verification.py            # Numerical verification for LA
├── calculus/                      # Requires linear_algebra
│   ├── __init__.py
│   ├── finite_differences.py      # Forward, central, complex-step
│   ├── derivatives.py             # derivative(), gradient()
│   ├── partial_derivatives.py     # Partial derivatives, Jacobian
│   ├── chain_rule.py              # Chain rule composition
│   ├── analytical.py              # Analytical derivatives (sigmoid, etc.)
│   └── verification.py            # Analytical vs. numerical verification
├── probability/                   # Requires linear_algebra
│   ├── __init__.py
│   ├── distributions.py           # Bernoulli, Categorical, Gaussian
│   ├── expectations.py            # Expectation, variance, covariance
│   ├── entropy.py                 # Entropy, cross-entropy, KL
│   ├── likelihood.py              # Likelihood, MLE
│   └── verification.py            # Numerical verification
├── optimization/                  # Requires calculus, linear_algebra
│   ├── __init__.py
│   ├── gradient_descent.py        # GD, LR schedules
│   ├── sgd.py                     # Stochastic GD (conceptual)
│   ├── momentum.py                # Momentum, Nesterov
│   ├── adam.py                    # Adam (conceptual)
│   └── verification.py            # Convergence verification
├── neural_networks/               # Requires all above
│   ├── __init__.py
│   ├── layers.py                  # Linear, Embedding, LayerNorm
│   ├── activations.py             # Sigmoid, ReLU, Tanh, GELU, Softmax
│   ├── losses.py                  # MSE, CrossEntropy, BCE
│   ├── attention.py               # Scaled dot-product attention
│   ├── containers.py              # Sequential, Module base
│   └── verification.py            # Gradient checking
├── verification/                  # CROSS-CUTTING — Used by all
│   ├── __init__.py
│   ├── numerical.py               # Finite-difference utilities
│   ├── tolerances.py              # Default tolerances, policies
│   ├── comparison.py              # Analytical vs. numerical comparison
│   └── reporting.py               # Verification result reporting
├── visualization/                 # Requires matplotlib, all math modules
│   ├── __init__.py
│   ├── functions.py               # 1D/2D function plots
│   ├── gradients.py               # Gradient fields, vector fields
│   ├── optimization.py            # Optimization trajectories
│   ├── landscapes.py              # Loss landscapes (1D/2D slices)
│   └── distributions.py           # Probability distribution plots
├── experiments/                   # Requires all above
│   ├── __init__.py
│   ├── runner.py                  # Experiment execution
│   ├── config.py                  # Experiment configuration
│   ├── logging.py                 # Structured logging
│   ├── analysis.py                # Result analysis
│   └── templates/                 # Experiment templates
└── cli/                           # Optional, future
    ├── __init__.py
    └── commands.py                # CLI entry points
```

## Responsibilities

| Module | Responsibility | Key Exports |
|--------|----------------|-------------|
| `linear_algebra` | Primitive tensor operations | `Vector`, `Matrix`, `dot`, `matmul`, `norm`, `cosine_similarity` |
| `calculus` | Differentiation, gradients | `derivative`, `gradient`, `jacobian`, `chain_rule` |
| `probability` | Distributions, information theory | `Distribution`, `entropy`, `cross_entropy`, `kl_divergence` |
| `optimization` | Parameter update algorithms | `GradientDescent`, `SGD`, `Momentum`, `Adam` |
| `neural_networks` | NN building blocks | `Linear`, `Embedding`, `LayerNorm`, `Sequential` |
| `verification` | Numerical verification framework | `verify_gradient`, `verify_jacobian`, `Tolerance` |
| `visualization` | Mathematical plotting | `plot_function`, `plot_gradient_field`, `plot_trajectory` |
| `experiments` | Structured experimentation | `Experiment`, `ExperimentConfig`, `run_experiment` |

## Dependency Direction Rules

1. **Primitives have no internal deps** — `linear_algebra` imports only NumPy and standard library
2. **Higher levels import lower levels** — `calculus` imports `linear_algebra`; `optimization` imports `calculus` and `linear_algebra`
3. **Verification is imported by all** — `verification` imports only NumPy; other modules import `verification`
4. **Visualization imports math modules** — Never vice versa
5. **Experiments imports all** — Orchestration layer only
6. **CLI imports experiments/visualization** — Interface layer only

## Testing Boundaries

- **Unit tests** — Co-located with module under test in `tests/` mirror structure
- **Integration tests** — Cross-module tests in `tests/integration/`
- **Numerical verification tests** — In each module's `verification.py` and `tests/*/test_verification.py`
- **Experiment tests** — In `tests/experiments/` — test experiment runner, not mathematical correctness

## Experiment Boundaries

- Experiments are **scripts** in `experiments/` and `scripts/`, not library code
- Experiments **import** from `math_for_neural_networks` but are not imported by it
- Experiment results go to `results/` (gitignored)
- Experiment configs are YAML/JSON in `experiments/configs/`

## Visualization Boundaries

- Visualization functions are **pure** — take data, return figure/axes or save to file
- No side effects in visualization module except file I/O
- Visualization does not compute mathematics — only displays it
- Colormaps: viridis, cividis (colorblind-safe); no jet/rainbow

## Configuration

- All configuration via Python dataclasses / Pydantic (future) in `experiments/config.py`
- No global config singletons
- Explicit config objects passed to functions

## Error Handling

- **Validation errors** — `ValueError` with descriptive messages (shape mismatch, invalid input)
- **Numerical errors** — `RuntimeError` for non-convergence, NaN/Inf detection
- **Verification failures** — `AssertionError` with detailed diff (for tests)
- **No custom exception hierarchy** — Keep it simple for educational clarity

## Type Hints

- **Strict mypy** — `disallow_untyped_defs = true`
- **NumPy typing** — Use `numpy.typing.NDArray` with shape hints in docstrings
- **Protocol-based** — For extensibility (e.g., `Optimizer` protocol)
- **No `Any`** — Except where truly dynamic (experiment config loading)

## Extensibility Points

1. **New distributions** — Subclass `Distribution` protocol in `probability`
2. **New optimizers** — Implement `Optimizer` protocol in `optimization`
3. **New layers** — Subclass `Module` in `neural_networks`
4. **New visualizations** — Add functions to `visualization` submodules
5. **New experiments** — Add configs to `experiments/configs/`, scripts to `experiments/`

## Future Architecture Considerations

- **Tensor abstraction** — If/when moving beyond 2D, introduce `Tensor` base class in `linear_algebra`
- **Autograd engine** — Stage 6 may introduce minimal autograd in `calculus/autograd.py`
- **JIT compilation** — Not planned; NumPy is sufficient for educational scale
- **Distributed** — Out of scope