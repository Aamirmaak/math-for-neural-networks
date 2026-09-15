# Contributing to Math for Neural Networks

Thank you for your interest in contributing! This project is an educational/research toolkit for understanding the mathematics behind neural networks.

## Ways to Contribute

- **Mathematical corrections** — Fix errors in implementations, derivations, or explanations
- **Numerical verification improvements** — Better tolerances, more thorough verification, edge cases
- **Educational examples** — Notebooks, scripts, or documentation that clarify concepts
- **Documentation improvements** — Clearer explanations, fixed typos, better structure
- **Test coverage** — Unit tests, property tests, numerical verification tests
- **Visualization** — Plots that improve mathematical understanding

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Aamirmaak/math-for-neural-networks.git
cd math-for-neural-networks

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## Code Standards

- **Python >= 3.10** — Use modern type hints (`list[]`, `X | Y`, etc.)
- **Type hints required** — All public functions must have type annotations
- **NumPy for numerics** — Core mathematics uses NumPy only (no PyTorch/JAX in core)
- **Ruff for linting/formatting** — `ruff check . && ruff format .`
- **Mypy strict mode** — `mypy src/` must pass
- **Tests required** — New functionality must include tests

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=math_for_neural_networks --cov-report=term-missing

# Run specific test categories
pytest -m unit        # Unit tests only
pytest -m numerical   # Numerical verification tests
pytest -m integration # Integration tests
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Ensure all checks pass: `ruff check . && ruff format . && mypy src/ && pytest`
5. Update relevant documentation
6. Submit a pull request

## Commit Guidelines

- Use clear, descriptive commit messages
- Reference issues when applicable
- Keep commits focused (one logical change per commit)
- Suggested prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

## Mathematical Implementation Guidelines

When implementing mathematical concepts:

1. **Explain first** — Add docstring explaining the concept, intuition, and neural network connection
2. **Implement clearly** — Prefer readable NumPy code over clever optimizations
3. **Test thoroughly** — Known values, edge cases, property tests
4. **Verify numerically** — Compare analytical vs. finite-difference results with justified tolerances
5. **Document tolerances** — Explain why specific tolerances are used

## Numerical Verification

All mathematical functions with analytical derivatives must include verification:

```python
def test_sigmoid_gradient_verification():
    analytical = sigmoid_gradient
    numerical = central_difference(sigmoid, h=1e-6)
    # Test with justified tolerance
    assert relative_error(analytical(x), numerical(x)) < 1e-7
```

## Documentation

- Update relevant `.md` files in `docs/` when adding features
- Keep `10_DECISIONS.md` updated with architectural decisions
- Update `11_PROGRESS_LOG.md` with stage progress
- Update `14_CHANGELOG.md` with notable changes

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Mathematical questions
- Documentation improvements

For security issues, please email the maintainers directly.