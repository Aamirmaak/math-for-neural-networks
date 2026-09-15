# 02_TRD.md — Technical Requirements Document

## Python Version Strategy

- **Minimum:** Python 3.10
- **Target:** Python 3.10, 3.11, 3.12
- **Rationale:** Modern type hinting (union types `|`, `list[]`, `dict[]`), pattern matching, performance improvements. No 3.13+ yet for ecosystem stability.

## Dependencies

### Core (Mandatory)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| numpy | >=1.24.0 | Numerical computing, arrays, linear algebra | BSD-3-Clause |
| matplotlib | >=3.7.0 | Plotting, visualization | PSF |

### Development (Optional — `.[dev]`)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| pytest | >=7.4.0 | Testing framework | MIT |
| pytest-cov | >=4.1.0 | Coverage reporting | MIT |
| ruff | >=0.1.0 | Linting + formatting | MIT |
| mypy | >=1.5.0 | Static type checking | MIT |
| black | >=23.0.0 | Code formatting (fallback) | MIT |
| isort | >=5.12.0 | Import sorting | MIT |
| pre-commit | >=3.5.0 | Git hooks | MIT |

### Notebook (Optional — `.[notebook]`)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| jupyter | >=1.0.0 | Notebook server | BSD-3-Clause |
| ipykernel | >=6.25.0 | Kernel for notebooks | BSD-3-Clause |
| nbformat | >=5.9.0 | Notebook format handling | BSD-3-Clause |

### Experiment (Optional — `.[experiment]`)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation for experiment logs | BSD-3-Clause |
| scipy | >=1.11.0 | Scientific computing (stats, optimize) | BSD-3-Clause |

## Packaging Approach

- **Build backend:** setuptools (via pyproject.toml)
- **Source layout:** `src/math_for_neural_networks/`
- **Distribution:** wheel + sdist
- **Versioning:** Semantic versioning (MAJOR.MINOR.PATCH)
- **Current:** 0.1.0 (pre-alpha, Stage 0)

## Testing Stack

- **Framework:** pytest
- **Configuration:** `pyproject.toml` `[tool.pytest.ini_options]`
- **Markers:** `unit`, `integration`, `numerical`, `slow`
- **Coverage:** pytest-cov with branch coverage
- **Type checking:** mypy in strict mode
- **Linting:** ruff (replaces flake8, isort, black, pyupgrade)

## Visualization Stack

- **Primary:** Matplotlib (static plots, publication quality)
- **Backend:** Agg (non-interactive) for CI/headless; TkAgg/Qt5Agg for local
- **Style:** Clean, minimal, colorblind-safe palettes
- **Export:** PNG for docs, SVG for scalability
- **No:** Plotly, Bokeh, Altair (added complexity, not needed for MVP)

## Numerical Computing Approach

- **Primary:** NumPy arrays (`ndarray`) for all tensor operations
- **Precision:** float64 default; float32 where explicitly needed
- **Randomness:** `numpy.random.Generator` (PCG64) with explicit seeds
- **No:** PyTorch, JAX, TensorFlow for core mathematics
- **PyTorch:** Allowed ONLY for comparison/validation in experiments (optional dependency)

## Compatibility Considerations

| Concern | Approach |
|---------|----------|
| OS | Pure Python + NumPy — cross-platform (Linux, macOS, Windows) |
| Python versions | Test matrix: 3.10, 3.11, 3.12 |
| NumPy versions | Pin minimum (1.24), test with latest |
| Architecture | x86_64, ARM64 (Apple Silicon, Linux ARM) |
| CI | GitHub Actions (when enabled) — test matrix above |
| Dependencies | Minimal, well-maintained, permissive licenses |

## Project Structure

```
math-for-neural-networks/
├── pyproject.toml           # Build config, metadata, tool config
├── requirements.txt         # Core deps (for pip install -r)
├── LICENSE                  # MIT + dependency license notice
├── .gitignore               # Standard Python ignores
├── README.md                # Project overview
├── docs/                    # All documentation (PRD, TRD, Architecture, etc.)
├── src/
│   └── math_for_neural_networks/   # Package root
│       ├── __init__.py
│       ├── linear_algebra/   # PLANNED
│       ├── calculus/         # PLANNED
│       ├── probability/      # PLANNED
│       ├── optimization/     # PLANNED
│       ├── neural_networks/  # PLANNED
│       ├── verification/     # PLANNED
│       ├── visualization/    # PLANNED
│       └── experiments/      # PLANNED
├── tests/                   # pytest tests (mirrors src structure)
├── examples/                # Standalone Python scripts
├── notebooks/               # Jupyter notebooks
├── experiments/             # Experiment scripts + configs
├── scripts/                 # Utility scripts (CLI, helpers)
└── results/                 # Experiment outputs (gitignored)
```

## Configuration Files

- `pyproject.toml` — Single source of truth for build, tools, metadata
- `requirements.txt` — Human-readable core deps (generated from pyproject.toml)
- `.gitignore` — Standard Python ignores
- `.pre-commit-config.yaml` — To be added in Stage 1
- `CONTRIBUTING.md` — To be added in Stage 1

## Development Workflow

1. Clone repo
2. `python -m venv .venv && source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
3. `pip install -e ".[dev]"`
4. `pre-commit install` (when available)
5. `pytest` to run tests
6. `ruff check . && ruff format .` to lint/format
7. `mypy src/` to type-check