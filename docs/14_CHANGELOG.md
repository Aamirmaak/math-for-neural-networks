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

## [Unreleased] — Stage 0: Project Foundation (2026-09-15)

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

### [0.2.0] — Stage 1: Linear Algebra Primitives (Planned)
**Target:** Matrix/Vector operations, norms, cosine similarity, numerical verification framework

### [0.3.0] — Stage 2: Calculus & Numerical Differentiation (Planned)
**Target:** Finite differences, analytical derivatives, chain rule, gradient verification

### [0.4.0] — Stage 3: Probability & Statistics Foundations (Planned)
**Target:** Distributions, entropy, cross-entropy, MLE, verification

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