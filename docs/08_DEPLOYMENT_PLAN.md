# 08_DEPLOYMENT_PLAN.md — Deployment Plan

## Deployment Targets

| Target | Status | Description |
|--------|--------|-------------|
| Local development | ✅ Ready | `pip install -e .` |
| GitHub repository | 📋 Planned | Source hosting, issues, wiki |
| PyPI package | 📋 Future | `pip install math-for-neural-networks` |
| Conda-forge | 📋 Future | `conda install -c conda-forge math-for-neural-networks` |
| Documentation site | 📋 Future | GitHub Pages / ReadTheDocs |

## Local Development

### Setup
```bash
# Clone
git clone https://github.com/Aamirmaak/math-for-neural-networks.git
cd math-for-neural-networks

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify
pytest --collect-only
python -c "import math_for_neural_networks; print(math_for_neural_networks.__version__)"
```

### Development Workflow
1. Make changes
2. `ruff check . && ruff format .`
3. `mypy src/`
4. `pytest`
5. Commit

## Python Package Distribution

### Build
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build wheel and sdist
pip install build
python -m build

# Output: dist/math_for_neural_networks-0.1.0-py3-none-any.whl
#         dist/math_for_neural_networks-0.1.0.tar.gz
```

### Verify Build
```bash
# Check package contents
pip install twine
twine check dist/*

# Test install from local wheel
pip install dist/math_for_neural_networks-0.1.0-py3-none-any.whl
python -c "import math_for_neural_networks; print('OK')"
```

### PyPI Release (Future — When Ready)

**Prerequisites:**
- All tests passing on 3.10, 3.11, 3.12
- Documentation complete for public API
- Version bumped (semantic versioning)
- CHANGELOG updated
- Git tag created

**Release Process:**
```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit: "Release v0.X.Y"
# 4. Tag: git tag -a v0.X.Y -m "Release v0.X.Y"
# 5. Push: git push origin main --tags
# 6. Build: python -m build
# 7. Upload: twine upload dist/*
# 8. GitHub Release: create from tag with notes
```

**PyPI Configuration:**
- Project name: `math-for-neural-networks` (hyphens)
- Import name: `math_for_neural_networks` (underscores)
- Classifiers: Education, Science/Research, AI/ML
- Requires-Python: >=3.10

## CLI Distribution (Future)

If CLI is implemented (Stage 7+):
- Entry point in `pyproject.toml`: `[project.scripts]` → `mn = math_for_neural_networks.cli:main`
- Installable via `pip install math-for-neural-networks`
- Optional: `pipx install math-for-neural-networks` for isolated CLI

## Reproducibility

### Environment Locking
```bash
# Generate lock file
pip freeze > requirements-lock.txt

# Or use pip-tools
pip install pip-tools
pip-compile pyproject.toml -o requirements-lock.txt
```

### Docker (Future — If Needed)
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install -e .
COPY src/ ./src/
# Experiments/notebooks mounted at runtime
```

## Release Strategy

### Versioning: Semantic Versioning (SemVer)

| Version | Meaning |
|---------|---------|
| 0.1.0 | Stage 0 complete (current) |
| 0.2.0 | Stage 1 complete (Linear Algebra) |
| 0.3.0 | Stage 2 complete (Calculus) |
| 0.4.0 | Stage 3 complete (Probability) |
| 0.5.0 | Stage 4 complete (Optimization) |
| 0.6.0 | Stage 5 complete (NN Mathematics) |
| 0.7.0 | Stage 6 complete (Backprop) |
| 0.8.0 | Stage 7 complete (Experiments) |
| 0.9.0 | Stage 8 complete (PyTorch Comparison) |
| 1.0.0 | MVP complete, stable API |

**Pre-1.0:** Minor version = potentially breaking changes. Patch = fixes only.

### Release Cadence
- **Stage releases:** When each stage passes verification checkpoints
- **Patch releases:** Bug fixes, documentation updates
- **No fixed schedule** — Quality-gated, not time-gated

### Release Artifacts
- GitHub Release with notes
- PyPI wheel + sdist
- Changelog entry
- Updated documentation (if applicable)

## GitHub Repository Setup (When Published)

### Repository Settings
- **Visibility:** Public
- **Branch protection:** `main` — require PR, status checks
- **Default branch:** `main`
- **Issues:** Enabled
- **Wiki:** Enabled (for extended docs)
- **Discussions:** Enabled (for Q&A)
- **Projects:** Enabled (for roadmap tracking)

### Branch Strategy
- `main` — Stable, released versions only
- `develop` — Integration branch (optional, if team grows)
- `feature/*` — Feature branches from `main`
- `hotfix/*` — Urgent fixes from tags

### Labels
- `stage-0` through `stage-8`
- `area: linear-algebra`, `area: calculus`, etc.
- `type: feature`, `type: bug`, `type: docs`, `type: test`
- `status: planned`, `status: in-progress`, `status: review`, `status: done`
- `priority: high`, `medium`, `low`
- `good first issue` — For contributors

## Documentation Deployment (Future)

### Options
1. **GitHub Pages + MkDocs** — Simple, free, versioned
2. **ReadTheDocs** — Auto-builds, search, versions
3. **GitHub Wiki** — Low friction, no build

### Recommended: MkDocs + GitHub Pages
```yaml
# mkdocs.yml
site_name: Math for Neural Networks
theme:
  name: material
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference: api/
  - Tutorials: tutorials/
  - Experiments: experiments/
repo_url: https://github.com/Aamirmaak/math-for-neural-networks
```

## Contributor Onboarding (Future)

1. `CONTRIBUTING.md` — Guidelines, code style, PR process
2. `CODE_OF_CONDUCT.md` — Community standards
3. `DEVELOPMENT.md` — Detailed setup, workflow
4. Issue templates — Bug report, feature request, question
5. PR template — Checklist, related issues

## Current Status: Stage 0

**Ready for:**
- Local development (`pip install -e .`)
- GitHub push (all foundation files present)
- Stage 1 implementation start

**Not Ready for:**
- PyPI publication (no implementation)
- Public announcement (pre-alpha)
- Contributor onboarding (no CONTRIBUTING.md yet)