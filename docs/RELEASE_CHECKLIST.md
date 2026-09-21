# Release Checklist

Use this checklist before any public release.

## Pre-Release

- [ ] All tests passing (`pytest`)
- [ ] Lint passing (`ruff check src/`)
- [ ] Format check passing (`ruff format --check src/`)
- [ ] Type check passing (`mypy src/`)
- [ ] Coverage measured and acceptable
- [ ] Package builds (`python -m build`)
- [ ] Wheel installs correctly
- [ ] Source distribution builds
- [ ] `twine check dist/*` passes

## Documentation

- [ ] README complete and accurate
- [ ] All documentation links valid
- [ ] CHANGELOG updated with release version
- [ ] Version consistent across pyproject.toml and __init__.py
- [ ] CONTRIBUTING.md exists and is accurate
- [ ] LICENSE correct and consistent

## Repository

- [ ] No secrets or API keys in repository
- [ ] No placeholder URLs or usernames
- [ ] No broken links
- [ ] No accidental local paths
- [ ] .gitignore comprehensive
- [ ] No unnecessary generated files committed

## Code Quality

- [ ] No unused imports in source
- [ ] No hardcoded paths
- [ ] No hardcoded credentials
- [ ] All functions have type hints
- [ ] All public functions have docstrings

## Dependencies

- [ ] Core dependencies minimal (numpy, matplotlib)
- [ ] Dev dependencies documented
- [ ] Optional dependencies documented
- [ ] No unnecessary dependencies added

## Experiments

- [ ] All experiments executable
- [ ] Experiment results reproducible
- [ ] Experiment registry accurate

## Final

- [ ] `git diff` reviewed
- [ ] `git status` clean (no untracked files)
- [ ] Release artifacts inspected
- [ ] Version bumped appropriately
- [ ] Tag created (if applicable)
