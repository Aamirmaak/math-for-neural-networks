# 09_SECURITY.md — Security Considerations

## Threat Model

This is an **educational Python library** for mathematical learning. It is not a networked service, does not handle user data, authentication, or secrets. The primary security concerns are supply chain, dependency hygiene, and safe execution of user-provided code.

## Dependency Hygiene

### Principles
- **Minimal dependencies** — Only NumPy, Matplotlib, and dev tools
- **Pinned versions** — Minimum versions in `pyproject.toml`, lock files for CI
- **Regular updates** — Check for security advisories monthly
- **License compliance** — All dependencies have permissive licenses (BSD, MIT, PSF)

### Process
1. **Addition:** New dependency requires justification in PR, license check
2. **Updates:** `pip list --outdated` monthly; update in batches with testing
3. **Auditing:** `pip-audit` or `safety` in CI (when enabled)
4. **Transitive deps:** Monitor via `pipdeptree` for unexpected additions

### Current Dependency Risk Assessment

| Package | Risk | Mitigation |
|---------|------|------------|
| numpy | Low — mature, widely used, C extensions | Pin minimum, test on update |
| matplotlib | Low — pure Python + C extensions | Pin minimum, test on update |
| pytest | Dev only — not in runtime | Isolated in `[dev]` extra |
| ruff | Dev only — Rust binary | Isolated in `[dev]` extra |
| mypy | Dev only — pure Python | Isolated in `[dev]` extra |

## Unsafe Execution of Notebooks/Scripts

### Risk
Jupyter notebooks and Python scripts can execute arbitrary code. Users may run untrusted notebooks.

### Mitigation
- **Documentation:** Clear warning in README and notebook headers
- **No auto-execution:** Notebooks in repo are for reference; not executed in CI
- **Sandbox recommendation:** Users should run untrusted notebooks in isolated environments (Docker, VM, colab)
- **No `exec`/`eval`** in library code

## Input Validation

### Library Functions
- **Shape validation** — Explicit checks with descriptive `ValueError` messages
- **Type validation** — Type hints + runtime checks for public API
- **Value validation** — Domain checks (e.g., probability in [0,1], positive definite)
- **NaN/Inf handling** — Detect and raise `RuntimeError` with context

### Example
```python
def dot(v1: Vector, v2: Vector) -> float:
    if v1.shape != v2.shape:
        raise ValueError(f"Shape mismatch: {v1.shape} vs {v2.shape}")
    if not np.all(np.isfinite(v1.data)) or not np.all(np.isfinite(v2.data)):
        raise ValueError("Input contains NaN or Inf")
    return float(np.dot(v1.data, v2.data))
```

## Supply Chain Awareness

### Risks
- Compromised PyPI package (typosquatting, dependency confusion)
- Malicious PR/commit introducing backdoor
- Compromised maintainer account

### Mitigations
- **Verified publishers** — Only install from official PyPI
- **Hash verification** — `pip install --require-hashes` with lock file
- **Signed commits** — Require GPG-signed commits for maintainers (future)
- **2FA** — Mandatory for GitHub maintainer accounts
- **CODEOWNERS** — Require review for sensitive files

## Secrets Management

### Current State
- **No secrets in repo** — No API keys, tokens, passwords
- **No CI/CD secrets** — No deployment configured yet
- **Local env files** — `.env` in `.gitignore` for local development only

### Future (If CI/CD Added)
- GitHub Environments with secrets
- OIDC for PyPI publishing (no password/token in CI)
- No secrets in code, config, or logs

## Malicious Contribution Awareness

### Risks
- PRs that appear helpful but introduce vulnerabilities
- Obfuscated code in large PRs
- Dependency confusion via `requirements.txt` manipulation

### Mitigations
- **Small PRs preferred** — Easier to review
- **Required review** — At least one maintainer approval
- **Automated checks** — Lint, type, test must pass
- **Dependency review** — Manual check of `pyproject.toml` changes
- **No binary blobs** — All code in repo is source

## Vulnerability Reporting

### Process
1. **Private disclosure** — Email maintainer (no public issue)
2. **Acknowledgment** — Within 48 hours
3. **Assessment** — Severity, impact, fix timeline
4. **Fix** — Patch release if critical
5. **Disclosure** — Coordinated public announcement

### Contact
- Security issues: Create private GitHub Security Advisory or email maintainer

## What This Document Does NOT Cover

- **Application security** — Not a web app, no OWASP Top 10
- **Infrastructure security** — No servers, databases, networks
- **Data privacy** — No personal data processed
- **Compliance** — No regulatory requirements (GDPR, HIPAA, etc.)

## Security Checklist for Releases

- [ ] `pip-audit` clean (when CI enabled)
- [ ] No new dependencies without review
- [ ] No secrets in code, tests, or docs
- [ ] All PRs reviewed and approved
- [ ] Signed commits from maintainers (future)
- [ ] CHANGELOG notes security-relevant changes

## Note

This is a **living document**. As the project grows (CI, PyPI, contributors), security practices will evolve. The goal is proportionate security for an educational library — not enterprise-grade hardening.