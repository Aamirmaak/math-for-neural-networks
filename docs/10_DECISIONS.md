# 10_DECISIONS.md — Architectural Decisions Log

## Format
Each decision records: **Decision**, **Rationale**, **Alternatives Considered**, **Consequences**, **Status**.

---

## DEC-001: Python Version Minimum 3.10

**Decision:** Require Python >= 3.10

**Rationale:**
- Modern type hints: `list[]`, `dict[]`, `X | Y` union syntax
- Pattern matching (`match`/`case`) for cleaner control flow
- Performance improvements in 3.10+ (faster startup, better error messages)
- 3.10 is widely available (Ubuntu 22.04+, macOS Homebrew, Windows Store)
- 3.9 reaches EOL October 2025

**Alternatives Considered:**
- Python 3.9: Lacks modern typing syntax, EOL soon
- Python 3.11+: Not yet universally deployed

**Consequences:**
- Cannot run on older systems without Python upgrade
- Cleaner, more maintainable type annotations

**Status:** ✅ Accepted (Stage 0)

---

## DEC-002: NumPy as Primary Numerical Backend

**Decision:** Use NumPy (`ndarray`) for all tensor operations; no PyTorch/JAX/TensorFlow in core

**Rationale:**
- Educational transparency: NumPy operations are visible, inspectable
- Minimal dependency: Single well-maintained C-accelerated library
- Universal: NumPy is the lingua franca of scientific Python
- No hidden autograd/graph building — forces manual implementation for learning
- PyTorch/JAX available as optional comparison dependencies only

**Alternatives Considered:**
- Pure Python lists: Too slow, no broadcasting, no linear algebra
- PyTorch tensors: Adds heavy dependency, hides autograd mechanics
- JAX: Similar to PyTorch, functional paradigm adds cognitive load
- Custom tensor class wrapping NumPy: Adds abstraction without benefit

**Consequences:**
- No GPU acceleration (acceptable for educational scale)
- No automatic differentiation (manual implementation is the point)
- Must implement broadcasting, indexing manually where NumPy doesn't match math notation

**Status:** ✅ Accepted (Stage 0)

---

## DEC-003: Source Layout (`src/math_for_neural_networks`)

**Decision:** Use `src/` layout with package under `src/math_for_neural_networks/`

**Rationale:**
- Prevents accidental imports from working directory during development
- Enforces `pip install -e .` for testing (catches packaging issues early)
- Standard modern Python packaging layout (setuptools, pip, build)
- Clear separation of package code from tests, docs, scripts

**Alternatives Considered:**
- Flat layout (`math_for_neural_networks/` at root): Risks import confusion
- `src/` layout with flat package: Less common, tooling expects nested

**Consequences:**
- Slightly longer import path in editable install
- Must configure `pyproject.toml` `[tool.setuptools.packages.find]` correctly

**Status:** ✅ Accepted (Stage 0)

---

## DEC-004: MIT License

**Decision:** MIT License for the project

**Rationale:**
- Simple, permissive, widely understood
- Compatible with all downstream uses (commercial, academic, proprietary)
- Matches educational/open-source intent
- Minimal legal complexity for contributors

**Alternatives Considered:**
- Apache 2.0: Patent clause adds complexity not needed here
- GPL: Viral, restricts commercial use
- BSD-3: Similar to MIT but longer; no practical advantage

**Consequences:**
- Anyone can use, modify, distribute, sell
- No warranty or liability
- Must include license copy in distributions

**Status:** ✅ Accepted (Stage 0)

---

## DEC-005: No Web UI / Dashboard

**Decision:** CLI, notebooks, and scripts only; no web interface

**Rationale:**
- Educational focus: Web UI adds complexity (React, FastAPI, WebSockets, deployment)
- Maintenance burden: Frontend framework churn, security updates
- Notebooks provide interactive exploration better than dashboards
- CLI enables automation, scripting, CI integration
- Future: Only if genuine educational value proven (e.g., interactive gradient visualization)

**Alternatives Considered:**
- Streamlit: Simple but adds dependency, limited customization
- Custom React + FastAPI: High effort, low educational ROI
- Jupyter widgets: Could be explored later for specific visualizations

**Consequences:**
- No "pretty" portfolio demo (intentional)
- Focus remains on mathematical implementation quality

**Status:** ✅ Accepted (Stage 0)

---

## DEC-006: Verification as Cross-Cutting Module

**Decision:** `verification/` module imported by all mathematical modules

**Rationale:**
- Numerical verification is a universal concern, not tied to one domain
- Centralizes tolerance policies, comparison logic, reporting
- Avoids duplication of finite-difference code across modules
- Enables consistent verification methodology

**Alternatives Considered:**
- Verification in each module: Duplication, inconsistency
- External package: Overkill for internal utilities
- Tests only: Verification needed at runtime for experiments too

**Consequences:**
- All math modules depend on `verification`
- `verification` must have zero internal dependencies (only NumPy, stdlib)

**Status:** ✅ Accepted (Stage 0)

---

## DEC-007: Strict Type Checking (mypy strict mode)

**Decision:** Enable all strict mypy flags: `disallow_untyped_defs`, `strict_equality`, `no_implicit_optional`, etc.

**Rationale:**
- Catches bugs early (None handling, missing returns, type mismatches)
- Serves as executable documentation
- Forces explicit handling of edge cases (Optional, Union)
- Educational value: Types clarify mathematical function signatures

**Alternatives Considered:**
- Basic type checking: Misses many issues
- Gradual typing (allow untyped): Defeats purpose

**Consequences:**
- More verbose type annotations
- Slower initial development (more upfront thinking)
- Higher quality, fewer runtime surprises

**Status:** ✅ Accepted (Stage 0)

---

## DEC-008: Ruff for Linting + Formatting

**Decision:** Use Ruff (replaces flake8, isort, black, pyupgrade, autoflake)

**Rationale:**
- Extremely fast (Rust-based)
- Single tool replaces 5+ tools
- Consistent configuration in `pyproject.toml`
- Active development, growing rule set
- Drop-in replacement for black/isort

**Alternatives Considered:**
- Black + isort + flake8: Multiple tools, slower, config fragmentation
- Pyright: Good type checker but not a linter/formatter

**Consequences:**
- Some niche flake8 plugins not available (acceptable)
- Configuration syntax differs slightly from flake8

**Status:** ✅ Accepted (Stage 0)

---

## DEC-009: Experiment Reports as JSON + Markdown

**Decision:** Structured JSON reports with Markdown summaries

**Rationale:**
- JSON: Machine-readable, queryable, indexable, versionable
- Markdown: Human-readable, renderable on GitHub, includes plots
- Dual format serves both automation and documentation
- Experiment registry (`index.json`) enables discovery

**Alternatives Considered:**
- CSV: No nesting, poor for complex metadata
- Database: Overkill, not portable
- Plain text: Not parseable

**Consequences:**
- Need JSON serialization for all result types
- Plot files referenced by path in JSON

**Status:** ✅ Accepted (Stage 0)

---

## DEC-010: Tolerance Policy for Numerical Verification

**Decision:** Default tolerances per comparison type (see Testing Strategy)

**Rationale:**
- Float64 precision limits: ~1e-15 relative
- Finite difference error: O(h²) for central, optimal h ~ 1e-6 → ~1e-12
- Complex step: No cancellation → ~1e-14
- PyTorch comparison: Different op ordering, fusion → ~1e-5
- Monte Carlo: Statistical variance → ~1e-2 to 1e-3

**Alternatives Considered:**
- Single tolerance: Too strict for some, too loose for others
- Per-function tolerance: Hard to maintain consistency

**Consequences:**
- Tests must specify which tolerance policy applies
- Documentation of rationale required for non-default tolerances

**Status:** ✅ Accepted (Stage 0)

---

## DEC-011: Randomness via numpy.random.Generator (PCG64)

**Decision:** Use `numpy.random.Generator(np.random.PCG64(seed))` exclusively

**Rationale:**
- Modern, high-quality RNG (replaces legacy `RandomState`)
- Explicit seed → deterministic, reproducible
- Substream support via `SeedSequence` for parallel experiments
- Standard in NumPy 1.17+

**Alternatives Considered:**
- `np.random.seed()` + global state: Not thread-safe, legacy
- Python `random`: Slower, not array-oriented
- JAX/PyTorch RNG: Not available in core

**Consequences:**
- All stochastic code must accept `rng` parameter or create from seed
- No global random state manipulation

**Status:** ✅ Accepted (Stage 0)

---

## DEC-012: Stage-Gated Implementation with Verification Checkpoints

**Decision:** Each stage must pass verification checkpoints before next stage

**Rationale:**
- Prevents building on broken foundations
- Ensures numerical correctness propagates
- Forces documentation updates at each milestone
- Makes progress measurable

**Alternatives Considered:**
- Continuous implementation: Risk of compounding errors
- Big bang at end: No feedback, high integration risk

**Consequences:**
- Slower initial progress (more verification per stage)
- Higher confidence in final result
- Clear go/no-go criteria per stage

**Status:** ✅ Accepted (Stage 0)

---

## DEC-013: Status Markers (PLANNED/IMPLEMENTED/VERIFIED/DEFERRED)

**Decision:** All documentation uses explicit status markers

**Rationale:**
- Prevents confusion about what exists vs. what's planned
- Enables automated checking (grep for IMPLEMENTED without tests)
- Honest communication with users/contributors
- Tracks verification state separately from implementation

**Alternatives Considered:**
- Version-based (v0.1, v0.2): Doesn't capture verification state
- Done/Not Done: Too binary, misses "implemented but not verified"

**Consequences:**
- Every feature list, API doc, roadmap must use markers
- Discipline required to update markers

**Status:** ✅ Accepted (Stage 0)

---

## DEC-014: No Fabricated Results in Documentation

**Decision:** Never write numerical results unless experiment actually executed

**Rationale:**
- Scientific integrity
- Prevents hallucinated benchmarks
- Forces actual experimentation
- Builds trust in project outputs

**Alternatives Considered:**
- Placeholder results with [TODO]: Still misleading if forgotten
- Theoretical predictions labeled as such: Better but still not measured

**Consequences:**
- Documentation stays sparse until experiments run
- Results sections remain empty until Stage 7+

**Status:** ✅ Accepted (Stage 0)

---

## DEC-015: Optional Dependencies via Extras

**Decision:** Use `pyproject.toml` `[project.optional-dependencies]` for dev, notebook, experiment deps

**Rationale:**
- Users install only what they need
- Core install stays minimal (NumPy + Matplotlib only)
- Clear separation of concerns
- Standard pip/setuptools feature

**Alternatives Considered:**
- Single requirements.txt with all: Bloats minimal installs
- Separate requirements files: Less discoverable, not standard

**Consequences:**
- Must document extras in README
- CI must test core + each extra combination

**Status:** ✅ Accepted (Stage 0)

---

## DEC-016: Experiment Configs as YAML/JSON Files

**Decision:** Store experiment configurations in `experiments/configs/` as versioned files

**Rationale:**
- Reproducibility: Config + seed = exact reproduction
- Version control: Track config evolution with code
- Separation: Config separate from code logic
- Sharing: Easy to share/exchange experiment setups

**Alternatives Considered:**
- Python config scripts: Executable but harder to parse/validate
- Environment variables: Not versionable, limited structure
- Database: Overkill

**Consequences:**
- Need config loading/validation in `experiments/config.py`
- Schema evolution requires migration strategy

**Status:** ✅ Accepted (Stage 0)

---

## DEC-017: Colorblind-Safe Visualization Defaults

**Decision:** Use viridis/cividis colormaps; avoid jet/rainbow

**Rationale:**
- ~8% of males have color vision deficiency
- Viridis/cividis are perceptually uniform
- Scientific standard (matplotlib default since 2.0)
- Accessible without extra effort

**Alternatives Considered:**
- Custom palettes: Hard to get right
- Jet/rainbow: Perceptually misleading, not colorblind-safe

**Consequences:**
- All visualization code uses approved colormaps
- User can override if needed

**Status:** ✅ Accepted (Stage 0)

---

## DEC-018: No Automatic Differentiation in Core

**Decision:** Implement gradients manually (analytical + finite difference); no autograd engine in core

**Rationale:**
- Educational goal: Understand chain rule by implementing it
- Autograd hides the mathematical structure
- Manual implementation forces explicit graph thinking
- Autograd can be added as separate module in future (Stage 6+)

**Alternatives Considered:**
- Micrograd-style autograd: Valuable but separate learning objective
- PyTorch autograd: Defeats purpose of learning

**Consequences:**
- More boilerplate for gradient computation
- Deeper understanding of backpropagation mechanics
- Verification via finite difference is mandatory

**Status:** ✅ Accepted (Stage 0)

---

## DEC-019: Float64 Default Precision

**Decision:** Use float64 (NumPy default) for all computations

**Rationale:**
- Higher precision reduces numerical verification noise
- Float32 introduces quantization errors that mask implementation bugs
- Educational clarity: See true mathematical behavior first
- Float32 can be added as optimization later

**Alternatives Considered:**
- Float32 default: Matches ML practice but obscures numerical issues
- Configurable: Adds complexity without early benefit

**Consequences:**
- Slower than float32 (negligible for educational scale)
- Higher memory (negligible for educational scale)
- Must explicitly cast for PyTorch comparison (float32)

**Status:** ✅ Accepted (Stage 0)

---

## DEC-020: Documentation in `docs/` with Numbered Prefixes

**Decision:** All project docs in `docs/` with `NN_NAME.md` format

**Rationale:**
- Logical ordering (PRD → TRD → Plan → Architecture → ...)
- Easy navigation and reference
- Separate from package code
- Version controlled with code

**Alternatives Considered:**
- README-only: Insufficient for complex project
- Wiki: Not versioned with code
- `docs/` without numbering: Hard to maintain order

**Consequences:**
- 14 core documents + future additions
- Must maintain cross-references

**Status:** ✅ Accepted (Stage 0)

---

## DEC-021: Ruff Only for Linting + Formatting (No Black)

**Decision:** Remove `black` from dev dependencies; use Ruff for both linting and formatting

**Rationale:**
- Ruff's formatter is a drop-in replacement for Black
- Single tool reduces complexity and configuration
- Faster execution (Rust-based)
- Consistent configuration in single `pyproject.toml` section

**Alternatives Considered:**
- Keep both Black and Ruff: Redundant, potential conflicts
- Use Black for formatting, Ruff for linting: Two tools to maintain

**Consequences:**
- Slightly different formatting defaults (configurable via `[tool.ruff.format]`)
- One less dependency in dev environment

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-022: Fix pyproject.toml TOML Syntax for per-file-ignores

**Decision:** Use inline TOML table syntax for `per-file-ignores` instead of multiline

**Rationale:**
- Python 3.14's tomllib (standard library) is stricter about TOML syntax
- Inline table `{ "key" = ["value"] }` is valid; multiline with colon syntax caused parse error
- Avoids dependency on external `tomli` package

**Alternatives Considered:**
- Upgrade setuptools/pip to handle newer TOML: Not a reliable fix
- Use external tomli: Adds unnecessary dependency

**Consequences:**
- Configuration more compact but less readable for many entries
- Acceptable for current small number of per-file ignores

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-023: Results Directory Gitignore Policy

**Decision:** Ignore generated/temporary artifacts in `results/` but allow curated summaries and reports to be committed

**Rationale:**
- Reproducible experiment summaries (JSON, Markdown) and plots (PNG, SVG) have educational value
- Large binary artifacts (`.npy`, `.pkl`, `.h5`) should not be committed
- Temporary logs and cache files should be ignored
- Provides flexibility for future curated results

**Alternatives Considered:**
- Ignore entire `results/`: Simpler but loses ability to version important results
- Track all results: Bloats repo with large files

**Consequences:**
- Requires discipline to only commit curated summaries
- Git LFS may be needed for larger artifacts in future

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-024: Status Language Precision

**Decision:** Use precise status markers: `PLANNED`, `DOCUMENTED`, `IMPLEMENTED`, `TESTED`, `VERIFIED` — do not use `VERIFIED` or `COMPLETE` for documentation-only stages

**Rationale:**
- "Verified" implies actual validation (tests, numerical verification) was executed
- Stage 0 only created documentation and infrastructure — no mathematical validation occurred
- Accurate terminology prevents confusion and maintains scientific integrity

**Alternatives Considered:**
- Keep "VERIFIED" for Stage 0: Misleading, implies more than was done
- Use "DONE": Too ambiguous

**Consequences:**
- Documentation must be updated to reflect accurate status
- CHANGELOG, PROGRESS_LOG, README updated accordingly

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-025: GitHub URL Correction

**Decision:** Replace all placeholder `yourusername` GitHub URLs with actual repository `Aamirmaak`

**Rationale:**
- Project metadata should reference actual repository
- Enables proper links in PyPI, documentation, and package metadata
- Professional open-source practice

**Consequences:**
- Updated in: pyproject.toml, README.md, docs/08_DEPLOYMENT_PLAN.md, src/__init__.py

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-026: License Copyright Year and Dependency List Correction

**Decision:** Update LICENSE copyright year to 2026; remove black from dependency license list (no longer used)

**Rationale:**
- Copyright year should reflect actual project year
- Dependency license notice should match actual dependencies used
- "Dependency licenses should be reviewed and verified before release/distribution" — accurate cautious wording

**Status:** ✅ Accepted (Stage 0 correction)

---

## DEC-027: CONTRIBUTING.md Creation

**Decision:** Create CONTRIBUTING.md since README references it

**Rationale:**
- Professional open-source projects should have contribution guidelines
- Helps maintain code quality and project standards
- Documents mathematical implementation guidelines specific to this project

**Status:** ✅ Accepted (Stage 0 correction)

---

*End of decisions. New decisions appended as made.*