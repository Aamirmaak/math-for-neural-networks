# 06_EXPERIMENT_PLAN.md — Experiment Plan

## Experiment Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Verification** | Confirm analytical vs. numerical agreement | Gradient check on all functions |
| **Convergence** | Measure optimizer behavior | GD on convex/non-convex, LR sensitivity |
| **Visualization** | Generate educational plots | Gradient fields, loss landscapes, trajectories |
| **Comparison** | Compare implementations | Custom vs. NumPy vs. PyTorch |
| **Ablation** | Isolate component effects | With/without momentum, norm type |
| **Regression** | Detect regressions | Baseline metrics on fixed seeds |

## Experiment Lifecycle (Mandatory)

Every experiment must follow this structure:

```
HYPOTHESIS
  What is being tested? What is expected?
  Example: "Central difference gradient matches analytical within 1e-8 for smooth functions."

METHOD
  Exact procedure, parameters, random seeds
  Example: "Test f(x)=x^3-2x+1 on [-2,2] with 100 points. Seed=42. Central difference h=1e-6."

BASELINE
  What is the reference? Analytical? NumPy? PyTorch? Prior run?
  Example: "Analytical derivative 3x^2-2 computed via our implementation."

EXPERIMENT
  Actual execution — code, config, environment
  Example: "Run verification script with config Y. NumPy 1.26. Python 3.11."

RESULT
  Measured outcomes — numbers, plots, logs
  Example: "Max relative error: 2.3e-9. Mean relative error: 1.1e-10. Plot saved to results/grad_check.png."

ANALYSIS
  Interpretation of results
  Example: "Error within tolerance. Slight increase near x=0 due to floating point. Acceptable."

CONCLUSION
  Pass/fail, action items, follow-up
  Example: "PASS. Central difference verified for polynomials. Add test for sigmoid next."
```

## Experiment Report Format

All experiments produce a structured report (JSON + Markdown):

```json
{
  "experiment_id": "grad_verification_sigmoid_001",
  "category": "verification",
  "timestamp": "2024-01-15T10:30:00Z",
  "hypothesis": "Analytical sigmoid gradient matches central difference within 1e-7",
  "method": {
    "function": "sigmoid",
    "domain": [-10, 10],
    "n_points": 1000,
    "fd_method": "central",
    "h": 1e-6,
    "seed": 42
  },
  "baseline": "Analytical: sigmoid(x) * (1 - sigmoid(x))",
  "results": {
    "max_rel_error": 3.2e-8,
    "mean_rel_error": 1.1e-9,
    "max_abs_error": 2.1e-8,
    "passed": true,
    "tolerance": 1e-7
  },
  "analysis": "Error well within tolerance. Peak error at x≈0 where gradient=0.25.",
  "conclusion": "PASS. Sigmoid gradient verified.",
  "artifacts": ["results/grad_sigmoid_001.png", "results/grad_sigmoid_001.json"]
}
```

## Hypotheses (Planned)

### Linear Algebra Verification
- H1: Matrix multiplication matches NumPy within 1e-12 relative error
- H2: Dot product matches NumPy within 1e-12
- H3: Norm computations match NumPy within 1e-12
- H4: Cosine similarity in [-1, 1] with correct geometric interpretation

### Calculus Verification
- H5: Central difference matches analytical for polynomials (degree ≤ 5) within 1e-10
- H6: Central difference matches analytical for sigmoid/tanh/ReLU within 1e-7
- H7: Complex-step differentiation achieves 1e-14 for analytic functions
- H8: Chain rule composition matches manual gradient within 1e-7
- H9: Gradient of multivariate functions (e.g., f(x,y)=x²+y²) matches analytical

### Probability Verification
- H10: Distribution PMF/PDF integrates/sums to 1 within 1e-12
- H11: Entropy matches analytical for known distributions within 1e-10
- H12: Cross-entropy matches analytical for categorical distributions within 1e-10
- H13: MLE converges to true parameters for Gaussian (known variance)

### Optimization Verification
- H14: GD converges to minimum of convex quadratic with exact line search
- H15: Momentum accelerates convergence on ill-conditioned quadratics
- H16: Adam adapts learning rates per-parameter on sparse gradients
- H17: LR schedules produce expected decay curves

### Neural Network Verification
- H18: Linear layer forward + backward matches PyTorch within 1e-5
- H19: Softmax + cross-entropy gradient matches PyTorch within 1e-5
- H20: Backprop on 2-layer network matches PyTorch within 1e-5
- H21: Embedding gradient only non-zero at indexed positions

### Visualization Experiments
- H22: Gradient field plots correctly show direction/magnitude for f(x,y)=x²+y²
- H23: Optimization trajectory shows convergence path for GD on Rosenbrock
- H24: Loss landscape 2D slice reveals convex/non-convex structure correctly

## Baselines

| Experiment Type | Baseline |
|-----------------|----------|
| Verification | Analytical formula (primary), NumPy (secondary), PyTorch (tertiary) |
| Convergence | Theoretical convergence rates, prior experiment runs |
| Comparison | PyTorch (when available), NumPy, known mathematical results |
| Visualization | Known mathematical properties (e.g., gradient points uphill) |

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `max_rel_error` | Maximum relative error vs. baseline | < 1e-7 (verification) |
| `mean_rel_error` | Mean relative error | < 1e-9 (verification) |
| `convergence_rate` | Loss reduction per iteration | Matches theory |
| `iteration_count` | Steps to tolerance | Within expected range |
| `wall_time` | Execution time | Informational |
| `memory_peak` | Peak memory usage | Informational |

## Reproducibility Requirements

1. **Explicit seeds** — All randomness uses `numpy.random.Generator(PCG64(seed))`
2. **Pinned dependencies** — `requirements.txt` + `pip freeze` in experiment logs
3. **Environment capture** — Python version, OS, NumPy version in metadata
4. **Config versioning** — Experiment configs in `experiments/configs/` with version
5. **Deterministic execution** — Same config + seed = identical results
6. **Artifact preservation** — Plots, logs, JSON reports saved to `results/`

## Result Recording

- **Primary:** JSON report (machine-readable) in `results/experiments/`
- **Human:** Markdown summary in `results/experiments/` with plots embedded
- **Index:** `results/experiments/index.json` — searchable experiment registry
- **Visualization:** PNG/SVG in `results/figures/`

## Analysis Expectations

1. **Never fabricate results** — Only report what was actually measured
2. **Report failures honestly** — Failed verification is valuable data
3. **Distinguish error sources** — Implementation bug vs. floating-point vs. algorithmic limitation
4. **Quantify uncertainty** — Report confidence intervals where applicable (Monte Carlo)
5. **Link to theory** — Explain how results relate to mathematical expectations
6. **Document limitations** — Domain restrictions, numerical issues, assumptions

## Experiment Templates

Each category has a template in `experiments/templates/`:
- `verification_template.py`
- `convergence_template.py`
- `visualization_template.py`
- `comparison_template.py`

New experiments copy and adapt templates.