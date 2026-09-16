# 12_EXPERIMENT_LOG.md — Experiment Log

## Purpose

Record all experiments executed during the project. Each entry follows the mandatory lifecycle format from EXPERIMENT_PLAN.md.

**CRITICAL RULE:** Never fabricate experimental results. Only record experiments that have actually been executed. Empty sections are honest — they indicate work not yet done.

## Experiment Entry Format

```markdown
## EXP-XXX: Short Descriptive Name

**Date:** YYYY-MM-DD
**Stage:** N
**Category:** verification | convergence | visualization | comparison | ablation | regression
**Status:** PLANNED | RUNNING | COMPLETE | FAILED

### HYPOTHESIS
What is being tested? What is expected?

### METHOD
Exact procedure, parameters, random seeds, environment.

### BASELINE
Reference implementation or theoretical value.

### EXPERIMENT
Actual execution details: script, config, command run.

### RESULT
Measured outcomes: numbers, plots, logs. Links to artifacts.

### ANALYSIS
Interpretation of results.

### CONCLUSION
PASS/FAIL, action items, follow-up experiments.

### ARTIFACTS
- results/experiments/EXP-XXX.json
- results/figures/EXP-XXX.png
- results/experiments/EXP-XXX.md
```

---

## Experiment Index

| ID | Name | Stage | Category | Status | Date |
|----|------|-------|----------|--------|------|
| EXP-401 | Learning Rate Sensitivity Analysis | 4 | ablation | COMPLETE | 2026-09-16 |
| EXP-402 | 2D Gradient Descent Path | 4 | visualization | COMPLETE | 2026-09-16 |
| EXP-403 | Momentum vs Gradient Descent | 4 | comparison | COMPLETE | 2026-09-16 |
| EXP-404 | Adam vs Momentum vs GD | 4 | comparison | COMPLETE | 2026-09-16 |
| EXP-405 | Initialization Effect on Convergence | 4 | ablation | COMPLETE | 2026-09-16 |

---

## Stage 1 Experiments (Planned)

### EXP-101: Matrix Multiplication Numerical Verification
**Category:** verification
**Hypothesis:** Custom matmul matches NumPy within 1e-12 relative error
**Planned:** Stage 1 completion

### EXP-102: Dot Product Verification
**Category:** verification
**Hypothesis:** Custom dot matches NumPy within 1e-12
**Planned:** Stage 1 completion

### EXP-103: Norm Computation Verification
**Category:** verification
**Hypothesis:** L1/L2/Linf norms match NumPy within 1e-12
**Planned:** Stage 1 completion

### EXP-104: Cosine Similarity Range Verification
**Category:** verification
**Hypothesis:** Cosine similarity always in [-1, 1] for non-zero vectors
**Planned:** Stage 1 completion

---

## Stage 2 Experiments (Planned)

### EXP-201: Finite Difference Accuracy on Polynomials
**Category:** verification
**Hypothesis:** Central difference O(h²) error observed for polynomials degree ≤ 5
**Planned:** Stage 2 completion

### EXP-202: Complex-Step Differentiation Accuracy
**Category:** verification
**Hypothesis:** Complex step achieves ~1e-14 for analytic functions
**Planned:** Stage 2 completion

### EXP-203: Sigmoid/Tanh/ReLU Gradient Verification
**Category:** verification
**Hypothesis:** Analytical gradients match central difference within 1e-7
**Planned:** Stage 2 completion

### EXP-204: Chain Rule Composition Verification
**Category:** verification
**Hypothesis:** Composed function gradients match manual chain rule within 1e-7
**Planned:** Stage 2 completion

### EXP-205: Multivariate Gradient Verification
**Category:** verification
**Hypothesis:** Gradient of f(x,y)=x²+y² matches [2x, 2y] within tolerance
**Planned:** Stage 2 completion

---

## Stage 3 Experiments (Planned)

### EXP-301: Distribution Normalization Verification
**Category:** verification
**Hypothesis:** PMF sums to 1, PDF integrates to 1 within 1e-12
**Planned:** Stage 3 completion

### EXP-302: Entropy Analytical Verification
**Category:** verification
**Hypothesis:** Computed entropy matches analytical for known distributions
**Planned:** Stage 3 completion

### EXP-303: Cross-Entropy Verification
**Category:** verification
**Hypothesis:** Cross-entropy matches analytical for categorical distributions
**Planned:** Stage 3 completion

### EXP-304: MLE Convergence for Gaussian
**Category:** convergence
**Hypothesis:** MLE converges to true parameters for Gaussian (known variance)
**Planned:** Stage 3 completion

---

## Stage 4 Experiments (Planned)

### EXP-401: GD Convergence on Convex Quadratic
**Category:** convergence
**Hypothesis:** GD with exact line search converges to minimum
**Planned:** Stage 4 completion

### EXP-402: Momentum Acceleration on Ill-Conditioned Quadratic
**Category:** comparison
**Hypothesis:** Momentum converges faster than plain GD on ill-conditioned problems
**Planned:** Stage 4 completion

### EXP-403: Adam Adaptive Learning Rates
**Category:** visualization
**Hypothesis:** Adam shows per-parameter LR adaptation on sparse gradients
**Planned:** Stage 4 completion

### EXP-404: LR Schedule Visualization
**Category:** visualization
**Hypothesis:** Step, cosine, warmup schedules produce expected curves
**Planned:** Stage 4 completion

---

## Stage 4 Experiments (Completed)

### EXP-401: Learning Rate Sensitivity Analysis
**Category:** ablation
**Status:** COMPLETE
**Hypothesis:** There is an optimal learning rate range; too small = slow, too large = diverge.
**Result:** For f(x) = x^2, optimal lr is 0.2-0.5. lr = 0.5 gives exact convergence in 1 step. lr > 0.5 causes oscillation. lr >= 1.0 causes divergence. Max stable lr = 1/a for f(x) = a*x^2.

### EXP-402: 2D Gradient Descent Path
**Category:** visualization
**Status:** COMPLETE
**Hypothesis:** GD follows steepest descent on quadratic; struggles on non-convex Rosenbrock.
**Result:** Quadratic converges cleanly to origin. Rosenbrock: GD gets close (distance 0.13 from minimum after 5000 iterations) but can't reach exact minimum due to narrow valley.

### EXP-403: Momentum vs Gradient Descent
**Category:** comparison
**Status:** COMPLETE
**Hypothesis:** Momentum accelerates convergence on ill-conditioned functions.
**Result:** beta=0.5 gives moderate speedup. beta=0.9 gives significant speedup. beta=0.95-0.99 causes overshooting. On f(x,y) = x^2 + 10*y^2, momentum with beta=0.5 reaches 0.00004 loss vs GD's 0.198 after 30 iterations.

### EXP-404: Adam vs Momentum vs GD
**Category:** comparison
**Status:** COMPLETE
**Hypothesis:** Adam combines best of momentum and adaptive scaling.
**Result:** On Rosenbrock: Adam reaches minimum in 1455 iterations, Momentum in 2521, GD doesn't converge in 5000. Adam's per-parameter scaling is most effective on anisotropic functions.

### EXP-405: Initialization Effect on Convergence
**Category:** ablation
**Status:** COMPLETE
**Hypothesis:** Starting point matters more for non-convex than convex functions.
**Result:** Convex (quadratic): all starting points converge to same minimum (17-22 iterations). Non-convex (Rosenbrock): different starts converge to different distances (0.096-0.294 from minimum). Some starting points cause GD to diverge.

---

## Stage 5 Experiments (Planned)

### EXP-501: Linear Layer Forward/Backward Parity
**Category:** comparison
**Hypothesis:** Custom Linear matches PyTorch within 1e-5
**Planned:** Stage 5 completion (requires PyTorch optional dep)

### EXP-502: Softmax + CrossEntropy Gradient Parity
**Category:** comparison
**Hypothesis:** Gradient matches PyTorch within 1e-5
**Planned:** Stage 5 completion

### EXP-503: Embedding Gradient Sparsity
**Category:** verification
**Hypothesis:** Embedding gradient only non-zero at indexed positions
**Planned:** Stage 5 completion

### EXP-504: Attention Numerical Verification
**Category:** verification
**Hypothesis:** Scaled dot-product attention gradients verified via finite difference
**Planned:** Stage 5 completion

---

## Stage 6 Experiments (Planned)

### EXP-601: Autograd Gradient Check
**Category:** verification
**Hypothesis:** Minimal autograd matches finite difference on test functions
**Planned:** Stage 6 completion

### EXP-602: 2-Layer Network Training on XOR
**Category:** convergence
**Hypothesis:** Custom backprop solves XOR (loss → 0)
**Planned:** Stage 6 completion

### EXP-603: MNIST Subset Training
**Category:** convergence
**Hypothesis:** Training on MNIST subset reaches >90% accuracy
**Planned:** Stage 6 completion

---

## Stage 7 Experiments (Planned)

### EXP-701: Gradient Field Visualization
**Category:** visualization
**Hypothesis:** Gradient field correctly shows direction/magnitude for f(x,y)=x²+y²
**Planned:** Stage 7 completion

### EXP-702: Optimization Trajectory on Rosenbrock
**Category:** visualization
**Hypothesis:** Trajectory shows characteristic curved path to minimum
**Planned:** Stage 7 completion

### EXP-703: Loss Landscape 2D Slice
**Category:** visualization
**Hypothesis:** 2D slice reveals convex/non-convex structure correctly
**Planned:** Stage 7 completion

---

## Stage 8 Experiments (Planned)

### EXP-801: Full Network PyTorch Parity
**Category:** comparison
**Hypothesis:** Small network training loss curves match PyTorch within tolerance
**Planned:** Stage 8 completion

---

## Experiment Registry (Machine-Readable)

Maintained at: `results/experiments/index.json`

```json
{
  "experiments": [],
  "last_updated": "2026-09-15T00:00:00Z",
  "schema_version": 1
}
```

*Populated as experiments are executed.*