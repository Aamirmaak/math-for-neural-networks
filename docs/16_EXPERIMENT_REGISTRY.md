# 16_EXPERIMENT_REGISTRY.md — Experiment Registry

## Status Legend
- **EXECUTED** — Experiment ran successfully with actual results
- **VERIFIED** — Results independently confirmed
- **PLANNED** — Not yet implemented

## Numbered Experiments (Stage 7)

| # | Experiment | Concept | Status | Command |
|---|-----------|---------|--------|---------|
| 01 | Vector Geometry | Linear Algebra | EXECUTED | `python experiments/01_vector_geometry.py` |
| 02 | Matrix Transformations | Linear Algebra | EXECUTED | `python experiments/02_matrix_transformations.py` |
| 03 | Derivative Accuracy | Calculus | EXECUTED | `python experiments/03_derivative_accuracy.py` |
| 04 | Gradient Field | Calculus | EXECUTED | `python experiments/04_gradient_field.py` |
| 05 | Learning Rate Effect | Optimization | EXECUTED | `python experiments/05_learning_rate.py` |
| 06 | Optimizer Comparison | Optimization | EXECUTED | `python experiments/06_optimizer_comparison.py` |
| 07 | Activation Functions | Neural Networks | EXECUTED | `python experiments/07_activation_functions.py` |
| 08 | Softmax Stability | Neural Networks | EXECUTED | `python experiments/08_softmax_stability.py` |
| 09 | Cross-Entropy | Neural Networks | EXECUTED | `python experiments/09_cross_entropy.py` |
| 10 | Backpropagation | Backpropagation | EXECUTED | `python experiments/10_backpropagation.py` |
| 11 | Gradient Checking | Backpropagation | EXECUTED | `python experiments/11_gradient_checking.py` |
| 12 | Toy Training | Backpropagation | EXECUTED | `python experiments/12_toy_training.py` |
| 13 | Gradient Flow | Backpropagation | EXECUTED | `python experiments/13_gradient_flow.py` |
| 14 | Vanishing Gradients | Backpropagation | EXECUTED | `python experiments/14_vanishing_exploding.py` |
| 15 | Embedding Geometry | Integration | EXECUTED | `python experiments/15_embedding_geometry.py` |
| 16 | Attention | Integration | EXECUTED | `python experiments/16_attention.py` |
| 17 | Attention Scaling | Integration | EXECUTED | `python experiments/17_attention_scaling.py` |
| 18 | Integrated Demo | Integration | EXECUTED | `python experiments/18_integrated_demo.py` |

## Standalone Experiments (Legacy, Unique Topics)

| File | Concept | Status |
|------|---------|--------|
| activation_gradients.py | Activation gradient magnitudes | EXECUTED |
| contour_path.py | GD path visualization on contours | EXECUTED |
| entropy_categorical.py | Entropy of categorical distributions | EXECUTED |
| gradient_norm_training.py | Gradient norm as convergence indicator | EXECUTED |
| init_effect.py | Initialization effect on convergence | EXECUTED |
| lr_sensitivity.py | Fine-grained learning rate sweep | EXECUTED |
| mle_convergence.py | MLE convergence with sample size | EXECUTED |
| softmax_ce_gradient.py | Softmax+CE gradient identity | EXECUTED |
| step_size_sensitivity.py | Step-size accuracy tradeoff | EXECUTED |

## PyTorch Comparison Tests (Stage 8)

| Test File | Concept | Status |
|-----------|---------|--------|
| test_pytorch_linear.py | Affine transformation parity | EXECUTED |
| test_pytorch_activations.py | Activation function parity | EXECUTED |
| test_pytorch_softmax.py | Softmax parity | EXECUTED |
| test_pytorch_losses.py | Loss function parity | EXECUTED |
| test_pytorch_gradients.py | Gradient computation parity | EXECUTED |
| test_pytorch_network.py | Small network parity | EXECUTED |
| test_pytorch_optimizers.py | Optimizer update parity | EXECUTED |

## Educational Examples

| File | Concept | Status |
|------|---------|--------|
| examples/pytorch_comparison/from_scratch_vs_pytorch.py | Side-by-side comparison | EXECUTED |

## How to Run All Experiments

```bash
# Run all numbered experiments
for i in $(seq -w 1 18); do
    python experiments/${i}_*.py
done

# Run PyTorch comparison tests
python -m pytest tests/comparison/ -v

# Run mathematical property tests
python -m pytest tests/test_mathematical_properties.py -v

# Run full test suite
python -m pytest tests/ -v
```
