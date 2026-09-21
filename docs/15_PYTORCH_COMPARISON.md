# 15_PYTORCH_COMPARISON.md — PyTorch Framework Parity

## Why PyTorch Is Introduced

PyTorch is the standard framework for neural network research and production.
By comparing our manual implementations against PyTorch, we verify:

1. Our mathematical implementations are correct
2. We understand what frameworks abstract away
3. The connection between "from scratch" math and practical frameworks

## Why PyTorch Is Optional

PyTorch is NOT required to use the core toolkit. The project implements:

- Linear algebra
- Calculus
- Probability
- Optimization
- Neural-network mathematics
- Backpropagation
- Experiments

All of these work without PyTorch. PyTorch is only needed for:

- Comparison tests
- Educational examples

## What the Project Implements Manually

| Mathematical Concept | Implementation |
|----------------------|----------------|
| Affine transformation | `affine_transform(x, W, b)` — explicit `x @ W.T + b` |
| Sigmoid | `sigmoid(x)` — numerically stable `1/(1+exp(-x))` |
| Tanh | `tanh(x)` — wraps `np.tanh` |
| ReLU | `relu(x)` — `np.maximum(0, x)` |
| GELU | `gelu(x)` — tanh approximation |
| Softmax | `softmax(x)` — numerically stable with max-subtraction |
| Cross-entropy | `cross_entropy_with_logits(logits, targets)` — log-sum-exp trick |
| MSE loss | `mean_squared_error(y_true, y_pred)` — `mean((y-y')^2)` |
| BCE loss | `binary_cross_entropy(y_true, y_pred)` — clipped log |
| Gradient check | `gradient_check(func, grad, x)` — central differences |
| Backward pass | `affine_backward`, `sigmoid_backward`, etc. |
| Training | `forward()`, `backward()`, `sgd_step()`, `train()` |
| Adam | `adam()`, `adam_step()` — bias-corrected moments |

## What PyTorch Abstracts Away

| Concept | From Scratch | PyTorch |
|---------|-------------|---------|
| Forward pass | Manual matrix multiplication | `nn.Module.forward()` |
| Backward pass | Manual chain rule implementation | `autograd.backward()` |
| Computational graph | Not stored | Built automatically |
| Gradient accumulation | Manual | Automatic via autograd |
| Parameter updates | Manual `theta -= lr * grad` | `optimizer.step()` |
| Device placement | CPU only (NumPy) | CPU/CUDA automatic |
| Mixed precision | Not applicable | `torch.cuda.amp` |
| Distribution | Not applicable | `DistributedDataParallel` |

## Concept Mapping

| Project Concept | PyTorch Equivalent |
|-----------------|---------------------|
| `affine_transform(x, W, b)` | `torch.nn.Linear(in, out)` or `x @ W.T + b` |
| `sigmoid(x)` | `torch.sigmoid(x)` |
| `tanh(x)` | `torch.tanh(x)` |
| `relu(x)` | `torch.relu(x)` |
| `gelu(x)` | `torch.nn.functional.gelu(x, approximate="tanh")` |
| `softmax(x)` | `torch.softmax(x, dim=-1)` |
| `mean_squared_error(a, b)` | `torch.nn.functional.mse_loss(a, b, reduction="mean")` |
| `binary_cross_entropy(a, b)` | `torch.nn.functional.binary_cross_entropy(a, b)` |
| `cross_entropy_with_logits(z, y)` | `torch.nn.functional.cross_entropy(z, y)` |
| `gradient_check(f, g, x)` | `torch.autograd.grad(f(x), x)` |
| Manual backward pass | `loss.backward()` |
| `sgd_step(params, grads, lr)` | `torch.optim.SGD(params, lr).step()` |
| `adam(params, grads, ...)` | `torch.optim.Adam(params, lr).step()` |

## Forward-Pass Comparison

Our affine transformation:
```
z = x @ W.T + b  (batch-first convention)
```

PyTorch's nn.Linear:
```
z = x @ W.T + b  (same mathematics, different API)
```

Numerical parity: Forward output max error is typically < 1e-15 (machine epsilon for float64).

## Gradient Comparison

Our manual backward pass:
```
dL/dW = dout.T @ x
dL/db = sum(dout, axis=0)
dL/dx = dout @ W
```

PyTorch autograd computes the same gradients by building a computational graph
and applying the chain rule in reverse order. The numerical results are identical
within floating-point precision.

**Key identity verified:**
```
analytical gradient ≈ numerical gradient ≈ PyTorch autograd gradient
```

All three agree within appropriate numerical tolerance (< 1e-10 for float64).

## Numerical Tolerance Policy

| dtype | Absolute Tolerance | Relative Tolerance | Rationale |
|-------|-------------------|-------------------|-----------|
| float64 | 1e-8 | 1e-5 | Machine epsilon ~2.2e-16, accumulate ~100 operations |
| float32 | 1e-5 | 1e-3 | Machine epsilon ~1.2e-7, more accumulation error |

Comparison criterion: `|a - b| <= atol + rtol * |b|`

## dtype Considerations

- Our toolkit always operates in float64 (NumPy default)
- PyTorch defaults to float32 for neural networks
- All comparison tests use `dtype=torch.float64` for parity
- float32 comparisons use relaxed tolerances

## CPU/Device Considerations

All comparison tests run on CPU. This is intentional:

1. Deterministic results (no CUDA non-determinism)
2. Simpler debugging
3. No GPU hardware required
4. Same mathematics regardless of device

If PyTorch is installed with GPU support, comparisons still run on CPU
for reproducibility.

## Known Implementation Differences

| Aspect | Our Implementation | PyTorch |
|--------|-------------------|---------|
| softmax_backward | Returns `probs - targets` (no 1/n) | Includes 1/n for `reduction="mean"` |
| Cross-entropy eps | Clips to [1e-15, inf) | Uses log-sum-exp internally |
| GELU | tanh approximation only | Supports `approximate="tanh"` and exact |
| Adam bias correction | Uses `t+1` for 1-indexed steps | Same convention |
| Convolution | Not implemented | `nn.Conv2d` etc. |

These are documented convention differences, not bugs.

## What This Teaches About Frameworks

1. **Frameworks are mathematical automation.** PyTorch's autograd does exactly what our manual backward pass does — it just builds the graph automatically.

2. **The math is the same.** Whether you write `x @ W.T + b` in NumPy or call `nn.Linear(x)`, the underlying operation is identical.

3. **Frameworks add engineering, not math.** PyTorch adds GPU support, automatic differentiation, distributed training, and optimization. The mathematics is unchanged.

4. **Understanding the math makes you a better framework user.** When you know what `loss.backward()` actually computes, you can debug gradient issues, understand training failures, and design better architectures.

## Running Comparison Tests

```bash
# Run all comparison tests (requires PyTorch)
python -m pytest tests/comparison/ -v

# Run specific comparison
python -m pytest tests/comparison/test_pytorch_linear.py -v
python -m pytest tests/comparison/test_pytorch_activations.py -v
python -m pytest tests/comparison/test_pytorch_gradients.py -v

# Run without PyTorch (comparison tests skip cleanly)
python -m pytest tests/ -v
```
