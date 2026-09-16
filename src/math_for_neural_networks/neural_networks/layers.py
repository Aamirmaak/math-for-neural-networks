"""
Linear Layer Mathematics
========================

The fundamental building block of neural networks is the affine transformation:

    z = Wx + b

Where:
    W = weight matrix (parameters learned during training)
    x = input vector
    b = bias vector
    z = pre-activation output

Shape conventions (batch-first):
    x: (input_features,) or (batch_size, input_features)
    W: (output_features, input_features)
    b: (output_features,)
    z: (output_features,) or (batch_size, output_features)

The output dimension is determined by the number of rows of W (output_features).
The input dimension is determined by the number of columns of W (input_features).

Why bias?
    Without bias, the affine transformation is z = Wx (linear).
    Bias allows shifting the output, enabling the model to fit data
    that doesn't pass through the origin.

Neural network connection:
    A linear/dense layer is fundamentally matrix multiplication plus bias.
    Every layer in a neural network performs this operation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def affine_transform(x: NDArray, W: NDArray, b: NDArray | None = None) -> NDArray:
    """Compute the affine transformation z = Wx + b.

    Supports single samples (1D input) and batched inputs (2D input).

    Shape conventions:
        x: (input_features,) or (batch_size, input_features)
        W: (output_features, input_features)
        b: (output_features,) or None
        output: (output_features,) or (batch_size, output_features)

    Args:
        x: Input vector or batch of inputs.
        W: Weight matrix, shape (output_features, input_features).
        b: Bias vector, shape (output_features,). Optional (defaults to zero).

    Returns:
        Transformed output z = Wx + b.

    Raises:
        ValueError: If shapes are incompatible.
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if W.ndim != 2:
        raise ValueError(f"W must be 2D, got {W.ndim}D")

    output_features, input_features = W.shape

    if x.ndim == 1:
        # Single sample: x is (input_features,)
        if x.shape[0] != input_features:
            raise ValueError(
                f"Input features mismatch: x has {x.shape[0]}, W expects {input_features}"
            )
        z = W @ x
    elif x.ndim == 2:
        # Batch: x is (batch_size, input_features)
        if x.shape[1] != input_features:
            raise ValueError(
                f"Input features mismatch: x has {x.shape[1]} features, W expects {input_features}"
            )
        z = x @ W.T
    else:
        raise ValueError(f"x must be 1D or 2D, got {x.ndim}D")

    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.ndim != 1:
            raise ValueError(f"b must be 1D, got {b.ndim}D")
        if b.shape[0] != output_features:
            raise ValueError(
                f"Bias size mismatch: b has {b.shape[0]}, W has {output_features} rows"
            )
        z = z + b

    result: NDArray = z
    return result
