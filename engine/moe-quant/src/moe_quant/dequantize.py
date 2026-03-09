"""Dequantization — reverse of quantize().

Reconstructs a float tensor from a QuantizedTensor.
Vectorized numpy operations for speed; all runs on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from moe_quant.quantize import QuantizedTensor, _unpack


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """Reconstruct a float tensor from a QuantizedTensor.

    Args:
        qt: QuantizedTensor produced by quantize()

    Returns:
        Float tensor of shape qt.shape, dtype qt.dtype
    """
    original_n = 1
    for s in qt.shape:
        original_n *= s

    # Unpack integers
    q_flat = _unpack(qt.data, qt.bits, original_n)  # (N,) int32

    # Reshape to (num_groups, group_size) for scale application
    num_groups = len(qt.scales)
    group_size = qt.group_size
    total_padded = num_groups * group_size

    q_padded = np.zeros(total_padded, dtype=np.float32)
    q_padded[:original_n] = q_flat.astype(np.float32)
    q_grouped = q_padded.reshape(num_groups, group_size)  # (G, group_size)

    # Dequantize: multiply by scale
    scales_np = qt.scales.numpy().reshape(-1, 1)          # (G, 1)
    x_grouped = q_grouped * scales_np                      # (G, group_size)

    # Flatten and trim padding
    x_flat = x_grouped.flatten()[:original_n]

    # Reshape to original shape
    tensor = torch.from_numpy(x_flat.reshape(qt.shape))
    return tensor.to(qt.dtype)
