"""Per-group quantization for MoE expert weights.

Supports Q2, Q4, Q8 (bits = 2, 4, 8). Uses symmetric per-group quantization:
  - Compute scale = max(|x|) / (2^(bits-1) - 1) per group
  - Quantize: q = round(x / scale)
  - Store q as packed bytes (Q2: 4 per byte, Q4: 2 per byte, Q8: 1 per byte)

This is intentionally a pure numpy/torch implementation (no CUDA required),
correct but not throughput-optimized. See GPU_TODO.md for CUDA dequant kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QuantizedTensor:
    """A quantized tensor with metadata needed for dequantization."""

    data: np.ndarray        # packed integer data (uint8 array)
    scales: torch.Tensor    # (num_groups,) float32 scale per group
    bits: int               # 2, 4, or 8
    shape: tuple            # original tensor shape
    dtype: torch.dtype      # original float dtype
    group_size: int         # elements per group

    @property
    def nbytes(self) -> int:
        return self.data.nbytes + self.scales.element_size() * self.scales.numel()

    def compression_ratio(self) -> float:
        original = 4 * 1  # float32 = 4 bytes
        quantized = self.bits / 8
        return original / quantized


def quantize(
    tensor: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> QuantizedTensor:
    """Quantize a float tensor to Q2, Q4, or Q8.

    Args:
        tensor: Float tensor to quantize (any shape)
        bits: Target bit-width: 2, 4, or 8
        group_size: Number of elements per quantization group

    Returns:
        QuantizedTensor with packed data and per-group scales
    """
    assert bits in (2, 4, 8), f"Unsupported bits={bits}. Use 2, 4, or 8."

    original_shape = tuple(tensor.shape)
    original_dtype = tensor.dtype

    x = tensor.detach().float().cpu().numpy().flatten()  # (N,)
    N = len(x)

    # Pad to multiple of group_size
    pad = (group_size - N % group_size) % group_size
    if pad > 0:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])

    num_groups = len(x) // group_size
    x_grouped = x.reshape(num_groups, group_size)  # (G, group_size)

    # Per-group symmetric quantization
    max_val = np.abs(x_grouped).max(axis=1, keepdims=True)  # (G, 1)
    max_val = np.maximum(max_val, 1e-8)                       # avoid div by zero
    q_max = float(2 ** (bits - 1) - 1)                       # 127, 7, or 1
    scales = (max_val[:, 0] / q_max).astype(np.float32)       # (G,)

    # Quantize: clip to [-q_max, q_max] and round
    q_grouped = np.round(x_grouped / max_val * q_max).astype(np.int32)
    q_grouped = np.clip(q_grouped, -int(q_max), int(q_max))  # (G, group_size)

    # Pack into bytes
    q_flat = q_grouped.flatten()[:N]  # trim padding
    packed = _pack(q_flat, bits)

    return QuantizedTensor(
        data=packed,
        scales=torch.from_numpy(scales),
        bits=bits,
        shape=original_shape,
        dtype=original_dtype,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def _pack(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack signed int array into uint8 bytes."""
    if bits == 8:
        return (q.astype(np.int8)).view(np.uint8)

    # Offset to unsigned: Q4 range [-7,7] → [0,14], Q2 range [-1,1] → [0,2]
    # We use 2s-complement style: map signed to unsigned via masking
    if bits == 4:
        # 4-bit: 2 values per byte
        unsigned = (q.astype(np.int8) & 0x0F).astype(np.uint8)  # mask to 4 bits
        n = len(unsigned)
        # Pad to even length
        if n % 2 != 0:
            unsigned = np.append(unsigned, np.uint8(0))
        packed = (unsigned[0::2]) | (unsigned[1::2] << 4)
        return packed

    if bits == 2:
        # 2-bit: 4 values per byte
        unsigned = (q.astype(np.int8) & 0x03).astype(np.uint8)  # mask to 2 bits
        n = len(unsigned)
        pad = (4 - n % 4) % 4
        if pad:
            unsigned = np.append(unsigned, np.zeros(pad, dtype=np.uint8))
        packed = (unsigned[0::4]
                  | (unsigned[1::4] << 2)
                  | (unsigned[2::4] << 4)
                  | (unsigned[3::4] << 6))
        return packed

    raise ValueError(f"Unsupported bits={bits}")


def _unpack(packed: np.ndarray, bits: int, original_n: int) -> np.ndarray:
    """Unpack uint8 bytes to signed int array of length original_n."""
    if bits == 8:
        return packed.view(np.int8).astype(np.int32)[:original_n]

    if bits == 4:
        low = (packed & 0x0F).astype(np.uint8)
        high = ((packed >> 4) & 0x0F).astype(np.uint8)
        interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
        interleaved[0::2] = low
        interleaved[1::2] = high
        # Sign-extend from 4 bits
        signed = interleaved.astype(np.int8)
        signed = np.where(signed > 7, signed - 16, signed)
        return signed.astype(np.int32)[:original_n]

    if bits == 2:
        b0 = (packed & 0x03).astype(np.uint8)
        b1 = ((packed >> 2) & 0x03).astype(np.uint8)
        b2 = ((packed >> 4) & 0x03).astype(np.uint8)
        b3 = ((packed >> 6) & 0x03).astype(np.uint8)
        interleaved = np.empty(len(packed) * 4, dtype=np.uint8)
        interleaved[0::4] = b0
        interleaved[1::4] = b1
        interleaved[2::4] = b2
        interleaved[3::4] = b3
        # Sign-extend from 2 bits: 0,1 → 0,1; 2,3 → -2,-1
        signed = interleaved.astype(np.int8)
        signed = np.where(signed > 1, signed - 4, signed)
        return signed.astype(np.int32)[:original_n]

    raise ValueError(f"Unsupported bits={bits}")
