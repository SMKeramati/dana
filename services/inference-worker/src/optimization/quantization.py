"""Quantization helpers for dynamic precision switching.

Provides utilities to quantize and dequantize weight tensors between
float16, int8, and int4 representations.  Supports per-channel and
per-tensor symmetric quantization with scale factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class QuantPrecision(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizedTensor:
    """A quantized tensor with its scale factors."""

    data: np.ndarray
    scales: np.ndarray
    precision: QuantPrecision
    original_shape: tuple[int, ...]
    per_channel: bool = False


def quantize_symmetric(
    tensor: np.ndarray,
    precision: QuantPrecision = QuantPrecision.INT8,
    per_channel: bool = False,
) -> QuantizedTensor:
    """Symmetric quantization of a float tensor.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (float16 or float32).
    precision : QuantPrecision
        Target precision.
    per_channel : bool
        If True, compute separate scale factors along axis 0.
    """
    tensor_f32 = tensor.astype(np.float32)
    original_shape = tensor.shape

    if precision == QuantPrecision.FP16:
        return QuantizedTensor(
            data=tensor.astype(np.float16),
            scales=np.array([1.0], dtype=np.float32),
            precision=precision,
            original_shape=original_shape,
        )

    if precision == QuantPrecision.INT8:
        qmin, qmax = -127, 127
    elif precision == QuantPrecision.INT4:
        qmin, qmax = -7, 7
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    if per_channel and tensor_f32.ndim >= 2:
        # Scale per output channel (axis 0)
        abs_max = np.abs(tensor_f32).max(axis=tuple(range(1, tensor_f32.ndim)), keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        scales = abs_max / qmax
        quantized = np.clip(np.round(tensor_f32 / scales), qmin, qmax).astype(np.int8)
        scales = scales.squeeze()
    else:
        abs_max = np.abs(tensor_f32).max()
        abs_max = max(abs_max, 1e-8)
        scale = abs_max / qmax
        quantized = np.clip(np.round(tensor_f32 / scale), qmin, qmax).astype(np.int8)
        scales = np.array([scale], dtype=np.float32)

    return QuantizedTensor(
        data=quantized,
        scales=scales,
        precision=precision,
        original_shape=original_shape,
        per_channel=per_channel,
    )


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Dequantize back to float32."""
    if qt.precision == QuantPrecision.FP16:
        return qt.data.astype(np.float32)

    data_f32 = qt.data.astype(np.float32)

    if qt.per_channel and qt.scales.ndim >= 1 and qt.scales.shape[0] > 1:
        # Broadcast scales along axis 0
        shape = [qt.scales.shape[0]] + [1] * (data_f32.ndim - 1)
        scales = qt.scales.reshape(shape)
        return data_f32 * scales
    else:
        return data_f32 * qt.scales[0]


def compute_quantization_error(
    original: np.ndarray,
    precision: QuantPrecision,
    per_channel: bool = False,
) -> float:
    """Compute the mean squared error introduced by quantization."""
    qt = quantize_symmetric(original, precision=precision, per_channel=per_channel)
    reconstructed = dequantize(qt)
    original_f32 = original.astype(np.float32)
    mse = float(np.mean((original_f32 - reconstructed) ** 2))
    return mse


def dynamic_precision_select(
    tensor: np.ndarray,
    error_threshold: float = 0.001,
) -> QuantPrecision:
    """Select the most aggressive quantization that stays within error threshold.

    Tries INT4 first, then INT8, falling back to FP16.
    """
    for prec in (QuantPrecision.INT4, QuantPrecision.INT8):
        err = compute_quantization_error(tensor, precision=prec, per_channel=True)
        if err <= error_threshold:
            return prec
    return QuantPrecision.FP16
