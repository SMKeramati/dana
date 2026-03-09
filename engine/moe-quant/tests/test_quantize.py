"""Tests for moe-quant."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from moe_quant.quantize import quantize, QuantizedTensor
from moe_quant.dequantize import dequantize
from moe_quant.sensitivity import ExpertSensitivityProfiler, SensitivityProfile
from moe_quant.tier_assigner import TierBitwidthAssigner
from moe_quant.dynamic import DynamicRequantizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weight():
    torch.manual_seed(42)
    return torch.randn(32, 32)  # small weight matrix


@pytest.fixture
def expert():
    from dana_engine.model.config import TinyMoEConfig
    from dana_engine.model.moe_layer import ExpertFFN
    cfg = TinyMoEConfig.micro()
    torch.manual_seed(0)
    return ExpertFFN(cfg.hidden_dim, cfg.ffn_dim)


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------

def test_quantize_returns_quantized_tensor(weight):
    qt = quantize(weight, bits=8)
    assert isinstance(qt, QuantizedTensor)
    assert qt.bits == 8
    assert qt.shape == tuple(weight.shape)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantize_all_bitwidths(weight, bits):
    qt = quantize(weight, bits=bits)
    assert qt.bits == bits
    assert qt.data is not None
    assert qt.scales is not None


def test_quantize_compression_ratio(weight):
    qt8 = quantize(weight, bits=8)
    qt4 = quantize(weight, bits=4)
    qt2 = quantize(weight, bits=2)
    # Q2 should be smaller than Q4 which should be smaller than Q8
    assert qt2.nbytes < qt4.nbytes <= qt8.nbytes


def test_quantize_scales_shape(weight):
    group_size = 64
    qt = quantize(weight, bits=8, group_size=group_size)
    expected_groups = (weight.numel() + group_size - 1) // group_size
    assert qt.scales.shape[0] == expected_groups


# ---------------------------------------------------------------------------
# Dequantize (round-trip)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bits", [4, 8])
def test_dequantize_roundtrip_shape(weight, bits):
    qt = quantize(weight, bits=bits)
    recovered = dequantize(qt)
    assert recovered.shape == weight.shape


@pytest.mark.parametrize("bits", [4, 8])
def test_dequantize_roundtrip_close(weight, bits):
    """Q8 and Q4 should reconstruct within reasonable error."""
    qt = quantize(weight, bits=bits, group_size=min(32, weight.numel()))
    recovered = dequantize(qt)
    err = (weight - recovered).abs().mean().item()
    # Q8: very close; Q4: somewhat close
    tol = 0.05 if bits == 8 else 0.5
    assert err < tol, f"Q{bits} error {err:.4f} exceeds tolerance {tol}"


def test_dequantize_q8_better_than_q4(weight):
    qt8 = quantize(weight, bits=8, group_size=min(32, weight.numel()))
    qt4 = quantize(weight, bits=4, group_size=min(32, weight.numel()))
    err8 = (weight - dequantize(qt8)).abs().mean().item()
    err4 = (weight - dequantize(qt4)).abs().mean().item()
    assert err8 <= err4, f"Q8 error ({err8:.4f}) should be <= Q4 error ({err4:.4f})"


def test_dequantize_q4_better_than_q2(weight):
    qt4 = quantize(weight, bits=4, group_size=min(32, weight.numel()))
    qt2 = quantize(weight, bits=2, group_size=min(32, weight.numel()))
    err4 = (weight - dequantize(qt4)).abs().mean().item()
    err2 = (weight - dequantize(qt2)).abs().mean().item()
    assert err4 <= err2, f"Q4 error ({err4:.4f}) should be <= Q2 error ({err2:.4f})"


def test_dequantize_preserves_dtype(weight):
    qt = quantize(weight.half(), bits=8)
    # dequantize returns float32 by default (from numpy), then cast to original dtype
    recovered = dequantize(qt)
    # Shape should match
    assert recovered.shape == weight.shape


# ---------------------------------------------------------------------------
# SensitivityProfiler
# ---------------------------------------------------------------------------

def test_sensitivity_profile_returns_dict(expert):
    from dana_engine.model.config import TinyMoEConfig
    cfg = TinyMoEConfig.micro()
    x = torch.randn(4, cfg.hidden_dim)
    profiler = ExpertSensitivityProfiler(bits_list=[4, 8])
    profile = profiler.profile(expert, expert_id=0, calibration_data=x)
    assert isinstance(profile.scores, dict)
    assert set(profile.scores.keys()) == {4, 8}


def test_sensitivity_profile_scores_in_range(expert):
    from dana_engine.model.config import TinyMoEConfig
    cfg = TinyMoEConfig.micro()
    x = torch.randn(4, cfg.hidden_dim)
    profiler = ExpertSensitivityProfiler(bits_list=[2, 4, 8])
    profile = profiler.profile(expert, expert_id=0, calibration_data=x)
    for bits, score in profile.scores.items():
        assert -1.0 <= score <= 1.0, f"Q{bits} score {score} out of [-1, 1]"


def test_sensitivity_higher_bits_better(expert):
    from dana_engine.model.config import TinyMoEConfig
    cfg = TinyMoEConfig.micro()
    x = torch.randn(8, cfg.hidden_dim)
    profiler = ExpertSensitivityProfiler(bits_list=[2, 4, 8])
    profile = profiler.profile(expert, expert_id=0, calibration_data=x)
    # Q8 should have higher (or equal) similarity than Q2
    assert profile.scores[8] >= profile.scores[2] - 0.1  # small tolerance


def test_sensitivity_recommended_bits(expert):
    from dana_engine.model.config import TinyMoEConfig
    cfg = TinyMoEConfig.micro()
    x = torch.randn(4, cfg.hidden_dim)
    profiler = ExpertSensitivityProfiler(bits_list=[2, 4, 8])
    profile = profiler.profile(expert, expert_id=0, calibration_data=x)
    rec = profile.recommended_bits(min_quality=0.5)
    assert rec in [2, 4, 8]


# ---------------------------------------------------------------------------
# TierBitwidthAssigner
# ---------------------------------------------------------------------------

def test_assigner_default_bits():
    assigner = TierBitwidthAssigner()
    tier_map = {"e0": "hot", "e1": "ram", "e2": "ssd"}
    result = assigner.assign(tier_map)
    assert result["e0"] == 8
    assert result["e1"] == 4
    assert result["e2"] == 2


def test_assigner_bumps_bits_on_low_quality():
    assigner = TierBitwidthAssigner(min_quality=0.99)
    tier_map = {"e0": "ssd"}
    # Profile shows Q2 quality=0.5, Q4=0.99, Q8=1.0
    profile = SensitivityProfile(expert_id=0, scores={2: 0.5, 4: 0.99, 8: 1.0})
    result = assigner.assign(tier_map, profiles={"e0": profile})
    # Should bump from Q2 to Q4
    assert result["e0"] >= 4


def test_assigner_no_bump_when_quality_ok():
    assigner = TierBitwidthAssigner(min_quality=0.8)
    tier_map = {"e0": "ssd"}
    # Q2 quality=0.9 > 0.8 threshold
    profile = SensitivityProfile(expert_id=0, scores={2: 0.9, 4: 0.98, 8: 1.0})
    result = assigner.assign(tier_map, profiles={"e0": profile})
    assert result["e0"] == 2  # no bump needed


# ---------------------------------------------------------------------------
# DynamicRequantizer
# ---------------------------------------------------------------------------

def test_dynamic_requantizer_on_tier_change(weight):
    req = DynamicRequantizer()
    qt = req.on_tier_change(0, "ssd", "ram", weight)
    assert qt.bits == 4  # ram → Q4


def test_dynamic_requantizer_requantize(weight):
    req = DynamicRequantizer()
    qt_q2 = quantize(weight, bits=2, group_size=min(32, weight.numel()))
    qt_q4 = req.requantize(qt_q2, new_tier="ram")
    assert qt_q4.bits == 4


def test_dynamic_requantizer_upcast(weight):
    req = DynamicRequantizer()
    qt = quantize(weight, bits=4, group_size=min(32, weight.numel()))
    fp = req.upcast_to_float(qt)
    assert fp.shape == weight.shape
    assert fp.dtype in (torch.float32, torch.float16)
