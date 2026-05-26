"""Phase 1 tests: Tiny MoE model correctness.

All tests run on CPU, complete in <5s, no downloads required.
"""

from __future__ import annotations

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.moe_layer import ExpertFFN, MoELayer, MoERouter
from dana_engine.model.attention import CausalSelfAttention
from dana_engine.model.transformer import TinyMoETransformer
from dana_engine.naive_inference import greedy_generate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> TinyMoEConfig:
    return TinyMoEConfig.micro()   # smallest possible: 2L, 4E, 64-dim


@pytest.fixture(scope="module")
def model(cfg: TinyMoEConfig) -> TinyMoETransformer:
    torch.manual_seed(42)
    return TinyMoETransformer(cfg)


@pytest.fixture
def dummy_ids(cfg: TinyMoEConfig) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (1, 8))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults() -> None:
    cfg = TinyMoEConfig.tiny()
    assert cfg.num_layers == 4
    assert cfg.num_experts == 8
    assert cfg.num_active == 2
    assert cfg.head_dim == cfg.hidden_dim // cfg.num_heads


def test_config_derived_head_dim() -> None:
    cfg = TinyMoEConfig(hidden_dim=256, num_heads=8)
    assert cfg.head_dim == 32


def test_config_validation() -> None:
    with pytest.raises(AssertionError):
        TinyMoEConfig(hidden_dim=256, num_heads=7)  # not divisible

    with pytest.raises(AssertionError):
        TinyMoEConfig(num_experts=4, num_active=8)  # active > total


# ---------------------------------------------------------------------------
# ExpertFFN
# ---------------------------------------------------------------------------

def test_expert_ffn_shape(cfg: TinyMoEConfig) -> None:
    expert = ExpertFFN(cfg.hidden_dim, cfg.ffn_dim)
    x = torch.randn(5, cfg.hidden_dim)
    out = expert(x)
    assert out.shape == (5, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# MoERouter
# ---------------------------------------------------------------------------

def test_router_output_shape(cfg: TinyMoEConfig) -> None:
    router = MoERouter(cfg.hidden_dim, cfg.num_experts, cfg.num_active)
    x = torch.randn(2, 10, cfg.hidden_dim)   # batch=2, seq=10
    out = router(x)
    assert out.indices.shape == (2, 10, cfg.num_active)
    assert out.weights.shape == (2, 10, cfg.num_active)
    assert out.logits.shape == (2, 10, cfg.num_experts)


def test_router_selects_exactly_k_experts(cfg: TinyMoEConfig) -> None:
    router = MoERouter(cfg.hidden_dim, cfg.num_experts, cfg.num_active)
    x = torch.randn(1, 5, cfg.hidden_dim)
    out = router(x)
    # Each token should have exactly num_active distinct expert indices
    for b in range(1):
        for t in range(5):
            ids = out.indices[b, t]
            assert len(ids.unique()) == cfg.num_active


def test_router_weights_sum_to_one(cfg: TinyMoEConfig) -> None:
    router = MoERouter(cfg.hidden_dim, cfg.num_experts, cfg.num_active)
    x = torch.randn(2, 6, cfg.hidden_dim)
    out = router(x)
    sums = out.weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

def test_moe_layer_output_shape(cfg: TinyMoEConfig) -> None:
    layer = MoELayer(cfg)
    x = torch.randn(2, 8, cfg.hidden_dim)
    out, router_logits = layer(x)
    assert out.shape == x.shape
    assert router_logits.shape == (2, 8, cfg.num_experts)


def test_moe_layer_router_logits_finite(cfg: TinyMoEConfig) -> None:
    layer = MoELayer(cfg)
    x = torch.randn(1, 4, cfg.hidden_dim)
    _, router_logits = layer(x)
    assert torch.isfinite(router_logits).all()


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------

def test_attention_output_shape(cfg: TinyMoEConfig) -> None:
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 10, cfg.hidden_dim)
    out, weights = attn(x)
    assert out.shape == (2, 10, cfg.hidden_dim)
    assert weights.shape == (2, cfg.num_heads, 10, 10)


def test_attention_causal_mask(cfg: TinyMoEConfig) -> None:
    """Upper-triangular attention weights should be 0 (masked out)."""
    attn = CausalSelfAttention(cfg)
    x = torch.randn(1, 6, cfg.hidden_dim)
    _, weights = attn(x)
    # Average over heads; upper triangle (i < j) should be ~0
    avg_weights = weights[0].mean(dim=0)   # (T, T)
    upper = torch.triu(avg_weights, diagonal=1)
    assert upper.abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# TinyMoETransformer
# ---------------------------------------------------------------------------

def test_transformer_forward_shape(model: TinyMoETransformer, dummy_ids: torch.Tensor, cfg: TinyMoEConfig) -> None:
    out = model(dummy_ids)
    B, T = dummy_ids.shape
    assert out.logits.shape == (B, T, cfg.vocab_size)
    assert len(out.all_router_logits) == cfg.num_layers


def test_transformer_router_logits_per_layer(model: TinyMoETransformer, dummy_ids: torch.Tensor, cfg: TinyMoEConfig) -> None:
    out = model(dummy_ids)
    B, T = dummy_ids.shape
    for layer_logits in out.all_router_logits:
        assert layer_logits.shape == (B, T, cfg.num_experts)


def test_transformer_logits_finite(model: TinyMoETransformer, dummy_ids: torch.Tensor) -> None:
    out = model(dummy_ids)
    assert torch.isfinite(out.logits).all()
    for rl in out.all_router_logits:
        assert torch.isfinite(rl).all()


def test_transformer_num_parameters(cfg: TinyMoEConfig) -> None:
    model = TinyMoETransformer(cfg)
    n = model.num_parameters()
    assert n > 0
    # Micro model should be small
    assert n < 5_000_000


def test_transformer_expert_weights_accessible(model: TinyMoETransformer, cfg: TinyMoEConfig) -> None:
    weights = model.expert_weights(layer=0, expert_id=0)
    assert len(weights) == 2
    assert weights[0].shape == (cfg.ffn_dim, cfg.hidden_dim)
    assert weights[1].shape == (cfg.hidden_dim, cfg.ffn_dim)


def test_transformer_hidden_states_returned(model: TinyMoETransformer, dummy_ids: torch.Tensor, cfg: TinyMoEConfig) -> None:
    out = model(dummy_ids, return_hidden_states=True)
    assert len(out.all_hidden_states) == cfg.num_layers


# ---------------------------------------------------------------------------
# Naive inference
# ---------------------------------------------------------------------------

def test_greedy_generate_length(model: TinyMoETransformer, cfg: TinyMoEConfig) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    result = greedy_generate(model, prompt, max_new_tokens=8)
    assert result.tokens.shape == (1, 4 + 8)
    assert result.num_new_tokens == 8
    assert result.steps == 8


def test_greedy_generate_router_logits_captured(model: TinyMoETransformer, cfg: TinyMoEConfig) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    result = greedy_generate(model, prompt, max_new_tokens=5)
    # One entry per step, each has num_layers router logit tensors
    assert len(result.all_router_logits) == 5
    assert len(result.all_router_logits[0]) == cfg.num_layers


def test_greedy_generate_deterministic(model: TinyMoETransformer, cfg: TinyMoEConfig) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    r1 = greedy_generate(model, prompt, max_new_tokens=6)
    r2 = greedy_generate(model, prompt, max_new_tokens=6)
    assert torch.equal(r1.tokens, r2.tokens)


def test_greedy_generate_tokens_per_step(model: TinyMoETransformer, cfg: TinyMoEConfig) -> None:
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    result = greedy_generate(model, prompt, max_new_tokens=10)
    assert result.tokens_per_step == 1.0   # naive: always 1 token/step
