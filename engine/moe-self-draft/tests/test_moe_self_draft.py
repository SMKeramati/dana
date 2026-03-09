"""Tests for moe-self-draft package.

Tests: MoeSelfDrafter, RouterLogitExtractor, SelfDraftVerifier.
All run on CPU with the synthetic tiny MoE model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../dana-engine/src"))

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer

from moe_self_draft.self_draft import MoeSelfDrafter, DraftResult
from moe_self_draft.logit_extractor import RouterLogitExtractor
from moe_self_draft.verify import SelfDraftVerifier


@pytest.fixture(scope="module")
def config():
    return TinyMoEConfig.micro()


@pytest.fixture(scope="module")
def model(config):
    m = TinyMoETransformer(config)
    m.eval()
    return m


@pytest.fixture
def input_ids(config):
    return torch.randint(0, config.vocab_size, (1, 5))


# ------------------------------------------------------------------
# RouterLogitExtractor
# ------------------------------------------------------------------

class TestRouterLogitExtractor:
    def test_attach_registers_hooks(self, model):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        assert extractor._attached
        assert len(extractor._hooks) > 0
        extractor.detach()

    def test_capture_logits_on_forward(self, model, input_ids, config):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        extractor.clear()
        model(input_ids)
        logits = extractor.get_logits()
        # One entry per MoE layer
        assert len(logits) == config.num_layers
        extractor.detach()

    def test_logit_shape(self, model, input_ids, config):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        extractor.clear()
        model(input_ids)
        logits = extractor.get_logits()
        for layer_logits in logits:
            # (batch, seq, num_experts)
            assert layer_logits.shape[0] == 1
            assert layer_logits.shape[1] == input_ids.shape[1]
            assert layer_logits.shape[2] == config.num_experts
        extractor.detach()

    def test_clear_removes_captured(self, model, input_ids):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        model(input_ids)
        assert len(extractor.get_logits()) > 0
        extractor.clear()
        assert len(extractor.get_logits()) == 0
        extractor.detach()

    def test_get_top_experts(self, model, input_ids, config):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        extractor.clear()
        model(input_ids)
        top = extractor.get_top_experts(k=2)
        assert len(top) == config.num_layers
        for layer_top in top:
            assert len(layer_top) == 2
            for eid in layer_top:
                assert 0 <= eid < config.num_experts
        extractor.detach()

    def test_detach_removes_hooks(self, model, input_ids):
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        extractor.detach()
        extractor.clear()
        model(input_ids)
        # After detach, no new logits should be captured
        assert len(extractor.get_logits()) == 0


# ------------------------------------------------------------------
# MoeSelfDrafter
# ------------------------------------------------------------------

class TestMoeSelfDrafter:
    def test_draft_returns_draft_result(self, model, input_ids):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=3)
        assert isinstance(result, DraftResult)

    def test_draft_token_count(self, model, input_ids):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=5)
        assert len(result.draft_tokens) == 5

    def test_draft_tokens_are_valid(self, model, input_ids, config):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=5)
        for tok in result.draft_tokens:
            assert 0 <= tok < config.vocab_size

    def test_router_logits_captured(self, model, input_ids, config):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=3)
        # router_logits is [step][layer]
        assert len(result.router_logits) == 3
        for step_logits in result.router_logits:
            assert len(step_logits) == config.num_layers

    def test_router_restored_after_draft(self, model, input_ids, config):
        """num_active must be restored after drafting."""
        drafter = MoeSelfDrafter(model, num_active_override=1)
        drafter.draft(input_ids, num_draft_tokens=3)
        # Check all routers are back to original num_active
        for block in model.blocks:
            assert block.moe.router.num_active == config.num_active

    def test_predicted_experts_nonempty(self, model, input_ids):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=5)
        experts = result.predicted_experts()
        assert len(experts) > 0

    def test_predicted_experts_valid(self, model, input_ids, config):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=5)
        for eid in result.predicted_experts():
            assert 0 <= eid < config.num_experts

    def test_draft_one_token(self, model, input_ids):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        result = drafter.draft(input_ids, num_draft_tokens=1)
        assert len(result.draft_tokens) == 1

    def test_draft_consistent(self, model, input_ids):
        """Same input → same draft (deterministic greedy)."""
        drafter = MoeSelfDrafter(model, num_active_override=1)
        r1 = drafter.draft(input_ids.clone(), num_draft_tokens=3)
        r2 = drafter.draft(input_ids.clone(), num_draft_tokens=3)
        assert r1.draft_tokens == r2.draft_tokens


# ------------------------------------------------------------------
# SelfDraftVerifier
# ------------------------------------------------------------------

class TestSelfDraftVerifier:
    def test_verify_returns_nonempty(self, model, input_ids):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        draft_result = drafter.draft(input_ids, num_draft_tokens=5)
        verifier = SelfDraftVerifier(model)
        accepted = verifier.verify(input_ids, draft_result)
        assert isinstance(accepted, list)
        assert len(accepted) >= 1

    def test_accepted_tokens_are_valid(self, model, input_ids, config):
        drafter = MoeSelfDrafter(model, num_active_override=1)
        draft_result = drafter.draft(input_ids, num_draft_tokens=5)
        verifier = SelfDraftVerifier(model)
        accepted = verifier.verify(input_ids, draft_result)
        for tok in accepted:
            assert 0 <= tok < config.vocab_size

    def test_verify_empty_draft_returns_one_token(self, model, input_ids):
        empty_draft = DraftResult(draft_tokens=[], router_logits=[])
        verifier = SelfDraftVerifier(model)
        accepted = verifier.verify(input_ids, empty_draft)
        assert len(accepted) == 1

    def test_verify_consistent(self, model, input_ids):
        """Same inputs → same accepted tokens."""
        drafter = MoeSelfDrafter(model, num_active_override=1)
        draft_result = drafter.draft(input_ids.clone(), num_draft_tokens=4)
        verifier = SelfDraftVerifier(model)
        r1 = verifier.verify(input_ids, draft_result)
        r2 = verifier.verify(input_ids, draft_result)
        assert r1 == r2

    def test_verify_accepts_at_most_draft_plus_one(self, model, input_ids):
        """Can't accept more tokens than were drafted + 1 correction."""
        num_draft = 4
        drafter = MoeSelfDrafter(model, num_active_override=1)
        draft_result = drafter.draft(input_ids, num_draft_tokens=num_draft)
        verifier = SelfDraftVerifier(model)
        accepted = verifier.verify(input_ids, draft_result)
        assert len(accepted) <= num_draft + 1
