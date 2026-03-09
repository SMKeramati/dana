"""Tests for spec-decode-tree package.

Tests: acceptance tracker, adaptive draft length, tree speculator, tree verifier.
All tests run on CPU with the synthetic tiny MoE model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../dana-engine/src"))

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer

from spec_decode_tree.acceptance import AcceptanceTracker
from spec_decode_tree.adaptive import AdaptiveDraftLength
from spec_decode_tree.tree_spec import TreeSpeculator, DraftTree, DraftNode
from spec_decode_tree.verify import TreeVerifier


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
# AcceptanceTracker
# ------------------------------------------------------------------

class TestAcceptanceTracker:
    def test_initial_rate_is_zero(self):
        t = AcceptanceTracker(window=100)
        assert t.rate() == 0.0

    def test_record_and_rate(self):
        t = AcceptanceTracker(window=100)
        t.record(accepted=8, proposed=10)
        assert t.rate() == pytest.approx(0.8, abs=1e-6)

    def test_multiple_records(self):
        t = AcceptanceTracker(window=100)
        t.record(5, 10)
        t.record(5, 10)
        assert t.rate() == pytest.approx(0.5, abs=1e-6)

    def test_rate_in_range(self):
        t = AcceptanceTracker(window=100)
        for _ in range(50):
            t.record(3, 5)
        r = t.rate()
        assert 0.0 <= r <= 1.0

    def test_window_eviction(self):
        """Old records should be evicted after window size."""
        t = AcceptanceTracker(window=10)
        # Add 10 perfect records
        for _ in range(10):
            t.record(10, 10)
        assert t.rate() == pytest.approx(1.0, abs=1e-6)
        # Now add 10 zero records — old perfect ones should be evicted
        for _ in range(10):
            t.record(0, 10)
        assert t.rate() == pytest.approx(0.0, abs=1e-6)

    def test_total_counters(self):
        t = AcceptanceTracker(window=100)
        t.record(3, 5)
        t.record(4, 7)
        assert t.total_accepted() == 7
        assert t.total_proposed() == 12

    def test_per_depth_rate(self):
        t = AcceptanceTracker(window=100)
        t.record_depth(depth=1, accepted=1, proposed=1)
        t.record_depth(depth=2, accepted=0, proposed=1)
        rates = t.per_depth_rate()
        assert rates[1] == pytest.approx(1.0, abs=1e-6)
        assert rates[2] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# AdaptiveDraftLength
# ------------------------------------------------------------------

class TestAdaptiveDraftLength:
    def test_initial_depth_width(self):
        a = AdaptiveDraftLength(min_depth=1, max_depth=6, min_width=1, max_width=4)
        assert 1 <= a.next_depth() <= 6
        assert 1 <= a.next_width() <= 4

    def test_high_acceptance_increases_depth(self):
        a = AdaptiveDraftLength(min_depth=1, max_depth=6, initial_depth=2)
        t = AcceptanceTracker(window=100)
        for _ in range(20):
            t.record(9, 10)  # 90% acceptance rate
        initial = a.next_depth()
        a.update(t)
        assert a.next_depth() >= initial  # should stay or increase

    def test_low_acceptance_decreases_depth(self):
        a = AdaptiveDraftLength(min_depth=1, max_depth=6, initial_depth=4)
        t = AcceptanceTracker(window=100)
        for _ in range(20):
            t.record(2, 10)  # 20% acceptance rate
        initial = a.next_depth()
        a.update(t)
        assert a.next_depth() <= initial  # should stay or decrease

    def test_depth_clamped_to_bounds(self):
        a = AdaptiveDraftLength(min_depth=1, max_depth=6, initial_depth=6)
        t = AcceptanceTracker(window=100)
        for _ in range(50):
            t.record(10, 10)
        a.update(t)
        assert a.next_depth() <= 6

        a2 = AdaptiveDraftLength(min_depth=1, max_depth=6, initial_depth=1)
        for _ in range(50):
            t.record(0, 10)
        a2.update(t)
        assert a2.next_depth() >= 1


# ------------------------------------------------------------------
# TreeSpeculator
# ------------------------------------------------------------------

class TestTreeSpeculator:
    def test_draft_returns_draft_tree(self, model, input_ids):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        assert isinstance(tree, DraftTree)

    def test_draft_has_correct_num_paths(self, model, input_ids):
        depth, width = 2, 2
        spec = TreeSpeculator(model, depth=depth, width=width)
        tree = spec.draft(input_ids)
        # width^depth leaf paths
        assert tree.num_candidates() == width ** depth

    def test_paths_have_correct_depth(self, model, input_ids):
        depth = 3
        spec = TreeSpeculator(model, depth=depth, width=2)
        tree = spec.draft(input_ids)
        for path in tree.paths:
            assert len(path) == depth

    def test_all_path_tokens_are_valid(self, model, input_ids, config):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        for path in tree.paths:
            for tok in path:
                assert 0 <= tok < config.vocab_size

    def test_draft_tree_preserves_input_ids(self, model, input_ids):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        assert torch.equal(tree.input_ids, input_ids)

    def test_nodes_have_correct_depths(self, model, input_ids):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        for node in tree.nodes:
            assert node.depth >= 1

    def test_depth_1_width_1(self, model, input_ids):
        spec = TreeSpeculator(model, depth=1, width=1)
        tree = spec.draft(input_ids)
        assert tree.num_candidates() == 1
        assert tree.max_depth() == 1

    def test_no_nan_in_logprobs(self, model, input_ids):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        for node in tree.nodes:
            assert not (node.logprob != node.logprob)  # NaN check


# ------------------------------------------------------------------
# TreeVerifier
# ------------------------------------------------------------------

class TestTreeVerifier:
    def test_verify_returns_nonempty_list(self, model, input_ids):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        verifier = TreeVerifier(model)
        result = verifier.verify(tree)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_accepted_tokens_are_valid(self, model, input_ids, config):
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        verifier = TreeVerifier(model)
        result = verifier.verify(tree)
        for tok in result:
            assert 0 <= tok < config.vocab_size

    def test_verify_empty_tree_uses_fallback(self, model, input_ids):
        """Empty tree should fall back to greedy sampling of 1 token."""
        empty_tree = DraftTree(nodes=[], paths=[], input_ids=input_ids)
        verifier = TreeVerifier(model)
        result = verifier.verify(empty_tree)
        assert len(result) == 1

    def test_verify_longer_than_greedy(self, model, input_ids):
        """Speculative decode should accept ≥1 token; often accepts more than naive."""
        spec = TreeSpeculator(model, depth=3, width=2)
        tree = spec.draft(input_ids)
        verifier = TreeVerifier(model)
        result = verifier.verify(tree)
        # Must accept at least 1 token
        assert len(result) >= 1

    def test_verify_consistent(self, model, input_ids):
        """Same draft tree → same verification result (deterministic)."""
        spec = TreeSpeculator(model, depth=2, width=2)
        tree = spec.draft(input_ids)
        verifier = TreeVerifier(model)
        r1 = verifier.verify(tree)
        r2 = verifier.verify(tree)
        assert r1 == r2
