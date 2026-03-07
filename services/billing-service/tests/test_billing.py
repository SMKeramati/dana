"""Tests for token counter, usage aggregator, and pricing engine."""
from __future__ import annotations

import time

from src.metering.token_counter import TokenCount, TokenCounter
from src.metering.usage_aggregator import TimeWindow, UsageAggregator, UsageRecord
from src.plans.pricing import PricingEngine
from src.plans.subscription import PlanTier

# ------------------------------------------------------------------
# Token counter tests
# ------------------------------------------------------------------


class TestTokenCounter:
    def test_count_with_input_and_output(self) -> None:
        counter = TokenCounter()
        tc = counter.count(model="dana-base", input_text="hello world", output_text="goodbye")
        assert tc.input_tokens > 0
        assert tc.output_tokens > 0
        assert tc.total_tokens == tc.input_tokens + tc.output_tokens

    def test_count_empty_output(self) -> None:
        counter = TokenCounter()
        tc = counter.count(model="dana-base", input_text="hello", output_text="")
        assert tc.output_tokens == 0

    def test_different_models_different_estimates(self) -> None:
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog." * 10
        tc_base = counter.count(model="dana-base", input_text=text)
        tc_code = counter.count(model="dana-code", input_text=text)
        assert tc_base.input_tokens != tc_code.input_tokens

    def test_custom_ratio(self) -> None:
        counter = TokenCounter()
        counter.register_model("custom-model", 2.0)
        tc = counter.count(model="custom-model", input_text="abcdefgh")
        assert tc.input_tokens > 0

    def test_to_dict(self) -> None:
        tc = TokenCount(input_tokens=10, output_tokens=20, total_tokens=30, model="test")
        d = tc.to_dict()
        assert d["input_tokens"] == 10
        assert d["output_tokens"] == 20
        assert d["total_tokens"] == 30


# ------------------------------------------------------------------
# Usage aggregator tests
# ------------------------------------------------------------------


class TestUsageAggregator:
    def _make_record(
        self,
        org: str = "org1",
        model: str = "dana-base",
        inp: int = 100,
        out: int = 50,
        ts: float | None = None,
    ) -> UsageRecord:
        return UsageRecord(
            org_id=org,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            timestamp=ts or time.time(),
        )

    def test_single_ingest_and_flush(self) -> None:
        agg = UsageAggregator()
        agg.ingest(self._make_record(), windows=[TimeWindow.HOURLY])
        results = agg.flush()
        assert len(results) == 1
        assert results[0].total_tokens == 150
        assert results[0].request_count == 1

    def test_multiple_records_same_bucket(self) -> None:
        agg = UsageAggregator()
        ts = time.time()
        agg.ingest(self._make_record(ts=ts), windows=[TimeWindow.HOURLY])
        agg.ingest(self._make_record(inp=200, out=100, ts=ts), windows=[TimeWindow.HOURLY])
        results = agg.flush()
        assert len(results) == 1
        assert results[0].total_tokens == 450
        assert results[0].request_count == 2

    def test_different_orgs(self) -> None:
        agg = UsageAggregator()
        agg.ingest(self._make_record(org="a"), windows=[TimeWindow.MONTHLY])
        agg.ingest(self._make_record(org="b"), windows=[TimeWindow.MONTHLY])
        results = agg.flush()
        assert len(results) == 2

    def test_peek_does_not_clear(self) -> None:
        agg = UsageAggregator()
        agg.ingest(self._make_record(), windows=[TimeWindow.DAILY])
        peeked = agg.peek("org1", "dana-base", TimeWindow.DAILY)
        assert len(peeked) == 1
        results = agg.flush()
        assert len(results) >= 1


# ------------------------------------------------------------------
# Pricing engine tests
# ------------------------------------------------------------------


class TestPricingEngine:
    def setup_method(self) -> None:
        self.engine = PricingEngine()

    def test_basic_pricing(self) -> None:
        result = self.engine.compute("gpt-4", 1000, 500, PlanTier.FREE)
        assert result.total_microcents > 0
        assert result.total_cents == result.total_microcents / 1_000_000

    def test_tier_multiplier_pro_cheaper(self) -> None:
        free = self.engine.compute("gpt-4", 1000, 500, PlanTier.FREE)
        pro = self.engine.compute("gpt-4", 1000, 500, PlanTier.PRO)
        assert pro.total_cents < free.total_cents

    def test_tier_multiplier_enterprise_cheapest(self) -> None:
        pro = self.engine.compute("gpt-4", 1000, 500, PlanTier.PRO)
        ent = self.engine.compute("gpt-4", 1000, 500, PlanTier.ENTERPRISE)
        assert ent.total_cents < pro.total_cents

    def test_volume_discount(self) -> None:
        no_vol = self.engine.compute("gpt-4", 1000, 500, PlanTier.FREE, monthly_tokens_so_far=0)
        with_vol = self.engine.compute("gpt-4", 1000, 500, PlanTier.FREE, monthly_tokens_so_far=50_000_000)
        assert with_vol.total_cents < no_vol.total_cents

    def test_zero_tokens(self) -> None:
        result = self.engine.compute("gpt-4", 0, 0, PlanTier.FREE)
        assert result.total_cents == 0.0

    def test_different_models_different_prices(self) -> None:
        gpt4 = self.engine.compute("gpt-4", 1000, 500, PlanTier.FREE)
        llama = self.engine.compute("llama", 1000, 500, PlanTier.FREE)
        assert gpt4.total_cents > llama.total_cents
