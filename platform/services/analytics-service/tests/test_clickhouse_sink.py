"""Tests for ClickHouse sink (offline — no real ClickHouse needed)."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.sinks.clickhouse_sink import (
    ApiRequestEvent,
    BillingEvent,
    ClickHouseSink,
    ErrorEvent,
    InferenceEvent,
    UserEvent,
    _row_to_json_line,
)


class TestEventDataclasses:
    def test_api_request_event_defaults(self) -> None:
        ev = ApiRequestEvent(
            request_id="req-1",
            user_id="user-1",
            api_key_prefix="dk-abc",
            endpoint="/v1/chat/completions",
            method="POST",
            status_code=200,
            response_time_ms=342,
            model="qwen3-235b-moe",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            tier="pro",
        )
        assert ev.total_tokens == 300
        assert ev.streaming == 0
        assert ev.timestamp  # auto-set

    def test_user_event_properties_serialised(self) -> None:
        ev = UserEvent(user_id="u1", event_type="registered", properties={"plan": "free"})
        line = _row_to_json_line(ev)
        data = json.loads(line)
        assert data["event_type"] == "registered"
        # properties should be a JSON string after serialisation
        props = json.loads(data["properties"])
        assert props["plan"] == "free"

    def test_inference_event_defaults(self) -> None:
        ev = InferenceEvent(
            job_id="j1",
            user_id="u1",
            model="qwen3-235b-moe",
            prompt_tokens=50,
            completion_tokens=100,
            time_to_first_token_ms=120,
            total_latency_ms=800,
            tokens_per_second=125.0,
        )
        assert ev.cache_hit == 0
        assert ev.speculative_accepted == 0

    def test_billing_event_serialised(self) -> None:
        ev = BillingEvent(
            user_id="u1",
            org_id="org1",
            event_type="charge",
            amount_cents=150,
            total_tokens=1000,
            model="qwen3-235b-moe",
            tier="free",
        )
        line = _row_to_json_line(ev)
        data = json.loads(line)
        assert data["amount_cents"] == 150

    def test_error_event(self) -> None:
        ev = ErrorEvent(
            service="api-gateway",
            error_type="RateLimitError",
            error_code="RATE_LIMIT_EXCEEDED",
            message="Too many requests",
        )
        line = _row_to_json_line(ev)
        data = json.loads(line)
        assert data["service"] == "api-gateway"


class TestClickHouseSink:
    def _make_sink(self) -> ClickHouseSink:
        return ClickHouseSink(batch_size=50, auto_flush_interval=60.0)

    def test_write_buffers_event(self) -> None:
        sink = self._make_sink()
        ev = ApiRequestEvent(
            request_id="r1", user_id="u1", api_key_prefix="dk-x",
            endpoint="/v1/chat/completions", method="POST", status_code=200,
            response_time_ms=100, model="qwen3", prompt_tokens=10,
            completion_tokens=20, total_tokens=30, tier="free",
        )
        sink.write(ev)
        assert len(sink._buffers.get("api_requests", [])) == 1

    def test_unknown_event_type_ignored(self) -> None:
        sink = self._make_sink()
        sink.write(object())  # not a known event type
        assert not sink._buffers

    def test_flush_clears_buffer(self) -> None:
        sink = self._make_sink()
        ev = ApiRequestEvent(
            request_id="r1", user_id="u1", api_key_prefix="dk-x",
            endpoint="/health", method="GET", status_code=200,
            response_time_ms=5, model="qwen3", prompt_tokens=0,
            completion_tokens=0, total_tokens=0, tier="free",
        )
        sink._buffers["api_requests"] = [_row_to_json_line(ev)]

        with patch("src.sinks.clickhouse_sink._execute") as mock_exec:
            sink.flush()
            mock_exec.assert_called_once()
            assert sink._buffers["api_requests"] == []
            assert sink._written == 1

    def test_flush_error_increments_counter(self) -> None:
        sink = self._make_sink()
        sink._buffers["api_requests"] = ['{"x":1}']
        with patch("src.sinks.clickhouse_sink._execute", side_effect=RuntimeError("conn")):
            sink.flush()
        assert sink._errors == 1

    def test_stats(self) -> None:
        sink = self._make_sink()
        sink._written = 42
        sink._errors = 2
        assert sink.stats() == {"written": 42, "errors": 2}

    def test_ping_success(self) -> None:
        sink = self._make_sink()
        with patch("src.sinks.clickhouse_sink._execute"):
            assert sink.ping() is True

    def test_ping_failure(self) -> None:
        sink = self._make_sink()
        with patch("src.sinks.clickhouse_sink._execute", side_effect=RuntimeError("down")):
            assert sink.ping() is False

    def test_auto_flush_on_batch_size(self) -> None:
        sink = ClickHouseSink(batch_size=2, auto_flush_interval=60.0)

        def make_ev() -> ApiRequestEvent:
            return ApiRequestEvent(
                request_id="r", user_id="u", api_key_prefix="dk-x",
                endpoint="/v1", method="POST", status_code=200,
                response_time_ms=50, model="m", prompt_tokens=10,
                completion_tokens=10, total_tokens=20, tier="free",
            )

        with patch("src.sinks.clickhouse_sink._execute"):
            sink.write(make_ev())
            assert len(sink._buffers.get("api_requests", [])) == 1
            sink.write(make_ev())  # triggers auto-flush at batch_size=2
            assert sink._buffers.get("api_requests", []) == []
