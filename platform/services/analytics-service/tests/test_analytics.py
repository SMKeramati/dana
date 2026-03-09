"""Tests for anomaly detector (with actual Z-score math) and usage pipeline."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from src.pipelines.anomaly_detector import AnomalyDetector, AnomalyLevel
from src.pipelines.usage_pipeline import RollupWindow, UsageEvent, UsagePipeline

# ------------------------------------------------------------------
# Anomaly Detector tests
# ------------------------------------------------------------------


class TestAnomalyDetector:
    """Tests verifying real Z-score and EWMA mathematics."""

    def test_normal_values_no_anomaly(self) -> None:
        """A stream of near-identical values should produce no anomalies."""
        detector = AnomalyDetector(window_size=50, z_threshold=3.0, ewma_alpha=0.3)
        results = detector.observe_batch([100.0] * 50)
        for r in results:
            assert r.level == AnomalyLevel.NORMAL

    def test_spike_detected_as_anomaly(self) -> None:
        """A single large spike after a stable baseline should be detected."""
        detector = AnomalyDetector(window_size=50, z_threshold=3.0, ewma_alpha=0.3)
        # Build stable baseline
        for _ in range(60):
            detector.observe(100.0)
        # Inject spike
        result = detector.observe(500.0)
        assert result.level in (AnomalyLevel.WARNING, AnomalyLevel.CRITICAL)
        assert abs(result.z_score) > 3.0

    def test_z_score_calculation_matches_numpy(self) -> None:
        """Verify that the Z-score matches a manual numpy calculation."""
        detector = AnomalyDetector(window_size=10, z_threshold=3.0)
        values = [10.0, 12.0, 11.0, 13.0, 10.5, 12.5, 11.5, 9.5, 14.0, 11.0]
        for v in values:
            detector.observe(v)

        # The window now holds all 10 values. Push one more and check.
        new_val = 25.0
        result = detector.observe(new_val)

        # Manually compute expected Z-score over the window (last 10 including new_val)
        window = values[-9:] + [new_val]  # window_size=10, oldest dropped
        arr = np.array(window)
        expected_z = (new_val - float(np.mean(arr))) / float(np.std(arr, ddof=1))
        assert abs(result.z_score - expected_z) < 1e-6

    def test_ewma_tracks_mean_shift(self) -> None:
        """EWMA mean should adapt to a shift in the data distribution."""
        detector = AnomalyDetector(ewma_alpha=0.5)
        for _ in range(30):
            detector.observe(100.0)
        old_mean = detector.current_ewma_mean

        # Shift distribution to 200
        for _ in range(30):
            detector.observe(200.0)
        new_mean = detector.current_ewma_mean

        assert new_mean > old_mean
        # With alpha=0.5 and 30 observations at 200, mean should be very close to 200
        assert abs(new_mean - 200.0) < 1.0

    def test_ewma_deviation_units(self) -> None:
        """EWMA deviation should be in units of EWMA standard deviation."""
        detector = AnomalyDetector(ewma_alpha=0.3, ewma_threshold=2.5)
        # Stable period
        for _ in range(50):
            detector.observe(50.0)
        # The EWMA std should be near zero after constant input
        assert detector.current_ewma_std < 1.0

    def test_critical_requires_both_signals(self) -> None:
        """CRITICAL level requires both Z-score AND EWMA to fire."""
        detector = AnomalyDetector(window_size=30, z_threshold=2.5, ewma_alpha=0.3, ewma_threshold=2.0)
        for _ in range(40):
            detector.observe(100.0)
        result = detector.observe(400.0)
        if result.level == AnomalyLevel.CRITICAL:
            assert abs(result.z_score) > 2.5
            assert abs(result.ewma_deviation) > 2.0

    def test_gradual_drift_eventually_normalises(self) -> None:
        """A gradual drift should eventually stop triggering anomalies as
        both the Z-score window and EWMA adapt."""
        detector = AnomalyDetector(window_size=20, z_threshold=3.0, ewma_alpha=0.3)
        # Baseline
        for _ in range(30):
            detector.observe(100.0)
        # Gradual drift from 100 to 200 over 50 steps
        for i in range(50):
            detector.observe(100.0 + 2.0 * i)
        # Now stable at 200 for a while
        results = [detector.observe(200.0) for _ in range(30)]
        # The last few observations should all be NORMAL
        assert all(r.level == AnomalyLevel.NORMAL for r in results[-10:])


# ------------------------------------------------------------------
# Usage Pipeline tests
# ------------------------------------------------------------------


class TestUsagePipeline:
    def _event(
        self,
        tenant: str = "t1",
        model: str = "dana-base",
        inp: int = 100,
        out: int = 50,
        latency: float = 45.0,
        ts: datetime | None = None,
    ) -> UsageEvent:
        return UsageEvent(
            tenant_id=tenant,
            model=model,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency,
            timestamp=ts or datetime(2026, 3, 6, 14, 30, 0, tzinfo=UTC),
        )

    def test_single_event_creates_buckets(self) -> None:
        pipeline = UsagePipeline(windows=[RollupWindow.HOUR, RollupWindow.DAY])
        pipeline.push(self._event())
        hourly = pipeline.query("t1", RollupWindow.HOUR)
        daily = pipeline.query("t1", RollupWindow.DAY)
        assert len(hourly) == 1
        assert len(daily) == 1
        assert hourly[0].total_tokens == 150

    def test_accumulation(self) -> None:
        pipeline = UsagePipeline(windows=[RollupWindow.HOUR])
        ts = datetime(2026, 3, 6, 14, 0, 0, tzinfo=UTC)
        pipeline.push(self._event(inp=100, out=50, latency=30.0, ts=ts))
        pipeline.push(self._event(inp=200, out=100, latency=60.0, ts=ts))
        buckets = pipeline.query("t1", RollupWindow.HOUR)
        assert len(buckets) == 1
        assert buckets[0].total_tokens == 450
        assert buckets[0].total_requests == 2
        assert buckets[0].avg_latency_ms == 45.0

    def test_flush_clears_buckets(self) -> None:
        pipeline = UsagePipeline(windows=[RollupWindow.DAY])
        pipeline.push(self._event())
        flushed = pipeline.flush()
        assert len(flushed) == 1
        assert pipeline.query("t1", RollupWindow.DAY) == []

    def test_different_tenants_separate(self) -> None:
        pipeline = UsagePipeline(windows=[RollupWindow.MONTH])
        pipeline.push(self._event(tenant="a"))
        pipeline.push(self._event(tenant="b"))
        assert len(pipeline.query("a", RollupWindow.MONTH)) == 1
        assert len(pipeline.query("b", RollupWindow.MONTH)) == 1
        assert len(pipeline.query("c", RollupWindow.MONTH)) == 0

    def test_min_max_latency(self) -> None:
        pipeline = UsagePipeline(windows=[RollupWindow.HOUR])
        ts = datetime(2026, 3, 6, 14, 0, 0, tzinfo=UTC)
        pipeline.push(self._event(latency=10.0, ts=ts))
        pipeline.push(self._event(latency=90.0, ts=ts))
        bucket = pipeline.query("t1", RollupWindow.HOUR)[0]
        assert bucket.min_latency_ms == 10.0
        assert bucket.max_latency_ms == 90.0
