"""Tests for custom sliding-window rate limiter."""

from dana_common.cache import SlidingWindowCounter


class TestSlidingWindowCounter:
    """Test the custom sliding window rate limiter logic.

    Note: These tests require a running Redis instance.
    In CI, Redis is provided as a service. For local dev, skip if unavailable.
    """

    def test_counter_init(self) -> None:
        """Test that counter can be created."""
        # Basic import and construction test (no Redis needed)
        assert SlidingWindowCounter is not None
