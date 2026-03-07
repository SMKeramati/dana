"""RabbitMQ event consumer for usage events.

Subscribes to the ``usage.events`` queue and feeds incoming messages into the
analytics pipeline.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from dana_common.logging import get_logger

from ..pipelines.usage_pipeline import UsageEvent, UsagePipeline

logger = get_logger(__name__)

DEFAULT_QUEUE = "usage.events"
DEFAULT_EXCHANGE = "dana.events"


class EventConsumer:
    """Consumes usage events from RabbitMQ and pushes them into a pipeline."""

    def __init__(
        self,
        pipeline: UsagePipeline,
        amqp_url: str = "amqp://guest:guest@localhost/",
        queue_name: str = DEFAULT_QUEUE,
        exchange_name: str = DEFAULT_EXCHANGE,
    ) -> None:
        self._pipeline = pipeline
        self._amqp_url = amqp_url
        self._queue_name = queue_name
        self._exchange_name = exchange_name
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._messages_processed: int = 0

    async def start(self) -> None:
        """Connect to RabbitMQ and begin consuming."""
        self._connection = await aio_pika.connect_robust(self._amqp_url)
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=100)

        exchange = await channel.declare_exchange(self._exchange_name, aio_pika.ExchangeType.TOPIC, durable=True)
        queue = await channel.declare_queue(self._queue_name, durable=True)
        await queue.bind(exchange, routing_key="usage.#")
        await queue.consume(self._on_message)

        logger.info("event_consumer.started", queue=self._queue_name)

    async def stop(self) -> None:
        if self._connection:
            await self._connection.close()
            logger.info("event_consumer.stopped")

    async def _on_message(self, message: AbstractIncomingMessage) -> None:
        async with message.process():
            try:
                payload = json.loads(message.body.decode())
                event = self._parse_event(payload)
                self._pipeline.push(event)
                self._messages_processed += 1
            except Exception:
                logger.exception("event_consumer.parse_error")

    @staticmethod
    def _parse_event(payload: dict[str, Any]) -> UsageEvent:
        ts_raw = payload.get("timestamp")
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(ts_raw, tz=UTC)
        else:
            ts = datetime.now(UTC)

        return UsageEvent(
            tenant_id=payload["tenant_id"],
            model=payload["model"],
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            latency_ms=payload.get("latency_ms", 0.0),
            timestamp=ts,
            metadata=payload.get("metadata", {}),
        )

    @property
    def messages_processed(self) -> int:
        return self._messages_processed
