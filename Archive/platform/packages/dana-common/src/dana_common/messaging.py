"""RabbitMQ publisher/consumer base classes.

Uses aio-pika for async AMQP communication between microservices.
Custom dead-letter handling and retry logic.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aio_pika

logger = logging.getLogger(__name__)


class MessagePublisher:
    """Publishes messages to RabbitMQ exchanges."""

    def __init__(self, amqp_url: str) -> None:
        self._url = amqp_url
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()

    async def publish(
        self,
        exchange_name: str,
        routing_key: str,
        body: dict[str, Any],
        priority: int = 0,
    ) -> None:
        if self._channel is None:
            await self.connect()
        assert self._channel is not None
        exchange = await self._channel.declare_exchange(
            exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        message = aio_pika.Message(
            body=json.dumps(body, default=str).encode(),
            content_type="application/json",
            priority=priority,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await exchange.publish(message, routing_key=routing_key)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


class MessageConsumer:
    """Consumes messages from RabbitMQ queues with dead-letter support."""

    def __init__(self, amqp_url: str) -> None:
        self._url = amqp_url
        self._connection: aio_pika.abc.AbstractConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None

    async def connect(self) -> None:
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=10)

    async def consume(
        self,
        queue_name: str,
        exchange_name: str,
        routing_key: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        max_retries: int = 3,
    ) -> None:
        if self._channel is None:
            await self.connect()
        assert self._channel is not None

        # Declare dead-letter exchange and queue
        dl_exchange = await self._channel.declare_exchange(
            f"{exchange_name}.dlx", aio_pika.ExchangeType.TOPIC, durable=True
        )
        dl_queue = await self._channel.declare_queue(
            f"{queue_name}.dlq", durable=True
        )
        await dl_queue.bind(dl_exchange, routing_key=routing_key)

        # Declare main exchange and queue with dead-letter config
        exchange = await self._channel.declare_exchange(
            exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": f"{exchange_name}.dlx",
                "x-dead-letter-routing-key": routing_key,
                "x-max-priority": 10,
            },
        )
        await queue.bind(exchange, routing_key=routing_key)

        async def _on_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with message.process(requeue=False):
                try:
                    body = json.loads(message.body)
                    await handler(body)
                except Exception:
                    raw_retry = (message.headers or {}).get("x-retry-count", 0)
                    retry_count = raw_retry if isinstance(raw_retry, int) else 0
                    if retry_count < max_retries:
                        logger.warning(
                            "Retrying message (attempt %d/%d)",
                            retry_count + 1,
                            max_retries,
                        )
                        retry_msg = aio_pika.Message(
                            body=message.body,
                            headers={"x-retry-count": retry_count + 1},
                            priority=message.priority or 0,
                            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        )
                        await exchange.publish(retry_msg, routing_key=routing_key)
                    else:
                        logger.error("Message exceeded max retries, sent to DLQ")
                        raise

        await queue.consume(_on_message)
        logger.info("Consuming from %s", queue_name)

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
