"""RabbitMQ job dispatch for inference routing.

Publishes InferenceJob messages to the inference queue and consumes
InferenceResult responses from workers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dana_common.config import rabbitmq as rmq_config
from dana_common.messaging import MessageConsumer, MessagePublisher
from dana_common.models import InferenceJob, InferenceResult

logger = logging.getLogger(__name__)

# Exchange and queue names
INFERENCE_EXCHANGE = "dana.inference"
INFERENCE_QUEUE = "inference.jobs"
RESULT_EXCHANGE = "dana.inference.results"
RESULT_QUEUE = "inference.results"
ROUTING_KEY_JOB = "inference.job.submit"
ROUTING_KEY_RESULT = "inference.result.completed"


class QueueManager:
    """Manages publishing inference jobs and consuming results via RabbitMQ."""

    def __init__(self, amqp_url: str | None = None) -> None:
        self._amqp_url = amqp_url or rmq_config.url
        self._publisher = MessagePublisher(self._amqp_url)
        self._consumer = MessageConsumer(self._amqp_url)
        self._result_callbacks: dict[str, asyncio.Future[InferenceResult]] = {}

    async def connect(self) -> None:
        """Establish connections for publishing and consuming."""
        await self._publisher.connect()
        await self._consumer.connect()
        logger.info("QueueManager connected to RabbitMQ")

    async def publish_job(self, job: InferenceJob, priority: int = 0) -> None:
        """Publish an inference job to the job queue.

        Args:
            job: The InferenceJob to dispatch.
            priority: AMQP message priority (0-10). Higher priority jobs
                      are consumed first by workers.
        """
        body = job.model_dump(mode="json")
        await self._publisher.publish(
            exchange_name=INFERENCE_EXCHANGE,
            routing_key=ROUTING_KEY_JOB,
            body=body,
            priority=priority,
        )
        logger.info("Published job %s with priority %d", job.job_id, priority)

    async def wait_for_result(self, job_id: str, timeout: float = 120.0) -> InferenceResult:
        """Wait for a result for a specific job_id.

        Creates an asyncio.Future that is resolved when a matching result
        arrives on the result queue.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        self._result_callbacks[job_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            logger.error("Timeout waiting for result of job %s", job_id)
            return InferenceResult(job_id=job_id, error="Inference timed out")
        finally:
            self._result_callbacks.pop(job_id, None)

    async def _handle_result(self, body: dict[str, Any]) -> None:
        """Internal handler for incoming result messages."""
        try:
            result = InferenceResult.model_validate(body)
        except Exception:
            logger.exception("Failed to parse inference result")
            return

        future = self._result_callbacks.get(result.job_id)
        if future and not future.done():
            future.set_result(result)
            logger.debug("Resolved result for job %s", result.job_id)
        else:
            logger.warning("Received result for unknown/expired job %s", result.job_id)

    async def start_result_consumer(self) -> None:
        """Start consuming inference results from the result queue."""
        await self._consumer.consume(
            queue_name=RESULT_QUEUE,
            exchange_name=RESULT_EXCHANGE,
            routing_key=ROUTING_KEY_RESULT,
            handler=self._handle_result,
        )
        logger.info("Result consumer started")

    async def close(self) -> None:
        """Close publisher and consumer connections."""
        await self._publisher.close()
        await self._consumer.close()
        # Cancel any pending futures
        for job_id, future in self._result_callbacks.items():
            if not future.done():
                future.set_result(
                    InferenceResult(job_id=job_id, error="Queue manager shutting down")
                )
        self._result_callbacks.clear()
        logger.info("QueueManager closed")
