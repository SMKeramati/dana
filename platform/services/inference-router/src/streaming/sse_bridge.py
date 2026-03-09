"""Server-Sent Events bridge for streaming inference responses.

Daneshbonyan: Internal Design & Development - Custom SSE bridge that
multiplexes token-by-token responses from GPU workers to individual
client HTTP streams. Each job_id maps to a dedicated asyncio.Queue
that feeds an SSE generator consumed by the FastAPI streaming response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from dana_common.models import ChatCompletionChunk

logger = logging.getLogger(__name__)

# Sentinel value to signal end of stream
_STREAM_END = object()
_STREAM_ERROR = object()


class SSEBridge:
    """Multiplexes worker streaming responses to per-client SSE streams.

    Daneshbonyan: Internal Design & Development - Not a generic SSE library.
    Custom implementation that manages per-job token queues, supports
    backpressure via bounded queues, and handles client disconnection
    gracefully by cleaning up resources.
    """

    def __init__(self, max_queue_size: int = 512) -> None:
        self._streams: dict[str, asyncio.Queue[Any]] = {}
        self._max_queue_size = max_queue_size
        self._active_count: int = 0

    def create_stream(self, job_id: str) -> None:
        """Register a new SSE stream for a job_id."""
        if job_id in self._streams:
            logger.warning("Stream already exists for job %s", job_id)
            return
        self._streams[job_id] = asyncio.Queue(maxsize=self._max_queue_size)
        self._active_count += 1
        logger.debug("Created SSE stream for job %s", job_id)

    async def push_token(self, job_id: str, chunk: ChatCompletionChunk) -> None:
        """Push a token chunk to the stream for a given job.

        If the client has disconnected (stream removed), the chunk is dropped.
        """
        queue = self._streams.get(job_id)
        if queue is None:
            return
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            logger.warning("SSE queue full for job %s, applying backpressure", job_id)
            await queue.put(chunk)

    async def push_error(self, job_id: str, error_message: str) -> None:
        """Push an error event and end the stream."""
        queue = self._streams.get(job_id)
        if queue is None:
            return
        error_data = {"error": error_message}
        await queue.put((_STREAM_ERROR, error_data))

    async def end_stream(self, job_id: str) -> None:
        """Signal that no more tokens will arrive for this job."""
        queue = self._streams.get(job_id)
        if queue is None:
            return
        await queue.put(_STREAM_END)
        logger.debug("Ended SSE stream for job %s", job_id)

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events for a given job.

        This is the generator consumed by FastAPI's StreamingResponse.
        It blocks on the queue until tokens arrive or the stream ends.
        """
        queue = self._streams.get(job_id)
        if queue is None:
            logger.error("No stream registered for job %s", job_id)
            return

        try:
            while True:
                item = await queue.get()

                if item is _STREAM_END:
                    # Send the final [DONE] event per OpenAI SSE convention
                    yield "data: [DONE]\n\n"
                    break

                if isinstance(item, tuple) and len(item) == 2 and item[0] is _STREAM_ERROR:
                    error_payload = json.dumps(item[1])
                    yield f"event: error\ndata: {error_payload}\n\n"
                    break

                if isinstance(item, ChatCompletionChunk):
                    payload = item.model_dump_json()
                    yield f"data: {payload}\n\n"
                else:
                    # Fallback for raw dict payloads
                    yield f"data: {json.dumps(item, default=str)}\n\n"

        finally:
            self._cleanup(job_id)

    def _cleanup(self, job_id: str) -> None:
        """Remove stream resources for a completed or disconnected job."""
        removed = self._streams.pop(job_id, None)
        if removed is not None:
            self._active_count -= 1
            logger.debug("Cleaned up SSE stream for job %s (active=%d)", job_id, self._active_count)

    @property
    def active_streams(self) -> int:
        return self._active_count

    def has_stream(self, job_id: str) -> bool:
        return job_id in self._streams
