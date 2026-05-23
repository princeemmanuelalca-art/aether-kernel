"""
EventBroker: asynchronous message bus for inter-agent communication.

Design rationale:
-----------------
- Uses asyncio.Queue rather than an external broker (Redis/RabbitMQ) to
  eliminate network hops and serialization overhead within a single process.
- Backpressure is enforced via maxsize on each per-role queue; when a
  consumer is slow, producers block rather than consuming unbounded memory.
- Each AgentRole gets its own dedicated queue, enabling selective consumer
  scaling without head-of-line blocking across heterogeneous agents.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from aether_kernel.core.exceptions import BrokerQueueFullError
from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import AgentMessage
from aether_kernel.core.types import AgentRole

logger = get_logger(__name__)

# Default backpressure limit per role queue.
# Tuned for high-throughput: 10_000 messages ≈ ~50 MB in-flight for typical
# AgentMessage sizes. Operators may override via the constructor.
_DEFAULT_QUEUE_MAXSIZE = 10_000


class EventBroker:
    """Fan-out message broker with per-role queues and backpressure.

    Lifecycle:
        1. ``await broker.start()`` spawns consumer tasks.
        2. ``broker.publish(msg)`` enqueues to the target role's queue.
        3. ``broker.register_handler(role, handler)`` binds a callable.
        4. ``await broker.stop()`` drains queues and cancels consumers.
    """

    def __init__(
        self,
        *,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        consumer_concurrency: dict[AgentRole, int] | None = None,
    ) -> None:
        # Each role maps to a bounded asyncio.Queue.  Bounded queues are
        # critical for memory safety under burst load: if a downstream agent
        # stalls, publish() will raise BrokerQueueFullError rather than OOM.
        self._queues: dict[AgentRole, asyncio.Queue[AgentMessage]] = {
            role: asyncio.Queue(maxsize=queue_maxsize) for role in AgentRole
        }

        # Handler registry: role -> list of async callables.
        # Multiple handlers per role enable fan-out patterns (e.g., metrics
        # sidecar + primary consumer) without duplicating queue reads.
        self._handlers: dict[AgentRole, list[Callable[[AgentMessage], Awaitable[None]]]] = {
            role: [] for role in AgentRole
        }

        # Consumer concurrency per role.  Defaults to 1; heavy consumers
        # (e.g., Executor) may be scaled to N to increase throughput.
        self._concurrency: dict[AgentRole, int] = {
            role: (consumer_concurrency or {}).get(role, 1) for role in AgentRole
        }

        # Background consumer tasks; kept for graceful shutdown.
        self._consumer_tasks: set[asyncio.Task[None]] = set()
        self._running: bool = False

        # Re-entrant lock protects handler registration during active
        # consumption—prevents race conditions in dynamic topology updates.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn consumer coroutines for all roles with registered handlers."""
        async with self._lock:
            if self._running:
                return
            self._running = True
            for role in AgentRole:
                count = self._concurrency[role]
                for _ in range(count):
                    task = asyncio.create_task(
                        self._consume(role), name=f"broker-consumer-{role.value}"
                    )
                    self._consumer_tasks.add(task)
                    # Eager cleanup on task exit without awaiting each one.
                    task.add_done_callback(self._consumer_tasks.discard)
        logger.info("EventBroker started", extra={"concurrency": self._concurrency})

    async def stop(self, *, timeout: float = 30.0) -> None:
        """Signal cancellation and await consumer teardown.

        Args:
            timeout: Seconds to wait for in-flight message processing.
        """
        self._running = False
        # Cancel all consumer tasks.
        for task in list(self._consumer_tasks):
            task.cancel()
        # Gather with timeout to avoid hanging on a stuck handler.
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._consumer_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("EventBroker stop timed out; forcing task cancellation")
        logger.info("EventBroker stopped")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, message: AgentMessage) -> None:
        """Enqueue *message* onto the target role's queue.

        Raises:
            BrokerQueueFullError: If the target queue is at max capacity.
        """
        queue = self._queues[message.target]
        if queue.full():
            raise BrokerQueueFullError(
                f"Queue for {message.target.value} is full (maxsize={queue.maxsize})",
                context={"trace_id": str(message.trace_id), "target": message.target.value},
            )
        queue.put_nowait(message)

    async def publish_batch(self, messages: list[AgentMessage]) -> None:
        """Best-effort batch publish; individual failures are logged, not raised."""
        for msg in messages:
            try:
                await self.publish(msg)
            except BrokerQueueFullError:
                logger.error(
                    "Dropped message due to backpressure",
                    extra={"context": {"trace_id": str(msg.trace_id), "target": msg.target.value}},
                )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def register_handler(
        self,
        role: AgentRole,
        handler: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """Bind an async handler to consume messages for *role*.

        Thread-safe for calls before ``start()``.  Dynamic registration after
        ``start()`` should be wrapped in the same ``async with broker._lock``.
        """
        self._handlers[role].append(handler)

    # ------------------------------------------------------------------
    # Internal consumer loop
    # ------------------------------------------------------------------

    async def _consume(self, role: AgentRole) -> None:
        """Infinite-loop consumer that fans out to all registered handlers."""
        queue = self._queues[role]
        while self._running:
            try:
                # Use wait_for so that periodic cancellation checks occur
                # even when the queue is empty.
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Fan-out to every registered handler concurrently.
            # If any handler raises, we log but continue—broker reliability
            # must not depend on consumer correctness.
            handler_coros = [
                self._invoke_handler(handler, message) for handler in self._handlers[role]
            ]
            if handler_coros:
                await asyncio.gather(*handler_coros, return_exceptions=True)

    async def _invoke_handler(
        self, handler: Callable[[AgentMessage], Awaitable[None]], message: AgentMessage
    ) -> None:
        try:
            await handler(message)
        except Exception as exc:
            logger.exception(
                "Handler error for role %s",
                message.target.value,
                extra={"context": {"trace_id": str(message.trace_id), "error": str(exc)}},
            )
