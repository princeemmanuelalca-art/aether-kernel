"""
L1 Hot Cache: in-process LRU dictionary with async locking.

Design rationale:
- Uses collections.OrderedDict for O(1) LRU eviction without third-party
  dependencies (e.g., cachetools).  OrderedDict.move_to_end gives us exact
  LRU semantics with full control over locking strategy.
- An asyncio.Lock guards all mutations so that concurrent agent tasks can
  safely share the cache without data races.
- The cache is scoped to a single process; multi-node deployments require
  an external hot cache (Redis) but still benefit from L1 as a local
  look-aside buffer to cut inter-service latency.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TypeVar
from uuid import UUID

from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import MemoryRecord
from aether_kernel.core.types import MemoryTier

logger = get_logger(__name__)

KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type


class L1HotCache:
    """Bounded LRU cache for MemoryRecords with async-safe operations.

    All public methods are coroutines (async def) so that callers never
    need to know whether the implementation yields control—this preserves
    the interface contract if we later swap to an async-capable external
    cache (e.g., aioredis).
    """

    def __init__(self, *, maxsize: int = 1_000) -> None:
        # OrderedDict preserves insertion order; move_to_end on access
        # implements LRU.  On overflow, popitem(last=False) evicts the
        # least-recently-used entry.
        self._cache: OrderedDict[UUID, MemoryRecord] = OrderedDict()
        self._maxsize = maxsize
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def get(self, record_id: UUID) -> MemoryRecord | None:
        """Retrieve a record by ID, promoting it to most-recently-used."""
        async with self._lock:
            record = self._cache.get(record_id)
            if record is not None:
                self._cache.move_to_end(record_id)
                self._hits += 1
                return record
            self._misses += 1
            return None

    async def get_by_trace(self, trace_id: UUID) -> list[MemoryRecord]:
        """Return all records for a given trace_id, ordered by recency.

        This is the primary read path for context assembly: the orchestrator
        fetches the current task's full conversation history from L1 before
        falling back to L2.
        """
        async with self._lock:
            # OrderedDict preserves LRU order; we filter without reordering
            # so that this bulk read does not distort the eviction priority.
            return [r for r in self._cache.values() if r.trace_id == trace_id]

    async def put(self, record: MemoryRecord) -> None:
        """Insert or update a record, evicting LRU entries if at capacity."""
        async with self._lock:
            if record.record_id in self._cache:
                self._cache.move_to_end(record.record_id)
                self._cache[record.record_id] = record
                return
            # Evict until we have room (handles batch inserts where
            # a single record might exceed maxsize, though that is pathological).
            while len(self._cache) >= self._maxsize:
                evicted_id, evicted = self._cache.popitem(last=False)
                logger.debug("L1 evicted record %s", evicted_id)
            self._cache[record.record_id] = record

    async def put_batch(self, records: list[MemoryRecord]) -> None:
        """Best-effort batch insert; evictions occur between each put."""
        for record in records:
            await self.put(record)

    async def delete(self, record_id: UUID) -> bool:
        """Remove a record; returns True if it existed."""
        async with self._lock:
            existed = self._cache.pop(record_id, None) is not None
            return existed

    async def clear(self) -> None:
        """Evict all entries.  Useful for testing and memory-pressure relief."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def stats(self) -> dict[str, int]:
        """Return hit/miss/cache-size metrics for observability."""
        async with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits // total if total else 0,
            }
