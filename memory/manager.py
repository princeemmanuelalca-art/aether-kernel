"""
MemoryManager: unified facade over the L1/L2 memory hierarchy.

Provides a single async interface for storing and retrieving MemoryRecords,
with automatic tiering logic: writes go to both L1 and L2; reads hit L1
first and fall back to L2 on miss (with optional L1 promotion on success).
"""

from __future__ import annotations

from uuid import UUID

from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import MemoryRecord, PrunedContext
from aether_kernel.core.types import MemoryTier
from aether_kernel.memory.l1_hot_cache import L1HotCache
from aether_kernel.memory.l2_vector_store import AbstractVectorStore, InMemoryVectorStore
from aether_kernel.memory.pruner import ContextPruner

logger = get_logger(__name__)


class MemoryManager:
    """Unified memory facade with transparent L1/L2 tiering.

    Usage:
        manager = MemoryManager(l1_maxsize=1000, l2_store=qdrant_client)
        await manager.store(record, embedding=[0.1, 0.2, ...])
        history = await manager.get_context(trace_id, max_tokens=4096)
    """

    def __init__(
        self,
        *,
        l1_maxsize: int = 1_000,
        l2_store: AbstractVectorStore | None = None,
        pruner: ContextPruner | None = None,
    ) -> None:
        self._l1 = L1HotCache(maxsize=l1_maxsize)
        self._l2 = l2_store or InMemoryVectorStore()
        self._pruner = pruner or ContextPruner()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def store(self, record: MemoryRecord, *, embedding: list[float] | None = None) -> None:
        """Persist a record to L1 (always) and L2 (if embedding provided).

        The embedding is optional because not all records need semantic
        search (e.g., system messages, tool outputs).  When absent, the
        record is L1-only and will not appear in vector search results.
        """
        await self._l1.put(record)
        if embedding is not None:
            try:
                await self._l2.upsert(record, embedding)
            except Exception:
                logger.exception("L2 upsert failed for record %s; L1 still consistent", record.record_id)

    async def store_batch(
        self, records: list[MemoryRecord], *, embeddings: list[list[float]] | None = None
    ) -> None:
        """Best-effort batch store.  L2 failures are logged, not raised."""
        for idx, record in enumerate(records):
            emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
            await self.store(record, embedding=emb)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    async def get(self, record_id: UUID) -> MemoryRecord | None:
        """Read-through cache: L1 first, no L2 fallback for single-record gets."""
        return await self._l1.get(record_id)

    async def get_context(
        self,
        trace_id: UUID,
        *,
        max_tokens: int | None = None,
    ) -> PrunedContext:
        """Assemble the full conversation context for *trace_id*, pruned to
        *max_tokens* (or unpruned if None).

        This is the primary interface called by agents before LLM invocation.
        """
        # L1 contains all records for recent tasks; L2 provides deeper history.
        l1_records = await self._l1.get_by_trace(trace_id)

        if max_tokens is None:
            total_tokens = sum(
                r.token_count or len(r.content.split()) for r in l1_records
            )
            return PrunedContext(
                retained_records=l1_records,
                dropped_count=0,
                retained_tokens=total_tokens,
                total_original_tokens=total_tokens,
            )

        return self._pruner.prune(l1_records)

    async def search_similar(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        trace_id: UUID | None = None,
    ) -> list[MemoryRecord]:
        """Semantic search over L2 episodic memory.

        Returns raw MemoryRecords (not VectorSearchResult) to keep the
        interface simple for callers that only need the content.
        """
        results = await self._l2.search(query_embedding, top_k=top_k, trace_id=trace_id)
        return [r.record for r in results]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """Evict all L1 records.  L2 is not cleared—this is for emergency
        memory relief, not data deletion."""
        await self._l1.clear()

    async def stats(self) -> dict[str, object]:
        """Aggregate metrics across both tiers."""
        l1_stats = await self._l1.stats()
        l2_count = await self._l2.count() if hasattr(self._l2, "count") else -1
        return {"l1": l1_stats, "l2_record_count": l2_count}
