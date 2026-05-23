"""
L2 Episodic Vector Store: stubbed interface for semantic retrieval.

Design rationale:
- The AbstractVectorStore protocol decouples the orchestrator from any
  specific vector DB (Qdrant, Milvus, Weaviate, pgvector).  Operators can
  inject a production client without modifying upstream code.
- The InMemoryVectorStore stub uses numpy for cosine similarity, enabling
  full integration tests without external infrastructure.
- All methods are async to match the L1HotCache contract and to accommodate
  network-bound production clients (which will have ~10-50ms p99 latency).
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable
from uuid import UUID

import numpy as np

from aether_kernel.core.exceptions import VectorDBConnectionError
from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import MemoryRecord, VectorSearchResult
from aether_kernel.core.types import MemoryTier

logger = get_logger(__name__)

# Stub embedding dimension.  Production systems using OpenAI text-embedding-3
# will use 1536 (or custom dimensions with the new truncation API).
_EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class AbstractVectorStore(Protocol):
    """Structural protocol for L2 episodic storage backends.

    Implementations must be async-safe and tolerate concurrent access from
    multiple agent tasks.  The interface is intentionally minimal to reduce
    the surface area that must be replicated across backends.
    """

    async def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        """Insert or update a record with its dense embedding vector."""

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        trace_id: UUID | None = None,
    ) -> list[VectorSearchResult]:
        """Return the top-k most similar records, optionally filtered by trace."""

    async def delete(self, record_id: UUID) -> bool:
        """Remove a record by ID; returns True if it existed."""


# ---------------------------------------------------------------------------
# In-memory stub implementation
# ---------------------------------------------------------------------------


class InMemoryVectorStore:
    """Stub vector store using cosine similarity over numpy arrays.

    Thread-safe via asyncio.Lock.  Suitable for integration tests and
    single-process deployments.  Not for production at scale.
    """

    def __init__(self, *, embedding_dim: int = _EMBEDDING_DIM) -> None:
        self._dim = embedding_dim
        # record_id -> (MemoryRecord, np.ndarray)
        self._storage: dict[UUID, tuple[MemoryRecord, np.ndarray]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dim}, got {len(embedding)}"
            )
        async with self._lock:
            self._storage[record.record_id] = (record, np.array(embedding, dtype=np.float32))

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        trace_id: UUID | None = None,
    ) -> list[VectorSearchResult]:
        query = np.array(query_embedding, dtype=np.float32)
        # Normalize for cosine similarity.
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        async with self._lock:
            scored: list[tuple[float, MemoryRecord]] = []
            for record, vec in self._storage.values():
                if trace_id is not None and record.trace_id != trace_id:
                    continue
                vec_norm = np.linalg.norm(vec)
                if vec_norm == 0:
                    continue
                similarity = float(np.dot(query, vec) / vec_norm)
                scored.append((similarity, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            VectorSearchResult(record=record, score=score)
            for score, record in scored[:top_k]
        ]

    async def delete(self, record_id: UUID) -> bool:
        async with self._lock:
            existed = record_id in self._storage
            self._storage.pop(record_id, None)
            return existed

    async def count(self) -> int:
        async with self._lock:
            return len(self._storage)


# ---------------------------------------------------------------------------
# Qdrant client stub (production-oriented placeholder)
# ---------------------------------------------------------------------------


class QdrantVectorStoreStub:
    """Stub that mirrors the Qdrant Python client's API surface.

    This demonstrates the adapter pattern: when the operator deploys Qdrant,
    they replace this stub with a thin wrapper around the official
    ``qdrant_client.QdrantClient`` (or the async ``AsyncQdrantClient``).

    The interface remains ``AbstractVectorStore``-compatible, so no upstream
    code changes are required.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "aether_memory",
        embedding_dim: int = _EMBEDDING_DIM,
    ) -> None:
        self._url = url
        self._collection = collection_name
        self._dim = embedding_dim
        self._connected = False

    async def connect(self) -> None:
        """Idempotent connection establishment."""
        if self._connected:
            return
        # Placeholder: real implementation would call
        # await self._client.get_collections() and verify the collection exists.
        logger.info("QdrantVectorStoreStub connected to %s", self._url)
        self._connected = True

    async def upsert(self, record: MemoryRecord, embedding: list[float]) -> None:
        if not self._connected:
            raise VectorDBConnectionError("Qdrant store not connected")
        # Placeholder: await self._client.upsert(
        #     collection_name=self._collection,
        #     points=[PointStruct(id=record.record_id, vector=embedding, payload=...)]
        # )
        logger.debug("Qdrant upsert stub: record %s", record.record_id)

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        trace_id: UUID | None = None,
    ) -> list[VectorSearchResult]:
        if not self._connected:
            raise VectorDBConnectionError("Qdrant store not connected")
        # Placeholder: results = await self._client.search(...)
        # For stub behavior, return empty.
        return []

    async def delete(self, record_id: UUID) -> bool:
        if not self._connected:
            raise VectorDBConnectionError("Qdrant store not connected")
        # Placeholder
        return True
