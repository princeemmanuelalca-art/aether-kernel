"""
ContextPruner: dynamic token-bounded windowing for LLM context optimization.

Design rationale:
- LLMs have finite context windows (4k–128k+ tokens).  Sending the entire
  conversation history on every call wastes tokens, increases latency, and
  degrades quality due to "lost in the middle" attention decay.
- The pruner applies a retention policy (recency + relevance) to drop older
  records while keeping the most salient context under a max-token threshold.
- Token counting uses a fast heuristic (words / 0.75) rather than importing
  tiktoken, keeping the dependency tree minimal.  Operators may subclass
  and override ``estimate_tokens`` to use exact tokenizers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import MemoryRecord, PrunedContext
from aether_kernel.core.types import MemoryTier

logger = get_logger(__name__)

# Heuristic: average 1 token ≈ 0.75 words for English prose.
# This is a fast approximation; override with tiktoken for production precision.
_WORDS_PER_TOKEN = 0.75


@runtime_checkable
class TokenEstimator(Protocol):
    """Protocol for pluggable token counting strategies."""

    def estimate_tokens(self, text: str) -> int:
        ...


class HeuristicTokenEstimator:
    """Fast word-count heuristic for environments without tiktoken."""

    def estimate_tokens(self, text: str) -> int:
        word_count = len(text.split())
        return max(1, int(word_count / _WORDS_PER_TOKEN))


class ContextPruner:
    """Prune a list of MemoryRecords to fit within a token budget.

    The default policy is:
        1. Always retain the most recent N records (recency bias).
        2. Compute remaining token budget.
        3. Greedily retain earlier records until the budget is exhausted.
        4. Everything else is dropped.

    This is O(N log N) due to the sort and O(N) for the greedy pass,
    which is negligible for context sizes (<10k records).
    """

    def __init__(
        self,
        *,
        max_tokens: int = 4_096,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._estimator = token_estimator or HeuristicTokenEstimator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, records: list[MemoryRecord]) -> PrunedContext:
        """Return a token-bounded subset of *records*.

        Args:
            records: Ordered list (oldest first) of conversation history.

        Returns:
            PrunedContext with retained records and accounting metadata.
        """
        if not records:
            return PrunedContext(
                retained_records=[],
                dropped_count=0,
                retained_tokens=0,
                total_original_tokens=0,
            )

        total_tokens = self._count_batch(records)
        if total_tokens <= self._max_tokens:
            # Fast path: no pruning needed.
            return PrunedContext(
                retained_records=list(records),
                dropped_count=0,
                retained_tokens=total_tokens,
                total_original_tokens=total_tokens,
            )

        # Strategy: always keep the most recent 30% of records, then
        # greedily keep older records that fit in the remaining budget.
        # This preserves the immediate conversation context while allowing
        # deeper history when token budget permits.
        recent_cutoff = max(1, len(records) // 3)
        recent_records = records[-recent_cutoff:]
        older_records = records[:-recent_cutoff]

        recent_tokens = self._count_batch(recent_records)
        remaining_budget = self._max_tokens - recent_tokens

        retained_older: list[MemoryRecord] = []
        consumed = 0
        for record in older_records:
            tokens = self._estimate_record(record)
            if consumed + tokens <= remaining_budget:
                retained_older.append(record)
                consumed += tokens
            else:
                break

        retained = retained_older + recent_records
        retained_tokens = self._count_batch(retained)
        dropped = len(records) - len(retained)

        logger.info(
            "Pruned %d -> %d records (%d tokens -> %d tokens)",
            len(records),
            len(retained),
            total_tokens,
            retained_tokens,
        )

        return PrunedContext(
            retained_records=retained,
            dropped_count=dropped,
            retained_tokens=retained_tokens,
            total_original_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_record(self, record: MemoryRecord) -> int:
        # Use the denormalized token_count if available; otherwise compute.
        if record.token_count > 0:
            return record.token_count
        return self._estimator.estimate_tokens(record.content)

    def _count_batch(self, records: list[MemoryRecord]) -> int:
        return sum(self._estimate_record(r) for r in records)
