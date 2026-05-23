"""
Module 2: Hierarchical Memory Interface.

Provides tiered storage with latency-appropriate access patterns:
  L1 (hot):  in-process LRU cache for sub-millisecond context retrieval.
  L2 (episodic): stubbed vector DB interface for semantic similarity search.

Also exposes ContextPruner for token-window management before LLM calls.
"""
