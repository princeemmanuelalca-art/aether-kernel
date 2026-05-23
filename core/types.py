"""
Domain types, enumerations, and protocol definitions for Aether-Kernel.

Centralizing type definitions eliminates magic strings/ints and enables
exhaustive pattern matching across the orchestrator, memory, and sandbox layers.
"""

from __future__ import annotations

import enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Agent lifecycle & topology
# ---------------------------------------------------------------------------


class AgentRole(str, enum.Enum):
    """Canonical roles in the multi-agent DAG.

    Using a string-backed enum ensures JSON (de)serializability while
    preserving type safety across the broker and state machine.
    """

    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"
    TOOL = "tool"


class TaskStatus(str, enum.Enum):
    """Finite state machine states for a single task trace.

    Transitions are enforced by the orchestrator state machine;
    no external caller may mutate status directly.
    """

    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    AWAITING_CRITIC = "awaiting_critic"
    CRITIC_RETRY = "critic_retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryTier(str, enum.Enum):
    """Latency tier identifiers for the hierarchical memory system."""

    L1_HOT = "l1_hot"      # In-process LRU; sub-millisecond access
    L2_EPISODIC = "l2_episodic"  # Vector DB; ~10-50ms retrieval


# ---------------------------------------------------------------------------
# Sandbox & security
# ---------------------------------------------------------------------------


class SandboxStatus(str, enum.Enum):
    """Lifecycle states for an ephemeral Docker sandbox."""

    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    CLEANING = "cleaning"
    CLEANED = "cleaned"


# ---------------------------------------------------------------------------
# Type variables & structural protocols
# ---------------------------------------------------------------------------

T = TypeVar("T")

# A validated agent output must satisfy this shape for routing.
AgentOutput: TypeAlias = dict[str, Any]

# Async callable signature expected by the EventBroker for agent nodes.
AgentCallable: TypeAlias = Callable[[AgentOutput], Awaitable[AgentOutput]]


@runtime_checkable
class Validatable(Protocol):
    """Structural protocol for outputs that expose deterministic validation.

    All Pydantic BaseModel subclasses satisfy this at runtime, but the
    protocol decouples the orchestrator from a direct Pydantic dependency
    should we need to swap the validation backend later.
    """

    def model_validate_json(self, data: str | bytes) -> "Validatable": ...

    def model_dump_json(self) -> str: ...


@runtime_checkable
class CriticRouter(Protocol):
    """Protocol for the deterministic Critic-Reflective Loop fallback.

    Implementations decide whether a failed validation should trigger
    a retry (with feedback injected into context) or escalate to FAILED.
    """

    async def evaluate(
        self,
        *,
        agent_role: AgentRole,
        raw_output: str,
        validation_error: str,
        attempt: int,
    ) -> CriticDecision: ...


class CriticDecision(str, enum.Enum):
    """Deterministic verdict from the Critic agent."""

    RETRY = "retry"
    ESCALATE = "escalate"
