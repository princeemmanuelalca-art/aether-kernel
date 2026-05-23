"""
Pydantic v2 schemas for all cross-module data structures.

Every boundary—agent input/output, memory records, sandbox payloads—
is modeled as a BaseModel to ensure structural correctness at runtime
and generate JSON Schemas for downstream documentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from aether_kernel.core.types import AgentRole, MemoryTier, TaskStatus


# ---------------------------------------------------------------------------
# Timestamp mixin
# ---------------------------------------------------------------------------


class Timestamped(BaseModel):
    """Immutable timestamp tracking for auditability and TTL pruning."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Agent & task schemas
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """Envelope for all inter-agent communication over the EventBroker.

    The payload field is intentionally typed as dict[str, Any] to preserve
    flexibility across heterogeneous agents while schema_version enables
    forward-compatible migration.
    """

    message_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID  # Correlates all messages in a single task DAG execution
    source: AgentRole
    target: AgentRole
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=1, ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    attempt: int = Field(default=0, ge=0, description="Retry counter for Critic-Reflective Loop")


class TaskDefinition(BaseModel):
    """Immutable specification submitted to the orchestrator to initiate a DAG run.

    The steps list defines the static topology; the orchestrator dynamically
    routes outputs based on this shape plus runtime state.
    """

    task_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=256)
    description: str = Field(default="")
    steps: list[AgentRole] = Field(..., min_length=1, description="Ordered DAG node sequence")
    context: dict[str, Any] = Field(default_factory=dict, description="Seed context for Planner")
    max_critic_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: float = Field(default=300.0, ge=1.0)


class TaskState(Timestamped):
    """Mutable runtime state for an in-flight task.

    This is the single source of truth for the orchestrator's state machine.
    All mutations pass through TaskStateManager with async locking.
    """

    task_id: UUID
    definition: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    current_step_index: int = Field(default=0, ge=0)
    step_outputs: dict[int, AgentMessage] = Field(default_factory=dict)
    accumulator: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged context accumulator passed between steps",
    )
    critic_attempts: int = Field(default=0, ge=0)

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, v: TaskStatus | str) -> TaskStatus:
        # Permissive coercion so Redis / wire-deserialized strings reconcile
        if isinstance(v, str):
            return TaskStatus(v)
        return v


class CriticFeedback(BaseModel):
    """Structured feedback injected into the retry context when validation fails.

    The executor re-reads this feedback (via the accumulator) and must
    produce corrected output that satisfies the original schema.
    """

    failed_output: str = Field(..., description="The raw output that failed validation")
    validation_error: str = Field(..., description="Human-readable schema violation details")
    guidance: str = Field(..., description="Actionable critique for the agent to self-correct")
    attempt_number: int = Field(..., ge=1)


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------


class MemoryRecord(BaseModel):
    """Unified record stored across all memory tiers.

    tier_id is denormalized into the record so that L1 and L2 views
    remain consistent without requiring a join / secondary lookup.
    """

    record_id: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    tier: MemoryTier
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class VectorSearchResult(BaseModel):
    """Result envelope from the L2 episodic vector store."""

    record: MemoryRecord
    score: float = Field(..., ge=0.0, le=1.0)


class PrunedContext(BaseModel):
    """Output of the ContextPruner: a token-bounded conversation window."""

    retained_records: list[MemoryRecord]
    dropped_count: int = Field(..., ge=0)
    retained_tokens: int = Field(..., ge=0)
    total_original_tokens: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Sandbox schemas
# ---------------------------------------------------------------------------


class SandboxPayload(BaseModel):
    """Code payload submitted to the Docker sandbox for isolated execution.

    language restricts interpreters; dependencies lists pip packages to
    pre-install inside the ephemeral container.
    """

    code: str = Field(..., min_length=1, description="Source code string to execute")
    language: str = Field(default="python", pattern=r"^(python|bash)$")
    dependencies: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    memory_limit_mb: int = Field(default=256, ge=64, le=2048)
    enable_network: bool = Field(default=False, description="Must remain False for security")

    @field_validator("enable_network")
    @classmethod
    def network_must_stay_disabled(cls, v: bool) -> bool:
        # Defense-in-depth: orchestrator should never enable network.
        if v:
            raise ValueError("Network access is prohibited in sandboxed execution")
        return v


class SandboxResult(BaseModel):
    """Immutable result of a sandboxed code execution run."""

    sandbox_id: UUID = Field(default_factory=uuid4)
    status: str  # SandboxStatus value (str to avoid circular import coupling)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = Field(..., ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
