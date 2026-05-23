"""
Exception hierarchy for Aether-Kernel.

All exceptions subclass AetherKernelError so that top-level supervisors can
catch a single root type while inner modules raise semantically precise
variants for structured logging and metrics.
"""

from __future__ import annotations

from typing import Any


class AetherKernelError(Exception):
    """Root exception for all engine-level failures."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


# ---------------------------------------------------------------------------
# Orchestrator errors
# ---------------------------------------------------------------------------


class OrchestratorError(AetherKernelError):
    """Base for DAG execution and state-machine failures."""


class TaskStateError(OrchestratorError):
    """Raised when an illegal state transition is attempted."""


class CriticLoopExhaustedError(OrchestratorError):
    """Raised when an agent exhausts max_critic_retries without passing validation."""


class BrokerQueueFullError(OrchestratorError):
    """Raised when the EventBroker backpressure limit is reached."""


# ---------------------------------------------------------------------------
# Memory errors
# ---------------------------------------------------------------------------


class MemoryError(AetherKernelError):
    """Base for hierarchical memory failures."""


class VectorDBConnectionError(MemoryError):
    """Raised when the L2 vector store client cannot connect or query."""


class ContextPrunerError(MemoryError):
    """Raised when token estimation or pruning logic fails."""


# ---------------------------------------------------------------------------
# Sandbox errors
# ---------------------------------------------------------------------------


class SandboxError(AetherKernelError):
    """Base for Docker sandbox execution failures."""


class SandboxTimeoutError(SandboxError):
    """Raised when code execution exceeds its allocated timeout."""


class SandboxBuildError(SandboxError):
    """Raised when the ephemeral Docker image build fails."""


class SandboxSecurityError(SandboxError):
    """Raised when a security policy violation is detected (network, volume, etc.)."""
