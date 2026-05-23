"""
TaskStateManager: deterministic finite-state machine with async locking.

Each TaskState instance is guarded by its own asyncio.Lock, enabling
concurrent execution of independent tasks while serializing mutations
on any single task trace.  This eliminates race conditions when the
Critic-Reflective Loop retries interleave with step completion events.

State-transition graph (valid edges):
    PENDING -> DISPATCHED -> RUNNING -> {COMPLETED, AWAITING_CRITIC, FAILED}
    AWAITING_CRITIC -> CRITIC_RETRY -> RUNNING
    CRITIC_RETRY -> RUNNING (re-enter current step with feedback)
    {PENDING, DISPATCHED, RUNNING, AWAITING_CRITIC, CRITIC_RETRY} -> CANCELLED
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from aether_kernel.core.exceptions import CriticLoopExhaustedError, TaskStateError
from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import (
    AgentMessage,
    CriticFeedback,
    TaskDefinition,
    TaskState,
)
from aether_kernel.core.types import AgentRole, TaskStatus

logger = get_logger(__name__)

# Adjacency list defining legal transitions.
# Centralizing this graph makes the state machine auditable and testable.
_LEGAL_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.DISPATCHED, TaskStatus.CANCELLED},
    TaskStatus.DISPATCHED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.AWAITING_CRITIC,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.AWAITING_CRITIC: {
        TaskStatus.CRITIC_RETRY,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.CRITIC_RETRY: {
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COMPLETED: set(),  # Terminal
    TaskStatus.FAILED: set(),     # Terminal
    TaskStatus.CANCELLED: set(),  # Terminal
}


class TaskStateManager:
    """In-memory registry of task states with per-task async locking.

    Production note: for multi-node deployments this must be backed by
    a distributed lock (Redis Redlock / PostgreSQL advisory locks) and
    persistent store so that task state survives process restarts.
    """

    def __init__(self) -> None:
        # task_id -> TaskState
        self._states: dict[UUID, TaskState] = {}
        # task_id -> asyncio.Lock  (one lock per task for fine-grained concurrency)
        self._locks: dict[UUID, asyncio.Lock] = {}
        # Global registry lock protects _states and _locks dict mutations.
        self._registry_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registry operations
    # ------------------------------------------------------------------

    async def create_task(self, definition: TaskDefinition) -> TaskState:
        """Allocate a new TaskState and its exclusive lock."""
        task_id = definition.task_id
        async with self._registry_lock:
            if task_id in self._states:
                raise TaskStateError(f"Task {task_id} already exists")
            state = TaskState(task_id=task_id, definition=definition)
            self._states[task_id] = state
            self._locks[task_id] = asyncio.Lock()
            logger.info("Task created", extra={"context": {"task_id": str(task_id)}})
            return state

    async def get_state(self, task_id: UUID) -> TaskStatus:
        """Read-only status query; does not acquire the per-task lock."""
        async with self._registry_lock:
            state = self._states.get(task_id)
            if state is None:
                raise TaskStateError(f"Task {task_id} not found")
            return state.status

    async def acquire_task_lock(self, task_id: UUID) -> asyncio.Lock:
        """Return the per-task lock for explicit scoped acquisition.

        Usage pattern:
            lock = await manager.acquire_task_lock(task_id)
            async with lock:
                ...
        """
        async with self._registry_lock:
            lock = self._locks.get(task_id)
            if lock is None:
                raise TaskStateError(f"Task {task_id} not found")
            return lock

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def transition(
        self,
        *,
        task_id: UUID,
        to_status: TaskStatus,
        step_output: AgentMessage | None = None,
    ) -> TaskState:
        """Atomically transition *task_id* to *to_status* if legal.

        This is the ONLY pathway through which TaskState.status may change.
        All mutations are serialized by the per-task lock.
        """
        async with await self.acquire_task_lock(task_id):
            state = await self._unsafe_get(task_id)
            from_status = state.status

            if to_status not in _LEGAL_TRANSITIONS[from_status]:
                raise TaskStateError(
                    f"Illegal transition: {from_status.value} -> {to_status.value}",
                    context={"task_id": str(task_id)},
                )

            state.status = to_status
            if step_output is not None:
                state.step_outputs[state.current_step_index] = step_output
                # Merge the output payload into the global accumulator so that
                # subsequent steps inherit context without coupling to prior
                # step internals.
                self._merge_accumulator(state, step_output.payload)

            logger.info(
                "Task %s transitioned %s -> %s",
                task_id,
                from_status.value,
                to_status.value,
                extra={"context": {"task_id": str(task_id), "step": state.current_step_index}},
            )
            return state

    async def advance_step(self, task_id: UUID) -> TaskState:
        """Increment step index after a successful step execution.

        This is separate from ``transition`` to allow the orchestrator to
        commit step output (transition -> RUNNING) before deciding whether
        the DAG has more steps.
        """
        async with await self.acquire_task_lock(task_id):
            state = await self._unsafe_get(task_id)
            state.current_step_index += 1
            # Reset critic counter on successful advance.
            state.critic_attempts = 0
            logger.info(
                "Task %s advanced to step %d",
                task_id,
                state.current_step_index,
                extra={"context": {"task_id": str(task_id)}},
            )
            return state

    async def apply_critic_retry(
        self,
        *,
        task_id: UUID,
        feedback: CriticFeedback,
    ) -> TaskState:
        """Handle Critic-Reflective Loop retry bookkeeping.

        Raises:
            CriticLoopExhaustedError: If retries exceed the task's configured max.
        """
        async with await self.acquire_task_lock(task_id):
            state = await self._unsafe_get(task_id)
            max_retries = state.definition.max_critic_retries

            if state.critic_attempts >= max_retries:
                raise CriticLoopExhaustedError(
                    f"Agent exceeded {max_retries} critic retries",
                    context={"task_id": str(task_id), "attempts": state.critic_attempts},
                )

            state.critic_attempts += 1
            # Inject feedback into the accumulator so the executor can
            # self-correct on its next invocation.
            feedback_key = f"_critic_feedback_step_{state.current_step_index}"
            state.accumulator[feedback_key] = feedback.model_dump()

            # Revert status so the orchestrator will re-dispatch the current step.
            state.status = TaskStatus.RUNNING

            logger.info(
                "Critic retry %d/%d for task %s step %d",
                state.critic_attempts,
                max_retries,
                task_id,
                state.current_step_index,
                extra={"context": {"task_id": str(task_id)}},
            )
            return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _unsafe_get(self, task_id: UUID) -> TaskState:
        """Dereference state assuming caller holds the per-task lock."""
        state = self._states.get(task_id)
        if state is None:
            raise TaskStateError(f"Task {task_id} not found")
        return state

    @staticmethod
    def _merge_accumulator(state: TaskState, payload: dict[str, Any]) -> None:
        """Deep-merge step output into the task accumulator.

        Nested dicts are merged; scalar values are overwritten.
        Keys prefixed with '_' are reserved for system metadata (e.g.,
        CriticFeedback) and are always overwritten.
        """
        for key, value in payload.items():
            if key.startswith("_"):
                state.accumulator[key] = value
            elif isinstance(value, dict) and isinstance(state.accumulator.get(key), dict):
                state.accumulator[key].update(value)
            else:
                state.accumulator[key] = value
