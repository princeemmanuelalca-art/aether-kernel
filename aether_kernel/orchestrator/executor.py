"""
DAGExecutor: the top-level orchestrator that walks a TaskDefinition through
its ordered step sequence, dispatching agent calls via the EventBroker and
advancing state through the TaskStateManager.

Architectural invariants:
- Exactly one TaskState exists per task_id at any time.
- State transitions are serialised per-task; independent tasks run concurrently.
- Agent outputs are validated before state advances; failures route through
  the Critic-Reflective Loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError

from aether_kernel.core.exceptions import CriticLoopExhaustedError, TaskStateError
from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import (
    AgentMessage,
    CriticFeedback,
    TaskDefinition,
    TaskState,
)
from aether_kernel.core.types import AgentRole, TaskStatus
from aether_kernel.orchestrator.broker import EventBroker
from aether_kernel.orchestrator.critic import CriticReflectiveLoop
from aether_kernel.orchestrator.state_machine import TaskStateManager

logger = get_logger(__name__)

# Type alias for user-provided agent callables.
# The orchestrator does not import LLM SDKs; it delegates to these callables.
AgentCallable = Callable[[dict[str, Any]], Awaitable[str | dict[str, Any]]]


class DAGExecutor:
    """Execute TaskDefinition instances as asynchronous DAGs over agent topologies.

    Example:
        executor = DAGExecutor(broker, state_manager, critic)
        executor.register_agent(AgentRole.PLANNER, my_planner_fn)
        task_state = await executor.submit(task_definition)
        final_state = await executor.run_to_completion(task_state.task_id)
    """

    def __init__(
        self,
        broker: EventBroker,
        state_manager: TaskStateManager,
        critic: CriticReflectiveLoop,
    ) -> None:
        self._broker = broker
        self._state_manager = state_manager
        self._critic = critic

        # Registry of user-provided agent callables.
        self._agents: dict[AgentRole, AgentCallable] = {}

        # Internal mapping of task_id -> asyncio.Task for cancellation.
        self._running_executions: dict[UUID, asyncio.Task[TaskState]] = {}
        self._execution_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, role: AgentRole, callable_: AgentCallable) -> None:
        """Bind an async callable to an AgentRole.

        The callable receives a dict payload (accumulated context) and must
        return either a JSON string or a dict that validates against the
        expected output schema for that role.
        """
        self._agents[role] = callable_

    # ------------------------------------------------------------------
    # Submission & execution
    # ------------------------------------------------------------------

    async def submit(self, definition: TaskDefinition) -> TaskState:
        """Create task state, register broker consumers, and dispatch first step."""
        state = await self._state_manager.create_task(definition)
        # Register broker handler for each step role that will receive output.
        for role in definition.steps:
            self._broker.register_handler(role, self._make_handler(role))
        await self._broker.start()
        await self._state_manager.transition(
            task_id=state.task_id,
            to_status=TaskStatus.DISPATCHED,
        )
        return state

    async def run_to_completion(self, task_id: UUID) -> TaskState:
        """Drive the task from DISPATCHED through to a terminal state.

        Returns the final TaskState (COMPLETED, FAILED, or CANCELLED).
        """
        async with self._execution_lock:
            task = asyncio.create_task(
                self._execute_loop(task_id), name=f"dag-exec-{task_id}"
            )
            self._running_executions[task_id] = task

        try:
            return await task
        except asyncio.CancelledError:
            logger.warning("Execution cancelled for task %s", task_id)
            raise
        finally:
            self._running_executions.pop(task_id, None)

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def _execute_loop(self, task_id: UUID) -> TaskState:
        """Sequential step driver with Critic-Reflective retry integration."""
        state = await self._state_manager.transition(
            task_id=task_id, to_status=TaskStatus.RUNNING
        )
        definition = state.definition

        while state.current_step_index < len(definition.steps):
            step_role = definition.steps[state.current_step_index]
            agent_callable = self._agents.get(step_role)
            if agent_callable is None:
                raise TaskStateError(
                    f"No agent registered for role {step_role.value}",
                    context={"task_id": str(task_id)},
                )

            # Build input: seed context + accumulator from prior steps.
            input_payload = {
                **definition.context,
                **state.accumulator,
                "_step_index": state.current_step_index,
                "_step_role": step_role.value,
            }

            try:
                raw_output = await agent_callable(input_payload)
            except Exception as exc:
                logger.exception("Agent %s raised exception", step_role.value)
                return await self._fail(task_id, str(exc))

            # Attempt validation.  If it fails, route through Critic loop.
            try:
                validated = self._validate_output(raw_output, step_role)
            except ValidationError as exc:
                try:
                    state = await self._handle_validation_failure(
                        task_id=task_id,
                        role=step_role,
                        raw_output=raw_output,
                        validation_error=str(exc),
                    )
                    # If critic says retry, continue without advancing step.
                    continue
                except CriticLoopExhaustedError:
                    return await self._fail(task_id, f"Critic loop exhausted: {exc}")

            # Commit validated output and advance.
            message = AgentMessage(
                trace_id=task_id,
                source=step_role,
                target=self._next_role(definition, state.current_step_index),
                payload=validated if isinstance(validated, dict) else validated.model_dump(),
            )
            state = await self._state_manager.transition(
                task_id=task_id,
                to_status=TaskStatus.RUNNING,
                step_output=message,
            )
            state = await self._state_manager.advance_step(task_id)

        # Terminal state.
        state = await self._state_manager.transition(
            task_id=task_id, to_status=TaskStatus.COMPLETED
        )
        return state

    # ------------------------------------------------------------------
    # Validation & Critic integration
    # ------------------------------------------------------------------

    def _validate_output(
        self, raw_output: str | dict[str, Any], role: AgentRole
    ) -> BaseModel | dict[str, Any]:
        """Validate agent output against a role-specific schema.

        When no schema is registered, the output is returned as-is (pass-through
        for unstructured agents).  Structured agents should register schemas
        via a future SchemaRegistry.
        """
        # Placeholder: in production, maintain role->BaseModel mapping.
        if isinstance(raw_output, str):
            import json
            return json.loads(raw_output)
        return raw_output

    async def _handle_validation_failure(
        self,
        *,
        task_id: UUID,
        role: AgentRole,
        raw_output: str | dict[str, Any],
        validation_error: str,
    ) -> TaskState:
        """Route failed validation through the Critic-Reflective Loop."""
        raw_str = raw_output if isinstance(raw_output, str) else str(raw_output)
        state = await self._state_manager._unsafe_get(task_id)
        decision = await self._critic.evaluate(
            agent_role=role,
            raw_output=raw_str,
            validation_error=validation_error,
            attempt=state.critic_attempts,
        )
        if decision.value == "escalate":
            raise CriticLoopExhaustedError(f"Role {role.value} failed validation")
        feedback = self._critic.build_feedback(
            raw_output=raw_str,
            validation_error=validation_error,
            attempt_number=state.critic_attempts + 1,
        )
        state = await self._state_manager.apply_critic_retry(
            task_id=task_id, feedback=feedback
        )
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_handler(
        self, role: AgentRole
    ) -> Callable[[AgentMessage], Awaitable[None]]:
        """Factory for broker handlers that log and persist messages."""
        async def handler(message: AgentMessage) -> None:
            logger.info(
                "Broker delivered msg %s -> %s",
                message.source.value,
                message.target.value,
                extra={"context": {"trace_id": str(message.trace_id)}},
            )
        return handler

    @staticmethod
    def _next_role(definition: TaskDefinition, current_index: int) -> AgentRole:
        """Return the role that should consume the output of step *current_index*."""
        if current_index + 1 < len(definition.steps):
            return definition.steps[current_index + 1]
        # Terminal step: route back to supervisor for completion.
        return AgentRole.SUPERVISOR

    async def _fail(self, task_id: UUID, reason: str) -> TaskState:
        """Transition task to FAILED and return terminal state."""
        logger.error("Task %s FAILED: %s", task_id, reason)
        return await self._state_manager.transition(
            task_id=task_id, to_status=TaskStatus.FAILED
        )

    async def cancel(self, task_id: UUID) -> None:
        """Request cancellation of an in-flight task."""
        task = self._running_executions.get(task_id)
        if task is not None:
            task.cancel()
        await self._state_manager.transition(
            task_id=task_id, to_status=TaskStatus.CANCELLED
        )
