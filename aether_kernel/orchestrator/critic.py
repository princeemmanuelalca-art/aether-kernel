"""
Critic-Reflective Loop: deterministic fallback for schema-validation failures.

When an agent produces output that fails Pydantic validation, the system does
not immediately fail the task. Instead, the output is routed to a Critic
handler that generates structured feedback.  The original agent is then
re-invoked with the feedback injected into its context accumulator.

This pattern—borrowed from ReAct / Reflexion literature but implemented here
without framework coupling—dramatically reduces end-to-end failure rates for
structured-generation tasks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import AgentMessage, CriticFeedback
from aether_kernel.core.types import AgentRole, CriticDecision

logger = get_logger(__name__)


class CriticReflectiveLoop:
    """Deterministic validation-feedback-retry orchestrator.

    Usage (inside the DAGExecutor step loop):
        1. Agent produces raw_output (dict or JSON string).
        2. ``loop.validate(raw_output, TargetSchema)`` raises ValidationError.
        3. ``decision = await loop.evaluate(...)`` returns RETRY or ESCALATE.
        4. If RETRY, ``feedback = loop.build_feedback(...)`` is injected into
           the accumulator and the step is re-dispatched.
    """

    def __init__(self, *, max_retries: int = 3) -> None:
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(raw_output: str | dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        """Attempt to coerce *raw_output* into *schema*.

        Args:
            raw_output: JSON string or pre-parsed dict from the LLM.
            schema: A concrete Pydantic v2 BaseModel subclass.

        Raises:
            ValidationError: If the output structurally violates the schema.
        """
        if isinstance(raw_output, str):
            return schema.model_validate_json(raw_output)
        return schema.model_validate(raw_output)

    # ------------------------------------------------------------------
    # Critic decision
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        *,
        agent_role: AgentRole,
        raw_output: str,
        validation_error: str,
        attempt: int,
    ) -> CriticDecision:
        """Return RETRY if attempts remain, otherwise ESCALATE.

        Production extension point: replace this deterministic heuristic with
        an LLM-powered Critic agent that analyzes the error and decides whether
        the failure is recoverable.  The interface (CriticDecision) remains
        unchanged, preserving backward compatibility.
        """
        if attempt < self._max_retries:
            logger.info(
                "Critic issuing RETRY for %s (attempt %d/%d)",
                agent_role.value,
                attempt,
                self._max_retries,
            )
            return CriticDecision.RETRY
        logger.warning(
            "Critic exhausted retries for %s; escalating to FAILED",
            agent_role.value,
        )
        return CriticDecision.ESCALATE

    # ------------------------------------------------------------------
    # Feedback construction
    # ------------------------------------------------------------------

    def build_feedback(
        self,
        *,
        raw_output: str,
        validation_error: str,
        attempt_number: int,
        guidance: str | None = None,
    ) -> CriticFeedback:
        """Construct structured feedback for the retry context.

        If *guidance* is not provided, auto-generate it from the error text.
        A future LLM-based Critic would produce richer, context-aware guidance.
        """
        if guidance is None:
            guidance = (
                f"Your previous output failed validation: {validation_error}. "
                f"Please correct the output and ensure all required fields are present "
                f"and correctly typed. This is retry attempt {attempt_number}."
            )
        return CriticFeedback(
            failed_output=raw_output,
            validation_error=validation_error,
            guidance=guidance,
            attempt_number=attempt_number,
        )

    # ------------------------------------------------------------------
    # Integration helper: wrap an agent call with critic protection
    # ------------------------------------------------------------------

    async def execute_with_retry(
        self,
        *,
        agent_role: AgentRole,
        raw_callable: callable,
        input_payload: dict[str, Any],
        output_schema: type[BaseModel],
        attempt: int = 0,
    ) -> BaseModel:
        """Invoke *raw_callable*, validate output, and either return or retry.

        This is the primary integration point for the DAGExecutor; it hides
        the retry machinery from the main step loop.

        Returns:
            A validated instance of *output_schema*.

        Raises:
            ValidationError: If all retries are exhausted.
        """
        raw_output = await raw_callable(input_payload)
        # Normalize to string for consistent feedback logging.
        if isinstance(raw_output, dict):
            import json
            raw_output_str = json.dumps(raw_output)
        else:
            raw_output_str = str(raw_output)

        try:
            validated = self.validate(raw_output_str, output_schema)
            return validated
        except ValidationError as exc:
            decision = await self.evaluate(
                agent_role=agent_role,
                raw_output=raw_output_str,
                validation_error=str(exc),
                attempt=attempt,
            )
            if decision == CriticDecision.ESCALATE:
                raise
            # Build feedback and recurse.  The caller (DAGExecutor) is
            # responsible for injecting feedback into the accumulator before
            # re-invoking.
            feedback = self.build_feedback(
                raw_output=raw_output_str,
                validation_error=str(exc),
                attempt_number=attempt + 1,
            )
            # Merge feedback into input so the agent can self-correct.
            input_payload["_critic_feedback"] = feedback.model_dump()
            return await self.execute_with_retry(
                agent_role=agent_role,
                raw_callable=raw_callable,
                input_payload=input_payload,
                output_schema=output_schema,
                attempt=attempt + 1,
            )
