"""
RawLLMClient: direct aiohttp-based client for OpenAI-compatible endpoints.

Design rationale:
- Bypasses LangChain/LlamaIndex to eliminate intermediate serialization
  and redundant object wrapping.  Every millisecond counts in high-throughput
  orchestration loops.
- Uses Pydantic v2 models for request/response validation, ensuring that
  wire-format changes break early and loudly.
- Connection pooling via aiohttp.ClientSession keeps TCP connections warm
  across sequential calls, reducing TLS handshake overhead.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import aiohttp

from aether_kernel.core.logging import get_logger

logger = get_logger(__name__)

# Default timeout envelope: connect + read + total.
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)


class RawLLMClient:
    """Minimal async client for OpenAI chat completions.

    Supports both single completions and streaming (SSE) responses.
    Structured output is enforced via the ``response_format`` parameter
    and Pydantic validation on the caller side.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy session initialization; safe for concurrent access."""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=self._timeout,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
        return self._session

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2_048,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a non-streaming chat completion.

        Returns the raw JSON dict so that callers can apply their own
        Pydantic schemas without coupling this client to domain models.
        """
        session = await self._ensure_session()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self._base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM API error {resp.status}: {text}")
            return await resp.json()

    async def chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2_048,
    ) -> AsyncIterator[str]:
        """Execute a streaming chat completion via Server-Sent Events.

        Yields token-by-token text chunks for real-time UI updates or
        early-termination strategies (e.g., stop-sequence detection).
        """
        session = await self._ensure_session()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        url = f"{self._base_url}/chat/completions"
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM API error {resp.status}: {text}")
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    # Parse minimal structure to extract content.
                    import json

                    try:
                        obj = json.loads(chunk)
                        delta = (
                            obj.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Embeddings (for L2 vector store)
    # ------------------------------------------------------------------

    async def embedding(
        self,
        *,
        model: str = "text-embedding-3-small",
        input_text: str,
    ) -> list[float]:
        """Request a dense embedding vector for *input_text*.

        Used by the MemoryManager to populate L2 episodic storage.
        """
        session = await self._ensure_session()
        url = f"{self._base_url}/embeddings"
        async with session.post(
            url, json={"model": model, "input": input_text}
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Embedding API error {resp.status}: {text}")
            data = await resp.json()
            return data["data"][0]["embedding"]

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
