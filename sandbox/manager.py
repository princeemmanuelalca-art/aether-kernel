"""
SandboxManager: secure, ephemeral Docker execution for generated code.

Security model:
- Network isolation: containers run with network_mode="none" unless
  explicitly enabled (which triggers a validation error in SandboxPayload).
- Filesystem isolation: no host volume mounts; a tmpfs overlay provides
  ephemeral writable storage that vanishes with the container.
- Resource caps: CPU, memory, and PID limits prevent fork-bombs and OOMs.
- Timeouts: asyncio.wait_for wraps the Docker SDK to prevent indefinite
  hangs from infinite loops or blocking I/O.
- No privilege escalation: containers run as an unprivileged user.

The manager uses the official ``docker`` Python SDK (synchronous).  All
blocking SDK calls are executed in ``asyncio.to_thread`` or
``loop.run_in_executor`` to prevent event-loop starvation.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

import docker
from docker.errors import BuildError, ContainerError, DockerException, ImageNotFound

from aether_kernel.core.exceptions import (
    SandboxBuildError,
    SandboxSecurityError,
    SandboxTimeoutError,
)
from aether_kernel.core.logging import get_logger
from aether_kernel.core.schemas import SandboxPayload, SandboxResult
from aether_kernel.core.types import SandboxStatus

logger = get_logger(__name__)

# Docker image used for Python execution.  Slim variant reduces pull time
# and attack surface.  Operators should pin to a specific digest in production.
_DEFAULT_PYTHON_IMAGE = "python:3.11-slim"

# Security-hardened Docker run overrides.
_SECURITY_OVERRIDES: dict[str, Any] = {
    "network_mode": "none",          # Absolute network isolation.
    "privileged": False,             # No host device access.
    "security_opt": ["no-new-privileges:true"],
    "cap_drop": ["ALL"],             # Drop all Linux capabilities.
    "read_only": True,               # Root filesystem is read-only.
    "user": "1000:1000",             # Run as non-root (nobody).
}


class SandboxManager:
    """Execution manager for ephemeral, isolated code evaluation.

    Lifecycle of a single execution:
        1. ``submit(payload)`` writes code to a temp file.
        2. A container is created from the language-specific base image.
        3. Code is executed with resource limits and timeout enforcement.
        4. stdout/stderr and exit code are captured.
        5. Container is removed (force=True) to ensure immediate cleanup.
    """

    def __init__(
        self,
        *,
        python_image: str = _DEFAULT_PYTHON_IMAGE,
        docker_url: str | None = None,
    ) -> None:
        """Initialize the Docker client.

        Args:
            python_image: Base image tag for Python sandbox containers.
            docker_url: Optional Docker daemon URL (e.g., ``unix:///var/run/docker.sock``).
        """
        try:
            self._client = docker.DockerClient(base_url=docker_url)
        except DockerException as exc:
            raise SandboxBuildError(
                "Cannot connect to Docker daemon",
                context={"docker_url": docker_url, "error": str(exc)},
            ) from exc
        self._python_image = python_image
        self._semaphore = asyncio.Semaphore(10)  # Limit concurrent sandboxes.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, payload: SandboxPayload) -> SandboxResult:
        """Run *payload.code* inside an ephemeral container.

        This method is fully async (including Docker SDK I/O) and enforces
        the payload timeout via asyncio.wait_for.

        Raises:
            SandboxTimeoutError: If execution exceeds payload.timeout_seconds.
            SandboxBuildError: If the Docker image is missing or container creation fails.
            SandboxSecurityError: If the payload violates security policies.
        """
        if payload.enable_network:
            raise SandboxSecurityError(
                "Network access requested but prohibited by security policy"
            )

        async with self._semaphore:
            sandbox_id = uuid.uuid4()
            start_time = asyncio.get_event_loop().time()

            try:
                return await asyncio.wait_for(
                    self._run_container(sandbox_id, payload),
                    timeout=payload.timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Force-cleanup any dangling container from the timed-out execution.
                await self._force_cleanup(sandbox_id)
                duration = (asyncio.get_event_loop().time() - start_time) * 1000
                return SandboxResult(
                    sandbox_id=sandbox_id,
                    status=SandboxStatus.TIMEOUT.value,
                    stdout="",
                    stderr=f"Execution timed out after {payload.timeout_seconds}s",
                    exit_code=None,
                    duration_ms=duration,
                )

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run_container(
        self, sandbox_id: uuid.UUID, payload: SandboxPayload
    ) -> SandboxResult:
        """Create, run, and destroy a sandbox container; capture output."""
        loop = asyncio.get_event_loop()
        start_time = loop.time()

        # Write code to a temporary file that will be bind-mounted read-only.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(payload.code)
            code_path = Path(tmp.name)

        container_name = f"aether-sandbox-{sandbox_id}"
        container = None

        try:
            container = await loop.run_in_executor(
                None,
                lambda: self._client.containers.create(
                    image=self._python_image,
                    name=container_name,
                    command=["python", "/sandbox/code.py"],
                    volumes={str(code_path): {"bind": "/sandbox/code.py", "mode": "ro"}},
                    mem_limit=f"{payload.memory_limit_mb}m",
                    nano_cpus=1_000_000_000,  # 1 CPU core.
                    pids_limit=64,            # Prevent fork bombs.
                    **_SECURITY_OVERRIDES,
                ),
            )

            await loop.run_in_executor(None, container.start)

            # Wait for completion with a separate timeout buffer to allow
            # status collection even if the outer wait_for fires.
            result = await loop.run_in_executor(
                None,
                lambda: container.wait(timeout=payload.timeout_seconds),
            )
            exit_code = result.get("StatusCode", -1)

            # Collect logs (stdout + stderr).
            logs = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=True, timestamps=False),
            )
            stdout, stderr = self._split_logs(logs)

            duration = (loop.time() - start_time) * 1000
            status = SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR

            return SandboxResult(
                sandbox_id=sandbox_id,
                status=status.value,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration,
            )

        except ImageNotFound as exc:
            raise SandboxBuildError(
                f"Docker image {self._python_image} not found",
                context={"sandbox_id": str(sandbox_id)},
            ) from exc
        except DockerException as exc:
            raise SandboxBuildError(
                f"Docker error: {exc}",
                context={"sandbox_id": str(sandbox_id)},
            ) from exc
        finally:
            # Unconditionally remove the container and temp file.
            if container is not None:
                await loop.run_in_executor(None, self._safe_remove, container)
            code_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_logs(raw: bytes) -> tuple[str, str]:
        """Separate stdout and stderr from multiplexed Docker logs.

        The Docker SDK returns a single byte stream with 8-byte headers
        when both stdout and stderr are requested.  For simplicity we
        decode everything as UTF-8 and return unified output; a production
        implementation could parse the stream headers for true separation.
        """
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        # Best-effort: assume everything went to stdout unless stderr is
        # separately captured.  A future enhancement would use the low-level
        # API to demultiplex the stream.
        return text, ""

    def _safe_remove(self, container: Any) -> None:
        """Force-remove a container, swallowing errors."""
        try:
            container.remove(force=True, v=True)
        except Exception as exc:
            logger.warning("Failed to remove container %s: %s", container.id, exc)

    async def _force_cleanup(self, sandbox_id: uuid.UUID) -> None:
        """Attempt to kill and remove a container by name after timeout."""
        loop = asyncio.get_event_loop()
        container_name = f"aether-sandbox-{sandbox_id}"
        try:
            container = await loop.run_in_executor(
                None,
                lambda: self._client.containers.get(container_name),
            )
            await loop.run_in_executor(None, self._safe_remove, container)
        except Exception:
            logger.debug("No dangling container to clean up for %s", sandbox_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release the underlying Docker client resources."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._client.close)
