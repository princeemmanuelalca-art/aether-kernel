# aether-kernel

**High-Throughput Event-Driven Multi-Agent Orchestration Runtime**

A production-grade, framework-agnostic backend infrastructure for building autonomous multi-agent systems. Provides deterministic execution guarantees, hierarchical memory, and secure sandboxed code execution—with zero dependency on LangChain or LlamaIndex.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules](#modules)
  - [core](#core)
  - [orchestrator](#orchestrator)
  - [memory](#memory)
  - [llm](#llm)
  - [sandbox](#sandbox)
- [Running Tests](#running-tests)
- [Security Model](#security-model)
- [Production Notes](#production-notes)
- [License](#license)

---

## Overview

`aether-kernel` is the backend runtime for the **Kimi Agent / Aether** system. It exposes a set of composable async components that handle:

- **Multi-agent DAG execution** — tasks are defined as ordered sequences of agent roles and executed through a finite-state machine with deterministic transitions.
- **Critic-Reflective Loop** — failed agent outputs are automatically retried with structured feedback injected into the context, up to a configurable retry ceiling.
- **Hierarchical Memory** — a two-tier memory system (L1 in-process LRU cache + L2 episodic vector store) provides sub-millisecond hot-path access and semantic search over long-term history.
- **Sandboxed Code Execution** — generated Python/Bash code runs in ephemeral Docker containers with strict resource caps, network isolation, and read-only filesystems.
- **Raw LLM Client** — a minimal `aiohttp`-based OpenAI-compatible client that bypasses framework overhead and uses persistent connection pooling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TaskDefinition                       │
│         (task_id, steps: [AgentRole], context)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               TaskStateManager (FSM)                    │
│  PENDING → DISPATCHED → RUNNING → AWAITING_CRITIC       │
│                              ↘ COMPLETED / FAILED       │
│           AWAITING_CRITIC → CRITIC_RETRY → RUNNING      │
└────────────────────────┬────────────────────────────────┘
                         │  publishes AgentMessages
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    EventBroker                          │
│   Per-role asyncio.Queue (backpressure via maxsize)     │
│   Fan-out to registered async handlers                  │
└───────┬────────────┬────────────┬────────────┬──────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
   [PLANNER]   [EXECUTOR]    [CRITIC]   [SUPERVISOR]
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────────────────────┐
│               MemoryManager (L1 + L2)                   │
│   L1HotCache (LRU, in-process)                          │
│   L2VectorStore (InMemory default / pluggable Qdrant)   │
│   ContextPruner (token-bounded sliding window)          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│               RawLLMClient                              │
│   aiohttp + connection pooling                          │
│   chat_completion / streaming / embeddings              │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (when code execution needed)
┌─────────────────────────────────────────────────────────┐
│               SandboxManager                            │
│   Docker SDK (asyncio.to_thread)                        │
│   network_mode=none, cap_drop=ALL, read_only=True       │
│   Semaphore(10) concurrent execution limit              │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
aether_kernel/
├── __init__.py                  # Package entry point, version: 0.1.0
├── pyproject.toml               # Build config and dependencies
├── test_integration.py          # Integration test suite
│
├── core/                        # Shared domain primitives
│   ├── types.py                 # Enums: AgentRole, TaskStatus, MemoryTier, SandboxStatus
│   ├── schemas.py               # Pydantic v2 models for all cross-module data
│   ├── exceptions.py            # Domain-specific exception hierarchy
│   └── logging.py               # Structured logging via structlog
│
├── orchestrator/                # Task execution engine
│   ├── broker.py                # EventBroker — per-role asyncio queues with backpressure
│   ├── state_machine.py         # TaskStateManager — FSM with per-task async locking
│   ├── executor.py              # Executor agent — runs a single DAG step
│   └── critic.py                # Critic agent — validates output; triggers retry loop
│
├── memory/                      # Hierarchical memory system
│   ├── manager.py               # MemoryManager — unified L1/L2 facade
│   ├── l1_hot_cache.py          # L1HotCache — in-process LRU cache
│   ├── l2_vector_store.py       # AbstractVectorStore + InMemoryVectorStore
│   └── pruner.py                # ContextPruner — token-bounded sliding window
│
├── llm/
│   └── client.py                # RawLLMClient — aiohttp OpenAI-compatible client
│
└── sandbox/
    └── manager.py               # SandboxManager — ephemeral Docker execution
```

---

## Requirements

- Python **3.11+**
- Docker daemon accessible (for `SandboxManager`)
- An OpenAI-compatible LLM API key (for `RawLLMClient`)

---

## Installation

### From source

```bash
git clone https://github.com/<your-username>/aether-kernel.git
cd aether-kernel
pip install -e ".[dev]"
```

### Dependencies (auto-installed)

| Package | Version | Purpose |
|---|---|---|
| `pydantic` | >=2.0 | Schema validation across all module boundaries |
| `aiohttp` | >=3.9 | Async HTTP client for LLM API + connection pooling |
| `docker` | >=7.0 | Python SDK for Docker sandbox management |
| `structlog` | >=24.1 | Structured JSON logging |

### Dev dependencies

```bash
pip install -e ".[dev]"
# installs: pytest, pytest-asyncio, mypy, ruff
```

---

## Configuration

No config files are required for basic use. All components are configured via constructor arguments.

| Component | Key Parameter | Default | Notes |
|---|---|---|---|
| `EventBroker` | `queue_maxsize` | `10_000` | Messages per role queue before backpressure |
| `EventBroker` | `consumer_concurrency` | `{role: 1}` | Scale heavy roles (e.g. Executor) to N workers |
| `MemoryManager` | `l1_maxsize` | `1_000` | Max records in L1 LRU cache |
| `MemoryManager` | `l2_store` | `InMemoryVectorStore()` | Swap in Qdrant/Weaviate for production |
| `RawLLMClient` | `base_url` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `SandboxManager` | `python_image` | `python:3.11-slim` | Pin to a digest in production |
| `TaskDefinition` | `max_critic_retries` | `3` | Max Critic-Reflective Loop retries |
| `TaskDefinition` | `timeout_seconds` | `300.0` | Per-task wall-clock timeout |

---

## Usage

### Define and submit a task

```python
import asyncio
from uuid import uuid4
from aether_kernel.core.schemas import TaskDefinition
from aether_kernel.core.types import AgentRole
from aether_kernel.orchestrator.broker import EventBroker
from aether_kernel.orchestrator.state_machine import TaskStateManager

async def main():
    broker = EventBroker(consumer_concurrency={AgentRole.EXECUTOR: 4})
    state_manager = TaskStateManager()

    task = TaskDefinition(
        name="summarize-and-validate",
        steps=[AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC],
        context={"input": "Summarize the Q3 earnings report."},
        max_critic_retries=3,
        timeout_seconds=120.0,
    )

    state = await state_manager.create_task(task)
    await broker.start()
    # register your agent handlers via broker.register_handler(role, handler)
    # ...
    await broker.stop()

asyncio.run(main())
```

### Store and retrieve memory

```python
from uuid import uuid4
from aether_kernel.memory.manager import MemoryManager
from aether_kernel.core.schemas import MemoryRecord
from aether_kernel.core.types import MemoryTier

manager = MemoryManager(l1_maxsize=500)
trace_id = uuid4()

record = MemoryRecord(
    trace_id=trace_id,
    tier=MemoryTier.L1_HOT,
    content="User asked about Q3 earnings.",
    token_count=8,
)

await manager.store(record)
context = await manager.get_context(trace_id, max_tokens=4096)
```

### Execute code in a sandbox

```python
from aether_kernel.sandbox.manager import SandboxManager
from aether_kernel.core.schemas import SandboxPayload

sandbox = SandboxManager()
result = await sandbox.execute(SandboxPayload(
    code="print(sum(range(100)))",
    language="python",
    timeout_seconds=10.0,
    memory_limit_mb=128,
))
print(result.stdout)   # "4950"
print(result.status)   # "success"
await sandbox.close()
```

### Stream LLM tokens

```python
from aether_kernel.llm.client import RawLLMClient

client = RawLLMClient(api_key="sk-...", base_url="https://api.openai.com/v1")

async for token in client.chat_completion_stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain gradient descent."}],
):
    print(token, end="", flush=True)

await client.close()
```

---

## Modules

### `core`

Shared primitives used across all other modules.

- **`types.py`** — String-backed enums: `AgentRole` (PLANNER, EXECUTOR, CRITIC, SUPERVISOR, TOOL), `TaskStatus` (8-state FSM), `MemoryTier` (L1_HOT, L2_EPISODIC), `SandboxStatus`, `CriticDecision`. Also defines structural protocols `Validatable` and `CriticRouter`.
- **`schemas.py`** — Pydantic v2 `BaseModel` definitions for: `AgentMessage`, `TaskDefinition`, `TaskState`, `CriticFeedback`, `MemoryRecord`, `VectorSearchResult`, `PrunedContext`, `SandboxPayload`, `SandboxResult`.
- **`exceptions.py`** — Domain exceptions: `BrokerQueueFullError`, `TaskStateError`, `CriticLoopExhaustedError`, `SandboxBuildError`, `SandboxSecurityError`, `SandboxTimeoutError`.
- **`logging.py`** — `get_logger(name)` factory backed by `structlog` for structured JSON output.

### `orchestrator`

#### `EventBroker` (`broker.py`)

Async fan-out message bus. Each `AgentRole` gets a dedicated `asyncio.Queue` with configurable `maxsize` for backpressure. Multiple handlers per role support fan-out patterns (e.g., metrics sidecars). Consumer concurrency per role is tunable.

```
publish(AgentMessage) → queue[target_role] → consumer loop → fan-out to handlers
```

#### `TaskStateManager` (`state_machine.py`)

In-memory FSM registry. Each task gets its own `asyncio.Lock` for fine-grained concurrency. The full transition graph is encoded as an adjacency list; illegal transitions raise `TaskStateError`. The `apply_critic_retry` method injects structured `CriticFeedback` into the task accumulator and decrements the retry budget.

#### `Executor` / `Critic` (`executor.py`, `critic.py`)

Agent node implementations registered as handlers on the `EventBroker`. The Executor runs a single DAG step and emits its output as an `AgentMessage`. The Critic validates the output; on failure it raises `TaskStatus.AWAITING_CRITIC` and the state machine triggers the retry loop.

### `memory`

#### `MemoryManager` (`manager.py`)

Unified facade. Write path: all records go to L1; records with an embedding vector also go to L2. Read path: single-record reads hit L1 only; `get_context(trace_id)` assembles the full conversation window and passes it through the `ContextPruner`.

#### `L1HotCache` (`l1_hot_cache.py`)

In-process LRU cache keyed by `record_id` and indexed by `trace_id` for efficient conversation retrieval. Sub-millisecond access. Capacity-bounded to prevent unbounded memory growth.

#### `AbstractVectorStore` / `InMemoryVectorStore` (`l2_vector_store.py`)

Pluggable interface for episodic semantic memory. The default `InMemoryVectorStore` uses cosine similarity over in-process numpy arrays. Swap in Qdrant, Weaviate, or Pinecone by implementing `AbstractVectorStore`.

#### `ContextPruner` (`pruner.py`)

Token-bounded sliding window. Trims the oldest records from a trace's history until the total token count fits within the configured budget. Returns a `PrunedContext` with retention statistics.

### `llm`

#### `RawLLMClient` (`client.py`)

Minimal async client for any OpenAI-compatible endpoint. Uses `aiohttp.ClientSession` with lazy initialization and a mutex to keep connections warm across calls. Supports:

- `chat_completion(...)` — single-shot completion, returns raw JSON dict.
- `chat_completion_stream(...)` — async generator yielding token strings via SSE.
- `embedding(...)` — dense vector for L2 memory storage.

No LangChain or LlamaIndex dependency. All validation is done by the caller via Pydantic schemas.

### `sandbox`

#### `SandboxManager` (`manager.py`)

Ephemeral Docker execution engine. All blocking Docker SDK calls are offloaded to `asyncio.to_thread` / `loop.run_in_executor` to prevent event-loop starvation. A `Semaphore(10)` limits concurrent containers.

**Security constraints enforced at runtime:**

| Control | Value |
|---|---|
| `network_mode` | `none` — absolute network isolation |
| `privileged` | `False` |
| `security_opt` | `no-new-privileges:true` |
| `cap_drop` | `ALL` |
| `read_only` | `True` (root filesystem) |
| `user` | `1000:1000` (non-root) |
| `pids_limit` | `64` (prevents fork bombs) |
| `mem_limit` | Configurable, default 256 MB |
| `nano_cpus` | 1 core |

---

## Running Tests

```bash
pytest aether_kernel/test_integration.py -v
```

For async tests, `pytest-asyncio` is required (included in `[dev]` extras).

To run with type checking:

```bash
mypy aether_kernel/
```

To lint:

```bash
ruff check aether_kernel/
```

---

## Security Model

- **Sandbox network access** is unconditionally disabled. Passing `enable_network=True` in `SandboxPayload` raises a `SandboxSecurityError` at the schema level before any Docker call is made.
- **Container cleanup** is guaranteed via `finally` blocks and a `_force_cleanup` path for timeout scenarios.
- **Backpressure** via bounded queues prevents OOM under burst load; producers receive `BrokerQueueFullError` rather than silently dropping messages.
- **State mutation** is gated exclusively through `TaskStateManager.transition()`; no external caller can write `TaskState.status` directly.

---

## Production Notes

- **Distributed deployments**: `TaskStateManager` is currently in-process only. For multi-node setups, replace the internal dict/lock with Redis Redlock + a persistent store (PostgreSQL, DynamoDB) to survive process restarts.
- **L2 Vector Store**: Replace `InMemoryVectorStore` with a production vector database (Qdrant recommended). Pass your client as `l2_store` to `MemoryManager`.
- **Docker image pinning**: In production, pin `python_image` to a specific SHA digest to prevent supply-chain drift.
- **LLM error handling**: `RawLLMClient` raises `RuntimeError` on non-200 responses. Wrap calls with retry logic (e.g., `tenacity`) at the agent handler level.
- **Structured logging**: All components emit `structlog` events with a `context` dict containing `task_id`, `trace_id`, and `step`. Configure a JSON renderer for log aggregation pipelines (Datadog, Loki, CloudWatch).

---

## License

Specify your license here (e.g., MIT, Apache 2.0, proprietary).
