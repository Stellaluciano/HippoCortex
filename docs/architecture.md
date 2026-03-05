# HippoCortex Architecture

HippoCortex separates memory responsibilities so agent systems can grow beyond raw transcript prompting.

## Components

### 1) AI Agent/App
The caller using LangGraph, CrewAI, or custom orchestration.

### 2) HippoCortex SDK
A thin orchestration layer exposing memory APIs and extension points.

### 3) Memory Router
Decides how to allocate context budget between recent turns and long-term recall.

### 4) Working Memory
Short-horizon context window manager for immediate conversational relevance. Input events are treated as newest → oldest, token budget is consumed latest-first, and selected turns may be reordered to chronological order only for presentation.

### 5) Hippocampus Layer (Episodic Store)
Fast capture of events with time/session provenance.

### 6) Consolidation Engine
Replay + distill pipeline that transforms episodic traces into durable semantic notes.

### 7) Cortex Layer (Semantic Index)
Retrieval-oriented layer used for long-term memory lookups.

### 8) LLM Provider + Tools/RAG
Retrieved semantic notes feed generation calls and tool plans.

### 9) Storage Adapters
Local-first by default, with optional adapters:
- SQLite / Postgres
- In-repo vector index or external vector stores
- Optional graph memory backend

## Data flow

1. Agent emits a message/event.
2. Event is captured into episodic memory.
3. Router allocates memory budget for the next model call.
4. Consolidation periodically replays episodic history and distills semantic notes.
5. Semantic retrieval augments downstream LLM/tool execution.

## Diagram sources

- Mermaid source: [`docs/architecture.mmd`](./architecture.mmd)
- Rendered PNG: generated locally via `python scripts/render_diagram.py` (binary not committed in this environment).
- Regeneration script: [`scripts/render_diagram.py`](../scripts/render_diagram.py)
