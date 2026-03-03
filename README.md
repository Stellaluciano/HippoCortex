# HippoCortex

> **HippoCortex** is a neuroscience-inspired **memory OS** for **AI agents**: capture fast **episodic memory**, distill via **memory consolidation**, and retrieve durable **semantic memory** for reliable **long-term memory** beyond the **LLM context window**.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](./pyproject.toml)
[![Status](https://img.shields.io/badge/status-PR1%20foundation-orange.svg)](./CHANGELOG.md)

- Build an engineering-first memory substrate for production-oriented AI systems.
- Separate short-horizon traces from durable knowledge, inspired by **hippocampus** ↔ **neocortex**.
- Start locally with zero external services; integrate external stores/providers later.

---

## Project metadata

**GitHub description (350 chars):**
HippoCortex is a dual-memory OS for AI agents, separating episodic memory capture (hippocampus) from semantic memory retrieval (neocortex) with replay-based memory consolidation. It ships local-first defaults, clear interfaces, and a practical path from short-term context to long-term memory for robust agent memory systems in production.

**Suggested repository name:** `hippocortex`

**Primary language:** Python 3.11+

---

## Why HippoCortex?

Most agent stacks still treat memory as prompt stuffing. That works until scale, cost, and reliability become painful.

### Pain today

- Context gets dropped when the **LLM context window** fills up.
- Agent behavior drifts without structured **agent memory**.
- Naive transcript retrieval mixes transient chatter with durable facts.

### Before / After

| State | Before HippoCortex | After HippoCortex |
|---|---|---|
| Memory model | Flat chat history | Layered memory architecture |
| Short-term handling | Prompt-only | Working memory + episodic store |
| Long-term handling | Ad hoc RAG | Semantic index + provenance |
| Learning loop | Manual summaries | Built-in replay-driven **memory consolidation** |

---

## Architecture

<img width="2379" height="1380" alt="hippocortex_architecture" src="https://github.com/user-attachments/assets/224c4836-a1c4-480f-803c-f54c157f1654" />


---

## Quickstart

The PR1 quickstart is intentionally tiny and runnable with Python stdlib only.

```bash
python examples/quickstart.py
```

Expected flow:

1. Create `HippoCortexMemory`.
2. Add 3 episodic events.
3. Consolidate into 2 semantic notes.
4. Search semantic notes and print ranked results.

---

## Concepts

### 1) Hippocampus layer (episodic memory)

- High-throughput event capture.
- Session-grounded temporal traces.
- Optimized for “what just happened?”

### 2) Cortex layer (semantic memory)

- Durable semantic notes and facts.
- Query-oriented retrieval.
- Better fit for **long-term memory** than raw transcript replay.

### 3) Consolidation (replay)

- Periodically distills episodic traces into semantic notes.
- Keeps provenance from episodic events to semantic artifacts.
- Supports future LLM-backed and heuristic distillers.

---

## Integrations (planned)

- Agent frameworks: LangGraph, CrewAI, custom orchestration.
- Model providers: OpenAI, Anthropic, and local models.
- Storage adapters: SQLite/Postgres, in-repo vector index, optional graph backends.

> Optional provider integrations are planned; the demo does **not** require API keys.

---


## Engineering maturity

To keep HippoCortex production-friendly as it grows:

- CI runs test jobs on Python 3.11/3.12 for every PR.
- Dependabot tracks Python and GitHub Actions dependency updates weekly.
- Contribution rules require docs/changelog updates for user-visible behavior.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor workflow.

---

## Roadmap

### MVP (this stage)

- [x] Project framing + docs + architecture assets.
- [x] Runnable quickstart story.
- [x] Governance + contribution metadata.

### v0.1

- [ ] Stable Python SDK (`add_event`, `consolidate`, `search`, `build_context`).
- [ ] Pluggable embedding + vector backends.
- [ ] Replay strategies and policy hooks.

### v1.0

- [ ] Robust adapter ecosystem.
- [ ] Benchmark suite + quality gates.
- [ ] Multi-agent memory partitioning and observability.

---

## Benchmarks

Benchmarks are **coming soon**.

Planned benchmark dimensions:

- Consolidation quality vs baseline transcript retrieval.
- Retrieval precision@k for semantic notes.
- Latency/cost under increasing session length.
- Memory utility for downstream task completion.

---

## FAQ

### Is this related to LSTM?

No. HippoCortex is not an RNN/LSTM architecture. It is an application-layer **memory OS** for **AI agents** inspired by biological memory systems (**hippocampus** and **neocortex**).

### Do I need OpenAI to run the demo?

No. The quickstart runs locally with Python standard library only.

### Is this a vector database?

No. HippoCortex is a memory architecture and SDK surface. It can use vector indices but also orchestrates episodic capture, replay, routing, and consolidation.

### Where does agent memory live?

PR1 quickstart uses in-memory Python structures. The broader project direction includes local-first persistence and optional external adapters.

---

## Citation / Acknowledgements

If HippoCortex helps your work, please cite this repository and star it to support development.

- Inspiration: systems neuroscience framing of episodic-to-semantic consolidation.
- Practical lineage: production lessons from RAG, agent orchestration, and memory middleware.

```bibtex
@software{hippocortex2026,
  title = {HippoCortex: Dual-Memory OS for AI Agents},
  author = {HippoCortex Contributors},
  year = {2026},
  url = {https://github.com/<org>/hippocortex}
}
```

---

## Contributing and community

- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Report vulnerabilities via [SECURITY.md](SECURITY.md)
- Review release updates in [CHANGELOG.md](CHANGELOG.md)

---

## Keywords

HippoCortex focuses on:

- **AI agents**
- **agent memory**
- **LLM context window**
- **long-term memory**
- **episodic memory**
- **semantic memory**
- **memory consolidation**
- **hippocampus**
- **neocortex**
- **memory OS**


---

## Design principles

1. **Local-first by default**
   - A developer should run the project in minutes.
   - Cloud dependencies should be optional, not required.

2. **Production-minded interfaces**
   - Keep APIs small and composable.
   - Avoid hard-coding provider assumptions.

3. **Memory lifecycle over static storage**
   - Capture events quickly.
   - Distill high-value patterns.
   - Retrieve only what matters for the current call.

4. **Clear boundaries**
   - Working memory is not long-term memory.
   - Episodic traces are not semantic facts.
   - Consolidation policy is a first-class concern.

---

## What success looks like

- Agents remember stable user preferences without carrying giant prompts.
- Teams can inspect how a memory was produced (provenance and replay steps).
- Memory quality improves over time as consolidation strategies evolve.
- System behavior remains legible to engineers and operators.

---

## Getting started for maintainers

```bash
# run quickstart
python examples/quickstart.py

# regenerate architecture asset
python scripts/render_diagram.py
```

Then open a PR with:

- Updated docs if behavior changes.
- A changelog entry for user-visible changes.
- A minimal reproducible example when fixing bugs.

---

## Status

This is the first PR-oriented foundation.

- Docs and architecture are intentionally explicit.
- Quickstart demonstrates the intended end-user story.
- The full SDK surface will continue to evolve in subsequent milestones.
