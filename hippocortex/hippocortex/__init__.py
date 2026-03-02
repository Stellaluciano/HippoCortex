from __future__ import annotations

from dataclasses import dataclass

from hippocortex.config import HippoConfig
from hippocortex.consolidation.replay import ReplayConsolidator
from hippocortex.cortex.semantic_store import InMemorySemanticStore
from hippocortex.embedders.base import Embedder
from hippocortex.embedders.dummy_embedder import DummyEmbedder
from hippocortex.hippo.episodic_store import SQLiteEpisodicStore
from hippocortex.router import MemoryRouter
from hippocortex.types import ConsolidationOutput, ContextPack
from hippocortex.working_memory import WorkingMemory


class CortexAPI:
    def __init__(self, sdk: "HippoCortex") -> None:
        self._sdk = sdk

    def search(self, agent_id: str, query: str, k: int = 5, filters: dict | None = None):
        vector = self._sdk.embedder.embed_text(query)
        return self._sdk.semantic_store.search(agent_id=agent_id, query_vector=vector, k=k, filters=filters)


@dataclass
class HippoCortex:
    config: HippoConfig
    embedder: Embedder

    def __post_init__(self) -> None:
        self.config.ensure_parent_dir()
        self.hippo = SQLiteEpisodicStore(self.config.db_path)
        self.semantic_store = InMemorySemanticStore(self.embedder.dimension)
        self.cortex = CortexAPI(self)
        self.router = MemoryRouter()
        self.working_memory = WorkingMemory(max_recent_turns=self.config.working_memory_turns)
        self._consolidator = ReplayConsolidator(replay_size=self.config.replay_episodes)

    @classmethod
    def default(cls, config: HippoConfig | None = None, embedder: Embedder | None = None) -> "HippoCortex":
        cfg = config or HippoConfig.from_env()
        emb = embedder or DummyEmbedder(dimension=cfg.embedding_dim)
        return cls(config=cfg, embedder=emb)

    def consolidate(self, agent_id: str, session_id: str | None = None, strategy: str = "replay_v1") -> ConsolidationOutput:
        if strategy != "replay_v1":
            raise ValueError(f"Unknown consolidation strategy: {strategy}")
        episodes = self._consolidator.select_episodes(self.hippo, agent_id=agent_id, session_id=session_id)
        return self._consolidator.run(agent_id=agent_id, episodes=episodes, embedder=self.embedder, semantic_store=self.semantic_store)

    def build_context(self, agent_id: str, session_id: str, user_message: str, max_tokens: int) -> ContextPack:
        decision = self.router.route(user_message=user_message, max_tokens=max_tokens)
        recent_events = self.hippo.list_events(agent_id=agent_id, session_id=session_id, limit=self.config.working_memory_turns)
        recent_events = list(reversed(recent_events))
        selected_recent = self.working_memory.select_recent(recent_events, token_budget=decision.working_memory_tokens)

        semantic_notes = []
        if decision.intent in {"semantic", "hybrid"}:
            semantic_notes = self.cortex.search(agent_id=agent_id, query=user_message, k=5)

        highlights = []
        if decision.include_highlights:
            highlights = self.hippo.top_events_by_importance(agent_id=agent_id, session_id=session_id, limit=3)

        return ContextPack(
            recent_turns=selected_recent,
            semantic_notes=semantic_notes,
            episodic_highlights=highlights,
            token_budget=max_tokens,
        )


__all__ = ["HippoCortex", "HippoConfig", "DummyEmbedder"]
