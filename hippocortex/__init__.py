from __future__ import annotations

from dataclasses import dataclass

from hippocortex.config import HippoConfig
from hippocortex.embedders.base import Embedder
from hippocortex.embedders.dummy_embedder import DummyEmbedder
from hippocortex.hippo.episodic_store import SQLiteEpisodicStore
from hippocortex.registry import get_consolidation_strategy, get_router_strategy, get_storage_backend, register_defaults
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
        register_defaults()
        self.config.validate()
        self.config.ensure_parent_dir()

        if self.embedder.dimension != self.config.model.embedding_dim:
            raise ValueError(
                f"Invalid config: embedder dimension {self.embedder.dimension} does not match "
                f"model.embedding_dim {self.config.model.embedding_dim}"
            )

        self.hippo = SQLiteEpisodicStore(self.config.storage.db_path)
        semantic_backend = self.config.storage.semantic_store_backend.lower()
        self.semantic_store = get_storage_backend(semantic_backend, self.config, self.embedder.dimension)
        self.cortex = CortexAPI(self)
        self.router = get_router_strategy(self.config.router.strategy)
        self.working_memory = WorkingMemory(max_recent_turns=self.config.runtime.working_memory_turns)
        self._consolidators: dict[str, object] = {}

        print(f"[HippoCortex] Effective config: {self.config.as_dict()}")

    @classmethod
    def default(cls, config: HippoConfig | None = None, embedder: Embedder | None = None) -> "HippoCortex":
        cfg = HippoConfig.from_env().merged(config)
        emb = embedder or DummyEmbedder(dimension=cfg.model.embedding_dim)
        return cls(config=cfg, embedder=emb)

    def consolidate(self, agent_id: str, session_id: str | None = None, strategy: str = "replay_v1") -> ConsolidationOutput:
        consolidator = self._consolidators.get(strategy)
        if consolidator is None:
            consolidator = get_consolidation_strategy(strategy, self.config)
            self._consolidators[strategy] = consolidator
        episodes = consolidator.select_episodes(self.hippo, agent_id=agent_id, session_id=session_id)
        return consolidator.run(agent_id=agent_id, episodes=episodes, embedder=self.embedder, semantic_store=self.semantic_store)

    def build_context(self, agent_id: str, session_id: str, user_message: str, max_tokens: int) -> ContextPack:
        decision = self.router.route(user_message=user_message, max_tokens=max_tokens)
        recent_events = self.hippo.list_events(agent_id=agent_id, session_id=session_id, limit=self.config.runtime.working_memory_turns)
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
