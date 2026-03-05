from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from hippocortex.config import HippoConfig
from hippocortex.embedders.base import Embedder
from hippocortex.embedders.dummy_embedder import DummyEmbedder
from hippocortex.hippo.episodic_store import SQLiteEpisodicStore
from hippocortex.observability import configure_json_logger
from hippocortex.registry import get_consolidation_strategy, get_router_strategy, get_storage_backend, register_defaults
from hippocortex.telemetry import NoOpTelemetry, Telemetry
from hippocortex.types import ConsolidationOutput, ContextPack
from hippocortex.working_memory import WorkingMemory


logger = logging.getLogger(__name__)
configure_json_logger(logger)


class CortexAPI:
    def __init__(self, sdk: "HippoCortex") -> None:
        self._sdk = sdk

    def search(
        self,
        agent_id: str,
        query: str,
        k: int = 5,
        filters: dict | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ):
        started = perf_counter()
        vector = self._sdk.embedder.embed_text(query)
        hits = self._sdk.semantic_store.search(agent_id=agent_id, query_vector=vector, k=k, filters=filters)
        duration_ms = (perf_counter() - started) * 1000
        hit_rate = len(hits) / k if k > 0 else 0.0
        tags = {"agent_id": agent_id}
        self._sdk.telemetry.observe("hippocortex.search.duration_ms", duration_ms, tags=tags)
        self._sdk.telemetry.observe("hippocortex.search.hit_rate", hit_rate, tags=tags)
        logger.info(
            "search_completed",
            extra={
                "agent_id": agent_id,
                "session_id": session_id,
                "request_id": request_id,
                "duration_ms": duration_ms,
                "hit_rate": hit_rate,
            },
        )
        return hits


@dataclass
class HippoCortex:
    config: HippoConfig
    embedder: Embedder
    telemetry: Telemetry = field(default_factory=NoOpTelemetry)

    def __post_init__(self) -> None:
        register_defaults()
        self.config.validate()
        self.config.ensure_parent_dir()

        if self.embedder.dimension != self.config.model.embedding_dim:
            raise ValueError(
                f"Invalid config: embedder dimension {self.embedder.dimension} does not match "
                f"model.embedding_dim {self.config.model.embedding_dim}"
            )

        self.hippo = SQLiteEpisodicStore(self.config.storage.db_path, telemetry=self.telemetry)
        semantic_backend = self.config.storage.semantic_store_backend.lower()
        self.semantic_store = get_storage_backend(semantic_backend, self.config, self.embedder.dimension)
        self.cortex = CortexAPI(self)
        self.router = get_router_strategy(self.config.router.strategy)
        self.working_memory = WorkingMemory(max_recent_turns=self.config.runtime.working_memory_turns)
        self._consolidators: dict[str, object] = {}

        logger.info("HippoCortex effective config: %s", self.config.as_dict())

    @classmethod
    def default(cls, config: HippoConfig | None = None, embedder: Embedder | None = None) -> "HippoCortex":
        cfg = HippoConfig.from_env().merged(config)
        emb = embedder or DummyEmbedder(dimension=cfg.model.embedding_dim)
        return cls(config=cfg, embedder=emb)

    def consolidate(
        self,
        agent_id: str,
        session_id: str | None = None,
        strategy: str = "replay_v1",
        request_id: str | None = None,
    ) -> ConsolidationOutput:
        started = perf_counter()
        consolidator = self._consolidators.get(strategy)
        if consolidator is None:
            consolidator = get_consolidation_strategy(strategy, self.config)
            self._consolidators[strategy] = consolidator
        episodes = consolidator.select_episodes(self.hippo, agent_id=agent_id, session_id=session_id)
        output = consolidator.run(agent_id=agent_id, episodes=episodes, embedder=self.embedder, semantic_store=self.semantic_store)
        duration_ms = (perf_counter() - started) * 1000
        self.telemetry.observe("hippocortex.consolidation.duration_ms", duration_ms, tags={"agent_id": agent_id, "strategy": strategy})
        logger.info(
            "consolidation_completed",
            extra={
                "agent_id": agent_id,
                "session_id": session_id,
                "request_id": request_id,
                "duration_ms": duration_ms,
            },
        )
        return output

    def build_context(
        self,
        agent_id: str,
        session_id: str,
        user_message: str,
        max_tokens: int,
        request_id: str | None = None,
    ) -> ContextPack:
        decision = self.router.route(user_message=user_message, max_tokens=max_tokens)
        logger.info(
            "router_decision",
            extra={
                "agent_id": agent_id,
                "session_id": session_id,
                "request_id": request_id,
                "intent": decision.intent,
                "explain": decision.explain,
            },
        )
        recent_events = self.hippo.list_events(agent_id=agent_id, session_id=session_id, limit=self.config.runtime.working_memory_turns)
        selected_recent = self.working_memory.select_recent(recent_events, token_budget=decision.working_memory_tokens)

        semantic_notes = []
        if decision.intent in {"semantic", "hybrid"}:
            semantic_notes = self.cortex.search(
                agent_id=agent_id,
                session_id=session_id,
                query=user_message,
                k=5,
                request_id=request_id,
            )

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
