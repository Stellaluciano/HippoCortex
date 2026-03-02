from __future__ import annotations

from hippocortex.consolidation.distill import distill_episodes
from hippocortex.types import ConsolidationOutput, SemanticNote
from hippocortex.utils.hashing import stable_id


class ReplayConsolidator:
    def __init__(self, replay_size: int = 20) -> None:
        self.replay_size = replay_size

    def select_episodes(self, store, agent_id: str, session_id: str | None = None) -> list:
        by_importance = store.top_events_by_importance(agent_id=agent_id, session_id=session_id, limit=self.replay_size)
        if by_importance:
            return by_importance
        return store.list_events(agent_id=agent_id, session_id=session_id, limit=self.replay_size)

    def run(self, agent_id: str, episodes: list, embedder, semantic_store) -> ConsolidationOutput:
        facts = distill_episodes(episodes)
        for i, fact in enumerate(facts):
            note_id = stable_id(f"{agent_id}:{fact}:{i}", prefix="note_")
            note = SemanticNote(
                id=note_id,
                agent_id=agent_id,
                text=fact,
                embedding=embedder.embed_text(fact),
                metadata={"source": "consolidation", "strategy": "replay_v1"},
                provenance_episode_ids=[e.id for e in episodes if e.id is not None],
            )
            semantic_store.add_note(note)
        return ConsolidationOutput(
            strategy="replay_v1",
            notes_created=len(facts),
            episode_ids=[ep.id for ep in episodes if ep.id is not None],
        )
