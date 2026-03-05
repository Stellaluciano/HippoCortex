from __future__ import annotations

from hippocortex.types import ConsolidationOutput, SemanticNote
from hippocortex.utils.hashing import stable_id


class ReplayConsolidator:
    def __init__(self, replay_size: int = 20, distill_strategy: str = "auto", strategy_name: str = "replay_v1") -> None:
        self.replay_size = replay_size
        self.distill_strategy = distill_strategy
        self.strategy_name = strategy_name

    def select_episodes(self, store, agent_id: str, session_id: str | None = None) -> list:
        by_importance = store.top_events_by_importance(agent_id=agent_id, session_id=session_id, limit=self.replay_size)
        if by_importance:
            return by_importance
        return store.list_events(agent_id=agent_id, session_id=session_id, limit=self.replay_size)

    def run(self, agent_id: str, episodes: list, embedder, semantic_store) -> ConsolidationOutput:
        from hippocortex.registry import get_distill_strategy

        strategy = self.strategy_name
        facts = get_distill_strategy(self.distill_strategy)(episodes)
        episode_ids = [ep.id for ep in episodes if ep.id is not None]
        episode_ids_sorted = sorted(episode_ids)
        run_basis = f"{agent_id}:{episode_ids_sorted}:{strategy}"
        run_id = stable_id(run_basis, prefix="con_run_")

        notes_created = 0
        notes_skipped_dedup = 0

        for i, fact in enumerate(facts):
            digest = stable_id(f"{run_basis}:{fact}", prefix="digest_")
            if semantic_store.has_equivalent_note(
                agent_id=agent_id,
                text=fact,
                provenance_episode_ids=episode_ids_sorted,
                digest=digest,
            ):
                notes_skipped_dedup += 1
                continue

            note_id = stable_id(f"{agent_id}:{fact}:{i}", prefix="note_")
            note = SemanticNote(
                id=note_id,
                agent_id=agent_id,
                text=fact,
                embedding=embedder.embed_text(fact),
                metadata={
                    "source": "consolidation",
                    "strategy": strategy,
                    "run_id": run_id,
                    "digest": digest,
                },
                provenance_episode_ids=episode_ids_sorted,
            )
            created = semantic_store.add_note(note, on_conflict="ignore")
            if created:
                notes_created += 1
            else:
                notes_skipped_dedup += 1

        return ConsolidationOutput(
            strategy=strategy,
            notes_created=notes_created,
            notes_skipped_dedup=notes_skipped_dedup,
            episode_ids=episode_ids,
        )
