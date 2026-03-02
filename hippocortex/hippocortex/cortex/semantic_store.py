from __future__ import annotations

from abc import ABC, abstractmethod

from hippocortex.cortex.vector_index import SimpleVectorIndex
from hippocortex.types import SearchResult, SemanticNote


class SemanticStore(ABC):
    @abstractmethod
    def add_note(self, note: SemanticNote) -> None: ...

    @abstractmethod
    def search(self, agent_id: str, query_vector: list[float], k: int = 5, filters: dict | None = None) -> list[SearchResult]: ...


class InMemorySemanticStore(SemanticStore):
    def __init__(self, dimension: int) -> None:
        self._index = SimpleVectorIndex(dimension=dimension)
        self._notes: dict[str, SemanticNote] = {}

    def add_note(self, note: SemanticNote) -> None:
        self._notes[note.id] = note
        payload = {"agent_id": note.agent_id, **note.metadata}
        self._index.upsert(note.id, note.embedding, payload=payload)

    def search(self, agent_id: str, query_vector: list[float], k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        merged_filters = {"agent_id": agent_id, **(filters or {})}
        hits = self._index.search(query_vector=query_vector, k=k, filters=merged_filters)
        return [SearchResult(note=self._notes[item_id], score=score) for item_id, score, _ in hits]
