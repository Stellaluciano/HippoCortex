from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from hippocortex.cortex.vector_index import SimpleVectorIndex, _dot, _normalize
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


class SQLiteSemanticStore(SemanticStore):
    def __init__(self, db_path: str, dimension: int) -> None:
        self.db_path = db_path
        self.dimension = dimension
        parent = Path(db_path).parent
        if parent != Path("."):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_notes (
                    note_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    provenance_episode_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_notes_agent ON semantic_notes(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_notes_created_at ON semantic_notes(created_at)")
            conn.commit()

    def add_note(self, note: SemanticNote) -> None:
        if len(note.embedding) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(note.embedding)}")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_notes (
                    note_id, agent_id, text, embedding, metadata, provenance_episode_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    agent_id = excluded.agent_id,
                    text = excluded.text,
                    embedding = excluded.embedding,
                    metadata = excluded.metadata,
                    provenance_episode_ids = excluded.provenance_episode_ids,
                    created_at = excluded.created_at
                """,
                (
                    note.id,
                    note.agent_id,
                    note.text,
                    json.dumps(_normalize(note.embedding)),
                    json.dumps(note.metadata),
                    json.dumps(note.provenance_episode_ids),
                    note.created_at.isoformat(),
                ),
            )
            conn.commit()

    def search(self, agent_id: str, query_vector: list[float], k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {len(query_vector)}")

        query = _normalize(query_vector)
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM semantic_notes WHERE agent_id = ?", (agent_id,)).fetchall()

        scored: list[SearchResult] = []
        for row in rows:
            note = self._row_to_note(row)
            if not self._matches_filters(note.metadata, filters):
                continue
            score = float(_dot(note.embedding, query))
            scored.append(SearchResult(note=note, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        return all(metadata.get(key) == value for key, value in filters.items())

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> SemanticNote:
        return SemanticNote(
            id=row["note_id"],
            agent_id=row["agent_id"],
            text=row["text"],
            embedding=[float(value) for value in json.loads(row["embedding"])],
            metadata=json.loads(row["metadata"]),
            provenance_episode_ids=[int(value) for value in json.loads(row["provenance_episode_ids"])],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
