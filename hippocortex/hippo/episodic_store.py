from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from hippocortex.types import EventRecord


class SQLiteEpisodicStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path(".") else None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")
            conn.commit()

    def add_event(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
        importance: float = 0.5,
    ) -> EventRecord:
        event = EventRecord(
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
            importance=importance,
        )
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (agent_id, session_id, role, content, metadata, timestamp, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.agent_id,
                    event.session_id,
                    event.role,
                    event.content,
                    json.dumps(event.metadata),
                    event.timestamp.isoformat(),
                    event.importance,
                ),
            )
            conn.commit()
            event.id = int(cur.lastrowid)
        return event

    def list_events(self, agent_id: str, session_id: str | None = None, limit: int = 50) -> list[EventRecord]:
        query = "SELECT * FROM events WHERE agent_id = ?"
        params: list[object] = [agent_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def top_events_by_importance(self, agent_id: str, session_id: str | None = None, limit: int = 20) -> list[EventRecord]:
        query = "SELECT * FROM events WHERE agent_id = ?"
        params: list[object] = [agent_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY importance DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            importance=float(row["importance"]),
        )
