from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for key in annotations:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif hasattr(self.__class__, key):
                    default = getattr(self.__class__, key)
                    setattr(self, key, default() if callable(default) else default)
                else:
                    setattr(self, key, None)

        def model_dump(self) -> dict[str, Any]:
            return {k: getattr(self, k) for k in getattr(self.__class__, "__annotations__", {})}

    def Field(default=None, default_factory=None):
        if default_factory is not None:
            return default_factory
        return default


class EventRecord(BaseModel):
    id: int | None = None
    agent_id: str
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: float = 0.5


class SemanticNote(BaseModel):
    id: str
    agent_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance_episode_ids: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    note: SemanticNote
    score: float


class ContextPack(BaseModel):
    recent_turns: list[EventRecord]
    semantic_notes: list[SearchResult]
    episodic_highlights: list[EventRecord] = Field(default_factory=list)
    token_budget: int


class RoutingDecision(BaseModel):
    intent: Literal["episodic", "semantic", "hybrid"]
    working_memory_tokens: int
    semantic_tokens: int
    include_highlights: bool = True


class ConsolidationOutput(BaseModel):
    strategy: str
    notes_created: int
    notes_skipped_dedup: int = 0
    episode_ids: list[int]
