from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hippocortex import HippoCortex


@dataclass
class AgentMemoryMiddleware:
    memory: HippoCortex
    agent_id: str
    session_id: str

    def before_llm(self, user_message: str, max_tokens: int = 1024) -> dict:
        context = self.memory.build_context(
            agent_id=self.agent_id,
            session_id=self.session_id,
            user_message=user_message,
            max_tokens=max_tokens,
        )
        return {
            "recent_turns": [event.model_dump() for event in context.recent_turns],
            "semantic_notes": [hit.note.text for hit in context.semantic_notes],
        }

    def after_llm(self, role: str, content: str, metadata: dict | None = None) -> None:
        self.memory.hippo.add_event(
            agent_id=self.agent_id,
            session_id=self.session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )
