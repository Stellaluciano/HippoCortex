from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodicEvent:
    role: str
    content: str


@dataclass
class SemanticNote:
    text: str
    source_indices: list[int]


@dataclass
class HippoCortexMemory:
    episodic_events: list[EpisodicEvent] = field(default_factory=list)
    semantic_notes: list[SemanticNote] = field(default_factory=list)

    def add_event(self, role: str, content: str) -> None:
        self.episodic_events.append(EpisodicEvent(role=role, content=content))

    def consolidate(self) -> list[SemanticNote]:
        if len(self.episodic_events) < 2:
            self.semantic_notes = []
            return self.semantic_notes

        first = self.episodic_events[0].content
        last = self.episodic_events[-1].content
        note1 = SemanticNote(
            text=f"User profile signal: {first}",
            source_indices=[0],
        )
        note2 = SemanticNote(
            text=f"Latest actionable context: {last}",
            source_indices=[len(self.episodic_events) - 1],
        )
        self.semantic_notes = [note1, note2]
        return self.semantic_notes

    def search(self, query: str, k: int = 3) -> list[SemanticNote]:
        tokens = set(query.lower().split())

        def score(note: SemanticNote) -> int:
            words = set(note.text.lower().split())
            return len(tokens & words)

        ranked = sorted(self.semantic_notes, key=score, reverse=True)
        return ranked[:k]
