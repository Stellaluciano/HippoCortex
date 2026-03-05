from __future__ import annotations

from hippocortex.types import EventRecord


class WorkingMemory:
    def __init__(self, max_recent_turns: int = 12) -> None:
        self.max_recent_turns = max_recent_turns

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def select_recent(self, events: list[EventRecord], token_budget: int) -> list[EventRecord]:
        """Select recent events from newest->oldest input under a token budget.

        The token budget is consumed in recency order (latest first). The return
        value is reversed to chronological order for easier downstream display.
        """
        selected: list[EventRecord] = []
        running = 0
        for event in events:
            cost = self.estimate_tokens(event.content)
            if running + cost > token_budget:
                break
            selected.append(event)
            running += cost
        return list(reversed(selected))
