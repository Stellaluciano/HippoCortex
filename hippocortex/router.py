from __future__ import annotations

from hippocortex.types import RoutingDecision


class MemoryRouter:
    def route(self, user_message: str, max_tokens: int) -> RoutingDecision:
        lowered = user_message.lower()
        semantic_keywords = ["remember", "history", "why", "pattern", "summarize", "context"]
        episodic_keywords = ["just now", "latest", "last message", "session"]

        semantic_score = sum(1 for k in semantic_keywords if k in lowered)
        episodic_score = sum(1 for k in episodic_keywords if k in lowered)

        if semantic_score > episodic_score:
            intent = "semantic"
        elif episodic_score > semantic_score:
            intent = "episodic"
        else:
            intent = "hybrid"

        wm_tokens = int(max_tokens * (0.6 if intent == "episodic" else 0.4))
        semantic_tokens = max_tokens - wm_tokens
        return RoutingDecision(
            intent=intent,
            working_memory_tokens=wm_tokens,
            semantic_tokens=semantic_tokens,
            include_highlights=intent != "semantic",
        )
