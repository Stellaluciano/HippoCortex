from __future__ import annotations

import os
from collections import Counter

from hippocortex.hippo.models import Episode


def heuristic_distill(episodes: list[Episode], max_points: int = 5) -> list[str]:
    if not episodes:
        return []
    text = " ".join(ep.content for ep in episodes)
    tokens = [tok.strip(".,!?;:\"'()[]{}") for tok in text.lower().split()]
    candidates = [tok for tok in tokens if len(tok) > 4]
    top = [word for word, _ in Counter(candidates).most_common(max_points)]
    latest = episodes[-min(len(episodes), 3) :]
    facts = [f"Recent interaction: {ep.role} said '{ep.content[:140]}'" for ep in latest]
    if top:
        facts.append(f"Recurring themes: {', '.join(top)}")
    return facts[:max_points]


def llm_distill(episodes: list[Episode]) -> list[str]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package not installed for LLM distillation.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    transcript = "\n".join([f"[{ep.role}] {ep.content}" for ep in episodes])
    client = OpenAI(api_key=api_key)
    prompt = (
        "Distill the following episodic transcript into up to 5 durable semantic notes. "
        "Return one fact per line.\n\n"
        f"{transcript}"
    )
    response = client.responses.create(model="gpt-4.1-mini", input=prompt)
    raw = response.output_text
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]


def distill_episodes(episodes: list[Episode]) -> list[str]:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return llm_distill(episodes)
        except Exception:
            return heuristic_distill(episodes)
    return heuristic_distill(episodes)
