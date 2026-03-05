from datetime import datetime

from hippocortex import DummyEmbedder, HippoConfig, HippoCortex
from hippocortex.config import ModelConfig, RuntimeConfig, StorageConfig
from hippocortex.types import EventRecord
from hippocortex.working_memory import WorkingMemory


def _event(content: str, event_id: int) -> EventRecord:
    return EventRecord(
        id=event_id,
        agent_id="agent",
        session_id="session",
        role="user",
        content=content,
        timestamp=datetime.utcnow(),
    )


def test_select_recent_prioritizes_latest_events_with_budget():
    wm = WorkingMemory()
    # Input order convention: newest -> oldest.
    newest = _event("n" * 4, 4)  # 1 token
    newer = _event("m" * 12, 3)  # 3 tokens
    older = _event("o" * 4, 2)  # 1 token
    oldest = _event("p" * 4, 1)  # 1 token

    selected = wm.select_recent([newest, newer, older, oldest], token_budget=4)

    # Selected events are rendered in chronological order for display.
    assert [event.id for event in selected] == [3, 4]


def test_build_context_uses_latest_events_not_earliest(tmp_path):
    db_path = tmp_path / "memory.db"
    sdk = HippoCortex.default(config=HippoConfig(storage=StorageConfig(db_path=str(db_path)), runtime=RuntimeConfig(working_memory_turns=4), model=ModelConfig(embedding_dim=8)), embedder=DummyEmbedder(dimension=8))

    sdk.hippo.add_event("agent", "session", "user", "oldest-aaaa")  # 2 tokens
    sdk.hippo.add_event("agent", "session", "assistant", "older-bbbb")  # 2 tokens
    sdk.hippo.add_event("agent", "session", "user", "newer-cccc")  # 2 tokens
    sdk.hippo.add_event("agent", "session", "assistant", "latest-dddd")  # 2 tokens

    pack = sdk.build_context(agent_id="agent", session_id="session", user_message="latest", max_tokens=8)

    # Router allocates 60% of max_tokens to working memory: floor(8 * 0.6) == 4.
    # So exactly two latest 2-token events should be kept.
    assert [event.content for event in pack.recent_turns] == ["newer-cccc", "latest-dddd"]
