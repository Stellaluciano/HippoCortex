from hippocortex import HippoCortex


def test_consolidation_creates_semantic_notes(tmp_path):
    memory = HippoCortex.default()
    memory.config.db_path = str(tmp_path / "hc.db")
    memory.__post_init__()

    memory.hippo.add_event("a1", "s1", "user", "I like tea and running.", {})
    memory.hippo.add_event("a1", "s1", "assistant", "Noted your preference for tea.", {})

    out = memory.consolidate(agent_id="a1", session_id="s1")
    assert out.notes_created >= 1

    hits = memory.cortex.search(agent_id="a1", query="What does user like?", k=5)
    assert len(hits) >= 1
