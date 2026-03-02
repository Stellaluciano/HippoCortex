from hippocortex.hippo.episodic_store import SQLiteEpisodicStore


def test_add_and_list_events(tmp_path):
    db = tmp_path / "events.db"
    store = SQLiteEpisodicStore(str(db))

    e1 = store.add_event("agent", "sess", "user", "hello", {"x": 1})
    assert e1.id is not None

    events = store.list_events("agent", "sess", limit=10)
    assert len(events) == 1
    assert events[0].content == "hello"
