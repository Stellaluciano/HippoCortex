from datetime import datetime

from hippocortex import HippoConfig, HippoCortex
from hippocortex.cortex.semantic_store import InMemorySemanticStore, SQLiteSemanticStore
from hippocortex.embedders.dummy_embedder import DummyEmbedder
from hippocortex.types import SemanticNote

def test_sqlite_semantic_store_matches_inmemory_behavior(tmp_path):
    db_path = tmp_path / "semantic.db"
    dimension = 4
    in_memory = InMemorySemanticStore(dimension=dimension)
    sqlite_store = SQLiteSemanticStore(db_path=str(db_path), dimension=dimension)

    notes = [
        SemanticNote(
            id="n1",
            agent_id="agent-1",
            text="alpha",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"topic": "ai"},
            provenance_episode_ids=[1],
            created_at=datetime.utcnow(),
        ),
        SemanticNote(
            id="n2",
            agent_id="agent-1",
            text="beta",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={"topic": "ml"},
            provenance_episode_ids=[2],
            created_at=datetime.utcnow(),
        ),
        SemanticNote(
            id="n3",
            agent_id="agent-2",
            text="gamma",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"topic": "ai"},
            provenance_episode_ids=[3],
            created_at=datetime.utcnow(),
        ),
    ]

    for note in notes:
        in_memory.add_note(note)
        sqlite_store.add_note(note)

    query_vector = [1.0, 0.0, 0.0, 0.0]
    memory_hits = in_memory.search(agent_id="agent-1", query_vector=query_vector, k=5, filters={"topic": "ai"})
    sqlite_hits = sqlite_store.search(agent_id="agent-1", query_vector=query_vector, k=5, filters={"topic": "ai"})

    assert [hit.note.id for hit in sqlite_hits] == [hit.note.id for hit in memory_hits]
    assert [round(hit.score, 6) for hit in sqlite_hits] == [round(hit.score, 6) for hit in memory_hits]

def test_semantic_store_persists_across_restart(tmp_path):
    db_path = tmp_path / "hippocortex.db"
    config = HippoConfig(
        db_path=str(db_path),
        embedding_dim=8,
        semantic_store_backend="sqlite",
    )
    embedder = DummyEmbedder(dimension=8)

    first = HippoCortex.default(config=config, embedder=embedder)
    note = SemanticNote(
        id="persistent-note",
        agent_id="agent-1",
        text="persistent memory",
        embedding=embedder.embed_text("persistent memory"),
        metadata={"kind": "fact"},
        provenance_episode_ids=[101],
        created_at=datetime.utcnow(),
    )
    first.semantic_store.add_note(note)

    second = HippoCortex.default(config=config, embedder=embedder)
    hits = second.semantic_store.search(
        agent_id="agent-1",
        query_vector=embedder.embed_text("persistent memory"),
        k=3,
        filters={"kind": "fact"},
    )

    assert len(hits) == 1
    assert hits[0].note.id == "persistent-note"
    assert hits[0].note.provenance_episode_ids == [101]

def test_semantic_store_add_note_ignore_conflict(tmp_path):
    db_path = tmp_path / "semantic_conflict.db"
    store = SQLiteSemanticStore(db_path=str(db_path), dimension=4)
    note = SemanticNote(
        id="n1",
        agent_id="agent-1",
        text="alpha",
        embedding=[1.0, 0.0, 0.0, 0.0],
        metadata={"topic": "ai", "digest": "d1"},
        provenance_episode_ids=[1],
        created_at=datetime.utcnow(),
    )

    assert store.add_note(note, on_conflict="ignore") is True
    assert store.add_note(note, on_conflict="ignore") is False
    assert store.has_equivalent_note(
        agent_id="agent-1",
        text="alpha",
        provenance_episode_ids=[1],
        digest="d1",
    )
