import pytest

from hippocortex import HippoConfig, HippoCortex
from hippocortex.config import ModelConfig, RuntimeConfig, StorageConfig
from hippocortex.embedders.dummy_embedder import DummyEmbedder


def test_default_uses_env_and_code_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HIPPOCORTEX_REPLAY_EPISODES", "11")
    monkeypatch.setenv("HIPPOCORTEX_WORKING_TURNS", "5")
    monkeypatch.setenv("HIPPOCORTEX_DB_PATH", str(tmp_path / "from_env.db"))

    override = HippoConfig(
        runtime=RuntimeConfig(replay_episodes=21, working_memory_turns=7),
        storage=StorageConfig(db_path=str(tmp_path / "override.db"), semantic_store_backend="memory"),
        model=ModelConfig(embedding_dim=16, distill_strategy="heuristic"),
    )

    memory = HippoCortex.default(config=override)

    assert memory.config.runtime.replay_episodes == 21
    assert memory.config.runtime.working_memory_turns == 7
    assert memory.config.storage.db_path.endswith("override.db")


def test_invalid_backend_and_embedding_dim_combo_raises(tmp_path):
    cfg = HippoConfig(
        storage=StorageConfig(db_path=str(tmp_path / "hc.db"), semantic_store_backend="sqlite"),
        model=ModelConfig(embedding_dim=2),
    )
    with pytest.raises(ValueError, match="sqlite backend requires embedding_dim >= 4"):
        HippoCortex.default(config=cfg, embedder=DummyEmbedder(dimension=2))


def test_unknown_consolidation_strategy_uses_registry_error():
    memory = HippoCortex.default()
    with pytest.raises(ValueError, match="Unknown consolidation strategy"):
        memory.consolidate(agent_id="a1", strategy="does_not_exist")
