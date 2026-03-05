from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class HippoConfig:
    db_path: str = "hippocortex.db"
    embedding_dim: int = 128
    replay_episodes: int = 20
    working_memory_turns: int = 12
    semantic_store_backend: str = "memory"
    semantic_store_db_path: str | None = None

    @classmethod
    def from_env(cls) -> "HippoConfig":
        env = os.getenv("HIPPOCORTEX_ENV", "development").lower()
        default_backend = "sqlite" if env == "production" else "memory"
        return cls(
            db_path=os.getenv("HIPPOCORTEX_DB_PATH", "hippocortex.db"),
            embedding_dim=int(os.getenv("HIPPOCORTEX_EMBEDDING_DIM", "128")),
            replay_episodes=int(os.getenv("HIPPOCORTEX_REPLAY_EPISODES", "20")),
            working_memory_turns=int(os.getenv("HIPPOCORTEX_WORKING_TURNS", "12")),
            semantic_store_backend=os.getenv("HIPPOCORTEX_SEMANTIC_STORE_BACKEND", default_backend),
            semantic_store_db_path=os.getenv("HIPPOCORTEX_SEMANTIC_STORE_DB_PATH"),
        )

    def ensure_parent_dir(self) -> None:
        for raw_path in [self.db_path, self.semantic_store_db_path]:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.parent and str(path.parent) != ".":
                path.parent.mkdir(parents=True, exist_ok=True)
