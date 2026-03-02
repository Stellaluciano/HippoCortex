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

    @classmethod
    def from_env(cls) -> "HippoConfig":
        return cls(
            db_path=os.getenv("HIPPOCORTEX_DB_PATH", "hippocortex.db"),
            embedding_dim=int(os.getenv("HIPPOCORTEX_EMBEDDING_DIM", "128")),
            replay_episodes=int(os.getenv("HIPPOCORTEX_REPLAY_EPISODES", "20")),
            working_memory_turns=int(os.getenv("HIPPOCORTEX_WORKING_TURNS", "12")),
        )

    def ensure_parent_dir(self) -> None:
        path = Path(self.db_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
