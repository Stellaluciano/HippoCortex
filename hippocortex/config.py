from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    replay_episodes: int = 20
    working_memory_turns: int = 12


@dataclass(slots=True)
class StorageConfig:
    db_path: str = "hippocortex.db"
    semantic_store_backend: str = "memory"
    semantic_store_db_path: str | None = None


@dataclass(slots=True)
class ModelConfig:
    embedding_dim: int = 128
    embedder: str = "dummy"
    distill_strategy: str = "auto"


@dataclass(slots=True)
class RouterConfig:
    strategy: str = "memory_v1"


@dataclass(slots=True)
class HippoConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    @classmethod
    def from_env(cls) -> "HippoConfig":
        env = os.getenv("HIPPOCORTEX_ENV", "development").lower()
        default_backend = "sqlite" if env == "production" else "memory"
        return cls(
            runtime=RuntimeConfig(
                replay_episodes=int(os.getenv("HIPPOCORTEX_REPLAY_EPISODES", "20")),
                working_memory_turns=int(os.getenv("HIPPOCORTEX_WORKING_TURNS", "12")),
            ),
            storage=StorageConfig(
                db_path=os.getenv("HIPPOCORTEX_DB_PATH", "hippocortex.db"),
                semantic_store_backend=os.getenv("HIPPOCORTEX_SEMANTIC_STORE_BACKEND", default_backend),
                semantic_store_db_path=os.getenv("HIPPOCORTEX_SEMANTIC_STORE_DB_PATH"),
            ),
            model=ModelConfig(
                embedding_dim=int(os.getenv("HIPPOCORTEX_EMBEDDING_DIM", "128")),
                embedder=os.getenv("HIPPOCORTEX_EMBEDDER", "dummy"),
                distill_strategy=os.getenv("HIPPOCORTEX_DISTILL_STRATEGY", "auto"),
            ),
            router=RouterConfig(strategy=os.getenv("HIPPOCORTEX_ROUTER_STRATEGY", "memory_v1")),
        )

    def merged(self, override: "HippoConfig | None" = None) -> "HippoConfig":
        if override is None:
            return self
        return HippoConfig(
            runtime=RuntimeConfig(**{**asdict(self.runtime), **asdict(override.runtime)}),
            storage=StorageConfig(**{**asdict(self.storage), **asdict(override.storage)}),
            model=ModelConfig(**{**asdict(self.model), **asdict(override.model)}),
            router=RouterConfig(**{**asdict(self.router), **asdict(override.router)}),
        )

    def ensure_parent_dir(self) -> None:
        for raw_path in [self.storage.db_path, self.storage.semantic_store_db_path]:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.parent and str(path.parent) != ".":
                path.parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if self.model.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.runtime.working_memory_turns <= 0:
            raise ValueError("working_memory_turns must be positive")
        if self.runtime.replay_episodes <= 0:
            raise ValueError("replay_episodes must be positive")
        if self.storage.semantic_store_backend.lower() == "sqlite" and self.model.embedding_dim < 4:
            raise ValueError("sqlite backend requires embedding_dim >= 4")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Backward-compatible flat accessors
    @property
    def db_path(self) -> str:
        return self.storage.db_path

    @db_path.setter
    def db_path(self, value: str) -> None:
        self.storage.db_path = value

    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, value: int) -> None:
        self.model.embedding_dim = value

    @property
    def replay_episodes(self) -> int:
        return self.runtime.replay_episodes

    @replay_episodes.setter
    def replay_episodes(self, value: int) -> None:
        self.runtime.replay_episodes = value

    @property
    def working_memory_turns(self) -> int:
        return self.runtime.working_memory_turns

    @working_memory_turns.setter
    def working_memory_turns(self, value: int) -> None:
        self.runtime.working_memory_turns = value

    @property
    def semantic_store_backend(self) -> str:
        return self.storage.semantic_store_backend

    @semantic_store_backend.setter
    def semantic_store_backend(self, value: str) -> None:
        self.storage.semantic_store_backend = value

    @property
    def semantic_store_db_path(self) -> str | None:
        return self.storage.semantic_store_db_path

    @semantic_store_db_path.setter
    def semantic_store_db_path(self, value: str | None) -> None:
        self.storage.semantic_store_db_path = value
