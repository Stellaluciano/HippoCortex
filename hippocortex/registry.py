from __future__ import annotations

from typing import Callable

from hippocortex.config import HippoConfig
from hippocortex.consolidation.distill import distill_episodes, heuristic_distill, llm_distill
from hippocortex.consolidation.replay import ReplayConsolidator
from hippocortex.cortex.semantic_store import InMemorySemanticStore, SQLiteSemanticStore
from hippocortex.router import MemoryRouter

RouterFactory = Callable[[], object]
DistillStrategy = Callable[[list], list[str]]
StorageFactory = Callable[[HippoConfig, int], object]
ConsolidationFactory = Callable[[HippoConfig], object]

_ROUTER_REGISTRY: dict[str, RouterFactory] = {}
_DISTILL_REGISTRY: dict[str, DistillStrategy] = {}
_STORAGE_REGISTRY: dict[str, StorageFactory] = {}
_CONSOLIDATION_REGISTRY: dict[str, ConsolidationFactory] = {}


def register_router_strategy(name: str, factory: RouterFactory) -> None:
    _ROUTER_REGISTRY[name] = factory


def get_router_strategy(name: str):
    try:
        return _ROUTER_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown router strategy: {name}") from exc


def register_distill_strategy(name: str, strategy: DistillStrategy) -> None:
    _DISTILL_REGISTRY[name] = strategy


def get_distill_strategy(name: str) -> DistillStrategy:
    try:
        return _DISTILL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown distill strategy: {name}") from exc


def register_storage_backend(name: str, factory: StorageFactory) -> None:
    _STORAGE_REGISTRY[name] = factory


def get_storage_backend(name: str, config: HippoConfig, dimension: int):
    try:
        return _STORAGE_REGISTRY[name](config, dimension)
    except KeyError as exc:
        raise ValueError(f"Unknown semantic store backend: {name}") from exc


def register_consolidation_strategy(name: str, factory: ConsolidationFactory) -> None:
    _CONSOLIDATION_REGISTRY[name] = factory


def get_consolidation_strategy(name: str, config: HippoConfig):
    try:
        return _CONSOLIDATION_REGISTRY[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown consolidation strategy: {name}") from exc


def _sqlite_factory(config: HippoConfig, dimension: int):
    semantic_db_path = config.storage.semantic_store_db_path or config.storage.db_path
    return SQLiteSemanticStore(db_path=semantic_db_path, dimension=dimension)


def _memory_factory(config: HippoConfig, dimension: int):
    return InMemorySemanticStore(dimension)


def _replay_factory(config: HippoConfig):
    return ReplayConsolidator(
        replay_size=config.runtime.replay_episodes,
        distill_strategy=config.model.distill_strategy,
        strategy_name="replay_v1",
    )


def register_defaults() -> None:
    if _ROUTER_REGISTRY:
        return
    register_router_strategy("memory_v1", MemoryRouter)

    register_distill_strategy("auto", distill_episodes)
    register_distill_strategy("heuristic", heuristic_distill)
    register_distill_strategy("llm", llm_distill)

    register_storage_backend("memory", _memory_factory)
    register_storage_backend("sqlite", _sqlite_factory)

    register_consolidation_strategy("replay_v1", _replay_factory)
