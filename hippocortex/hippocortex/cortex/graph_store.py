from __future__ import annotations

from abc import ABC, abstractmethod


class GraphStore(ABC):
    @abstractmethod
    def add_edge(self, source: str, target: str, relation: str) -> None: ...

    @abstractmethod
    def neighbors(self, node: str) -> list[tuple[str, str]]: ...


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self._edges: dict[str, list[tuple[str, str]]] = {}

    def add_edge(self, source: str, target: str, relation: str) -> None:
        self._edges.setdefault(source, []).append((target, relation))

    def neighbors(self, node: str) -> list[tuple[str, str]]:
        return list(self._edges.get(node, []))
