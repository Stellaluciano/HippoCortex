from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class VectorRow:
    id: str
    vector: list[float]
    payload: dict


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SimpleVectorIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._rows: dict[str, VectorRow] = {}

    def upsert(self, item_id: str, vector: list[float], payload: dict) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}")
        self._rows[item_id] = VectorRow(id=item_id, vector=_normalize(vector), payload=payload)

    def search(self, query_vector: list[float], k: int = 5, filters: dict | None = None) -> list[tuple[str, float, dict]]:
        if not self._rows:
            return []
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {len(query_vector)}")
        query = _normalize(query_vector)

        def pass_filters(payload: dict) -> bool:
            if not filters:
                return True
            return all(payload.get(key) == val for key, val in filters.items())

        scored: list[tuple[str, float, dict]] = []
        for row in self._rows.values():
            if not pass_filters(row.payload):
                continue
            score = float(_dot(row.vector, query))
            scored.append((row.id, score, row.payload))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
