from __future__ import annotations

import math

from hippocortex.embedders.base import Embedder


class DummyEmbedder(Embedder):
    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0 for _ in range(self._dimension)]
        for idx, byte in enumerate(text.encode("utf-8")):
            vec[idx % self._dimension] += (byte % 31) / 30.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
