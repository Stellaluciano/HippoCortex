from __future__ import annotations

import os

from hippocortex.embedders.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package not installed. Install with `pip install hippocortex[openai]`.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_text("dimension probe")
            self._dimension = len(probe)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        res = self._client.embeddings.create(model=self._model, input=text)
        vector = res.data[0].embedding
        if self._dimension is None:
            self._dimension = len(vector)
        return vector
