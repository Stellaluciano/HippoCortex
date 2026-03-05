from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Telemetry(Protocol):
    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None: ...

    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None: ...


@dataclass
class NoOpTelemetry:
    """Default telemetry backend that safely drops all metrics."""

    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        return None

    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None:
        return None


@dataclass
class InMemoryTelemetry:
    """Testing helper backend that records every metric call."""

    counters: list[tuple[str, int, dict[str, str]]] = field(default_factory=list)
    observations: list[tuple[str, float, dict[str, str]]] = field(default_factory=list)

    def increment(self, metric: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        self.counters.append((metric, value, tags or {}))

    def observe(self, metric: str, value: float, tags: dict[str, str] | None = None) -> None:
        self.observations.append((metric, value, tags or {}))

