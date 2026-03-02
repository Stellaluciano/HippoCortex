"""Optional Postgres adapter placeholder for future integration."""

from __future__ import annotations


class PostgresAdapter:
    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("Postgres adapter is optional and not implemented in MVP.")
