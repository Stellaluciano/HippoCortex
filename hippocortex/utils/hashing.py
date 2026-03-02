from __future__ import annotations

import hashlib


def stable_id(text: str, prefix: str = "") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"
