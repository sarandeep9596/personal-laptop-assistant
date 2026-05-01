"""Split a natural-language query into single-intent clauses."""
from __future__ import annotations
import re
from typing import List

_SPLIT_RE = re.compile(r"\s*(?:,|\band\b|\bthen\b)\s*", flags=re.IGNORECASE)


def split(query: str) -> List[str]:
    if not query or not query.strip():
        return []
    parts = _SPLIT_RE.split(query.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
