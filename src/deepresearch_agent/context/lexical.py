"""Small language-neutral lexical matcher shared by memory and skill lookup."""

from __future__ import annotations

import re


def lexical_terms(value: str) -> set[str]:
    """Return bounded word and CJK bigram terms for deterministic matching."""
    lowered = value.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", lowered))
    words.update(cjk[index:index + 2] for index in range(max(0, len(cjk) - 1)))
    return {item for item in words if item}
