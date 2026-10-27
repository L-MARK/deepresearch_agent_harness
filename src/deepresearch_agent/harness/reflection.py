"""Extension point for the independent post-Orchestrator reflection stage."""

from __future__ import annotations

from typing import Any, Protocol


class OverallReflectionHook(Protocol):
    async def reflect(self, *, context: Any, evidence_context_pack: dict[str, Any]) -> dict[str, Any] | None:
        """Inspect the complete Run after Execute and before Report."""


class NoopOverallReflection:
    """Default hook: preparation is active, semantic reflection is deferred."""

    async def reflect(self, *, context: Any, evidence_context_pack: dict[str, Any]) -> dict[str, Any] | None:
        return None


__all__ = ["OverallReflectionHook", "NoopOverallReflection"]
