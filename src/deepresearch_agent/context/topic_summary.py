"""Incremental Run topic summaries with source coverage and idempotency."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from deepresearch_agent.config import settings
from deepresearch_agent.persistence.repositories import TopicSummaryRepository

from .compression import EvidenceContextPack, PlanningContextPack


SUMMARY_KEYS = (
    "schema_version", "topic", "keywords", "research_directions", "task_overview",
    "key_findings", "completed_work", "decisions", "constraints", "generated_artifacts",
    "unresolved_questions", "failed_attempts", "source_refs", "memory_candidates",
)


def _row(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if isinstance(value, dict):
        return dict(value)
    return {"content": str(value)} if value is not None else {}


def _unique(items: Iterable[Any], key_fields: tuple[str, ...] = ("content", "claim", "task_id", "source_id")) -> list[Any]:
    seen: set[str] = set()
    result = []
    for item in items:
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
        identity = next((str(item.get(key)) for key in key_fields if isinstance(item, dict) and item.get(key)), encoded)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


class RunTopicSummaryService:
    """Merge stage outputs into one logical row per (session_id, run_id)."""

    def __init__(self, repository: TopicSummaryRepository, *, schema_version: int | None = None) -> None:
        self.repository = repository
        self.schema_version = schema_version or settings.CONTEXT_SUMMARY_SCHEMA_VERSION

    async def get_for_run(self, session_id: str, run_id: str) -> dict[str, Any] | None:
        row = await self.repository.get_for_run(session_id, run_id)
        return None if row is None else self.repository.decode(row)

    async def update_pre_plan(
        self,
        *,
        session_id: str,
        run_id: str,
        user_query: str,
        pack: PlanningContextPack | dict[str, Any],
    ) -> tuple[Any, bool]:
        if isinstance(pack, dict):
            pack = PlanningContextPack(
                resolved_query=str(pack.get("resolved_query") or user_query),
                intent=str(pack.get("intent") or "new_research"),
                topic=str(pack.get("topic") or user_query[:180]),
                keywords=list(pack.get("keywords") or []),
                research_directions=list(pack.get("research_directions") or []),
                active_constraints=list(pack.get("active_constraints") or []),
                relevant_history=list(pack.get("relevant_history") or []),
                prior_run_trajectory=list(pack.get("prior_run_trajectory") or []),
                existing_artifacts=list(pack.get("existing_artifacts") or []),
                reusable_findings=list(pack.get("reusable_findings") or []),
                ambiguities=list(pack.get("ambiguities") or []),
                query_expansion_hints=list(pack.get("query_expansion_hints") or []),
                source_refs=list(pack.get("source_refs") or []),
                content=str(pack.get("content") or ""),
                budget=dict(pack.get("budget") or {}),
                compression_status=str(pack.get("compression_status") or "compressed"),
            )
        previous = await self.get_for_run(session_id, run_id)
        summary = self._base(previous, pack.topic)
        summary.update({
            "topic": pack.topic,
            "keywords": _unique(summary["keywords"] + pack.keywords),
            "research_directions": _unique(summary["research_directions"] + pack.research_directions),
            "task_overview": {"intent": pack.intent, "resolved_query": pack.resolved_query},
            "constraints": _unique(summary["constraints"] + pack.active_constraints),
            "unresolved_questions": _unique(summary["unresolved_questions"] + [item.get("text") for item in pack.ambiguities]),
            "source_refs": _unique(summary["source_refs"] + pack.source_refs, ("source_id", "result_id", "message_id", "content_hash")),
            "memory_candidates": self._memory_candidates(pack.active_constraints, session_id, run_id),
        })
        summary["metadata"] = self._stage_metadata(summary, "pre_plan", pack.source_refs)
        return await self._write(session_id, run_id, user_query, summary)

    async def update_post_execute(
        self,
        *,
        session_id: str,
        run_id: str,
        user_query: str,
        pack: EvidenceContextPack | dict[str, Any],
        execution_records: Iterable[Any] = (),
        artifacts: Iterable[Any] = (),
    ) -> tuple[Any, bool]:
        if isinstance(pack, dict):
            pack = EvidenceContextPack(
                research_objective=str(pack.get("research_objective") or user_query),
                task_coverage=list(pack.get("task_coverage") or []),
                evidence_blocks=list(pack.get("evidence_blocks") or []),
                deduplicated_evidence_groups=list(pack.get("deduplicated_evidence_groups") or []),
                key_findings=list(pack.get("key_findings") or []),
                contradictions=list(pack.get("contradictions") or []),
                source_quality_issues=list(pack.get("source_quality_issues") or []),
                failed_searches=list(pack.get("failed_searches") or []),
                evidence_gaps=list(pack.get("evidence_gaps") or []),
                unresolved_questions=list(pack.get("unresolved_questions") or []),
                suggested_reflection_checks=list(pack.get("suggested_reflection_checks") or []),
                source_refs=list(pack.get("source_refs") or []),
                content=str(pack.get("content") or ""),
                budget=dict(pack.get("budget") or {}),
                compression_status=str(pack.get("compression_status") or "compressed"),
            )
        previous = await self.get_for_run(session_id, run_id)
        summary = self._base(previous, pack.research_objective)
        records = [_row(item) for item in execution_records]
        summary.update({
            "topic": summary.get("topic") or pack.research_objective[:180],
            "task_overview": {"objective": pack.research_objective, "task_coverage": pack.task_coverage},
            "key_findings": _unique(summary["key_findings"] + pack.key_findings),
            "completed_work": _unique(summary["completed_work"] + [item for item in records if item.get("status") not in {"failed", "error"}]),
            "generated_artifacts": _unique(summary["generated_artifacts"] + [_row(item) for item in artifacts]),
            "unresolved_questions": _unique(summary["unresolved_questions"] + pack.unresolved_questions + pack.evidence_gaps),
            "failed_attempts": _unique(summary["failed_attempts"] + pack.failed_searches),
            "source_refs": _unique(summary["source_refs"] + pack.source_refs, ("source_id", "result_id", "evidence_id", "content_hash")),
        })
        summary["metadata"] = self._stage_metadata(summary, "post_execute_pre_reflection", pack.source_refs)
        return await self._write(session_id, run_id, user_query, summary)

    async def update_final(self, *, session_id: str, run_id: str, user_query: str, extra: dict[str, Any] | None = None) -> tuple[Any, bool]:
        previous = await self.get_for_run(session_id, run_id)
        summary = self._base(previous, user_query)
        if extra:
            for key in ("completed_work", "generated_artifacts", "unresolved_questions", "decisions"):
                if key in extra:
                    summary[key] = _unique(summary[key] + (extra[key] if isinstance(extra[key], list) else [extra[key]]))
        summary["metadata"] = self._stage_metadata(summary, "final", [])
        return await self._write(session_id, run_id, user_query, summary)

    async def _write(self, session_id: str, run_id: str, user_query: str, summary: dict[str, Any]):
        summary["metadata"]["input_fingerprint"] = self._fingerprint(summary)
        text = self._render(summary)
        refs = [item for item in summary.get("source_refs", []) if isinstance(item, dict)]
        return await self.repository.upsert(
            session_id=session_id, run_id=run_id, user_query=user_query,
            summary=summary, summary_text=text, source_refs=refs,
        )

    def _base(self, previous: dict[str, Any] | None, topic: str) -> dict[str, Any]:
        base = {key: [] for key in SUMMARY_KEYS if key not in {"schema_version", "topic", "task_overview"}}
        base.update({"schema_version": self.schema_version, "topic": topic, "task_overview": {}, "metadata": {}})
        if previous:
            for key in base:
                if key in previous:
                    base[key] = previous[key]
        return base

    @staticmethod
    def _stage_metadata(summary: dict[str, Any], phase: str, refs: list[dict[str, Any]]) -> dict[str, Any]:
        old = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
        history = list(old.get("phase_history") or [])
        entry = {"phase": phase, "source_count": len(refs), "covered_source_refs": refs}
        if not history or history[-1].get("phase") != phase or history[-1].get("source_count") != len(refs):
            history.append(entry)
        return {**old, "phase": phase, "phase_history": history, "covered_source_refs": summary.get("source_refs", [])}

    @staticmethod
    def _memory_candidates(constraints: list[dict[str, Any]], session_id: str, run_id: str) -> list[dict[str, Any]]:
        return [
            {
                "memory_type": "session_short_term", "content": item.get("content"),
                "scope_type": "run", "scope_id": run_id, "source_refs": item.get("source_refs", []),
                "status": "pending_validation", "confidence": 1.0,
            }
            for item in constraints if item.get("content")
        ]

    @staticmethod
    def _fingerprint(summary: dict[str, Any]) -> str:
        payload = dict(summary)
        # Phase bookkeeping is operational metadata and must not make the
        # same logical summary look like new research.
        payload.pop("metadata", None)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _render(summary: dict[str, Any]) -> str:
        lines = [f"# RUN TOPIC SUMMARY\nTopic: {summary.get('topic', '')}"]
        for key in ("keywords", "research_directions", "completed_work", "key_findings", "decisions", "constraints", "unresolved_questions", "failed_attempts"):
            value = summary.get(key) or []
            lines.append(f"{key}:\n" + "\n".join(f"- {json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) if isinstance(item, (dict, list)) else item}" for item in value) if isinstance(value, list) else f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)}")
        return "\n\n".join(lines)


__all__ = ["RunTopicSummaryService", "SUMMARY_KEYS"]
