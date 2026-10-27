"""Stage-specific context packs for planning and reflection.

The module deliberately separates deterministic provenance handling from the
optional LLM semantic summarizer. Database identifiers and compression state
are always assigned by Python.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .lexical import lexical_terms
from .prompts import (
    EVIDENCE_CONTEXT_PACK_SCHEMA,
    PLANNER_COMPRESSION_SYSTEM_PROMPT,
    PLANNING_CONTEXT_PACK_SCHEMA,
    REFLECTION_COMPRESSION_SYSTEM_PROMPT,
)
from .tokens import ContextWindowGuard, count_tokens


COMPRESSION_STATUSES = {"uncompressed", "partially_compressed", "compressed", "excluded", "failed"}


@dataclass
class ContextSegment:
    segment_id: str
    source_type: str
    content: str
    session_id: str | None = None
    run_id: str | None = None
    source_id: str | None = None
    message_id: str | None = None
    task_id: str | None = None
    record_id: str | None = None
    tool_call_id: str | None = None
    result_id: str | None = None
    artifact_id: str | None = None
    summary_id: str | None = None
    created_at: str | None = None
    run_sequence: int | None = None
    event_sequence: int | None = None
    phase: str | None = None
    compression_status: str = "uncompressed"
    summary_phase: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.compression_status not in COMPRESSION_STATUSES:
            self.compression_status = "failed"
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def source_ref(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "segment_id": self.segment_id,
                "source_type": self.source_type,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "source_id": self.source_id,
                "message_id": self.message_id,
                "task_id": self.task_id,
                "record_id": self.record_id,
                "tool_call_id": self.tool_call_id,
                "result_id": self.result_id,
                "artifact_id": self.artifact_id,
                "summary_id": self.summary_id,
                "created_at": self.created_at,
                "run_sequence": self.run_sequence,
                "event_sequence": self.event_sequence,
                "phase": self.phase,
                "content_hash": self.content_hash,
                "compression_status": self.compression_status,
                "summary_phase": self.summary_phase,
            }.items()
            if value is not None
        }


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _terms(value: str) -> set[str]:
    return lexical_terms(value or "")


def _short(value: Any, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", _text(value)).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _response_text(response: Any) -> str:
    value = getattr(response, "content", response)
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value or "")


def _parse_json_object(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _invoke_semantic_compressor(
    llm: Any,
    *,
    system_prompt: str,
    schema: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Call an injected model without allowing it to assign provenance/state."""
    if llm is None:
        return None
    prompt = (
        f"{system_prompt}\n\nOUTPUT SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
        f"\n\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )
    try:
        return _parse_json_object(_response_text(llm.invoke(prompt)))
    except Exception:
        # Compression is an optimization. Deterministic selection and source
        # protection remain valid when the optional semantic call fails.
        return None


@dataclass
class PlanningContextPack:
    resolved_query: str
    intent: str
    topic: str
    keywords: list[str]
    research_directions: list[str]
    active_constraints: list[dict[str, Any]]
    relevant_history: list[dict[str, Any]]
    prior_run_trajectory: list[dict[str, Any]]
    existing_artifacts: list[dict[str, Any]]
    reusable_findings: list[dict[str, Any]]
    ambiguities: list[dict[str, Any]]
    query_expansion_hints: list[str]
    source_refs: list[dict[str, Any]]
    content: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    compression_status: str = "compressed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "resolved_query": self.resolved_query,
            "intent": self.intent,
            "topic": self.topic,
            "keywords": self.keywords,
            "research_directions": self.research_directions,
            "active_constraints": self.active_constraints,
            "relevant_history": self.relevant_history,
            "prior_run_trajectory": self.prior_run_trajectory,
            "existing_artifacts": self.existing_artifacts,
            "reusable_findings": self.reusable_findings,
            "ambiguities": self.ambiguities,
            "query_expansion_hints": self.query_expansion_hints,
            "source_refs": self.source_refs,
            "content": self.content,
            "budget": self.budget,
            "compression_status": self.compression_status,
        }


@dataclass
class EvidenceContextPack:
    research_objective: str
    task_coverage: list[dict[str, Any]]
    evidence_blocks: list[dict[str, Any]]
    deduplicated_evidence_groups: list[dict[str, Any]]
    key_findings: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    source_quality_issues: list[dict[str, Any]]
    failed_searches: list[dict[str, Any]]
    evidence_gaps: list[str]
    unresolved_questions: list[str]
    suggested_reflection_checks: list[str]
    source_refs: list[dict[str, Any]]
    content: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    compression_status: str = "compressed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "research_objective": self.research_objective,
            "task_coverage": self.task_coverage,
            "evidence_blocks": self.evidence_blocks,
            "deduplicated_evidence_groups": self.deduplicated_evidence_groups,
            "key_findings": self.key_findings,
            "contradictions": self.contradictions,
            "source_quality_issues": self.source_quality_issues,
            "failed_searches": self.failed_searches,
            "evidence_gaps": self.evidence_gaps,
            "unresolved_questions": self.unresolved_questions,
            "suggested_reflection_checks": self.suggested_reflection_checks,
            "source_refs": self.source_refs,
            "content": self.content,
            "budget": self.budget,
            "compression_status": self.compression_status,
        }


class PlanningContextCompressor:
    """Build a planning-specific pack from recent, recalled and durable context."""

    def __init__(self, *, guard: ContextWindowGuard | None = None, llm: Any = None, enabled: bool = True) -> None:
        self.guard = guard or ContextWindowGuard()
        self.llm = llm
        self.enabled = enabled

    def compress(
        self,
        *,
        original_query: str,
        resolved_query: str | None = None,
        current_session_messages: Iterable[Any] = (),
        historical_recall: Iterable[Any] = (),
        topic_summaries: Iterable[Any] = (),
        short_term_memory: Iterable[Any] = (),
        long_term_memory: Iterable[Any] = (),
        artifacts: Iterable[Any] = (),
        trajectory: Iterable[Any] = (),
        constraints: Iterable[Any] = (),
        source_segments: Iterable[ContextSegment] = (),
    ) -> PlanningContextPack:
        query = (resolved_query or original_query or "").strip()
        query_terms = _terms(query)
        refs: list[dict[str, Any]] = [{
            "source_type": "current_user_query", "source_id": "current_query",
            "content_hash": hashlib.sha256(original_query.encode("utf-8")).hexdigest(),
        }]

        history_rows: list[dict[str, Any]] = []
        for item in list(current_session_messages) + list(historical_recall):
            row = self._row(item)
            content = _text(row.get("content") or row.get("snippet") or row.get("summary"))
            if not content:
                continue
            score = len(query_terms & _terms(content))
            if row.get("role") == "user":
                score += 1
            row["content"] = _short(content, 900)
            row["relevance"] = score
            if score > 0 or row.get("source_type") in {"current_query", "task_constraint"}:
                history_rows.append(row)
                if row.get("source_ref"):
                    refs.append(row["source_ref"])
        history_rows.sort(key=lambda row: (int(row.get("relevance") or 0), _text(row.get("created_at"))), reverse=True)

        summaries = [self._row(item) for item in topic_summaries]
        summaries = [row for row in summaries if self._related(row, query_terms)]
        for row in summaries:
            if row.get("run_id"):
                summary_text = _text(row.get("summary_text") or row.get("content") or row.get("topic"))
                refs.append({
                    "source_type": "run_topic_summary",
                    "source_id": f"summary:{row['run_id']}",
                    "summary_id": row.get("summary_id"),
                    "run_id": row.get("run_id"),
                    "session_id": row.get("session_id"),
                    "content_hash": hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
                })
            if row.get("source_ref"):
                refs.append(row["source_ref"])
            refs.extend(item for item in (row.get("source_refs") or []) if isinstance(item, dict))

        reusable = []
        for item in summaries:
            reusable.append({
                "run_id": item.get("run_id"),
                "topic": item.get("topic") or item.get("summary_text") or _short(item.get("content"), 700),
                "completed_work": item.get("completed_work", []),
                "key_findings": item.get("key_findings", []),
                "unresolved_questions": item.get("unresolved_questions", []),
            })

        active_constraints = []
        for item in list(constraints) + list(short_term_memory) + list(long_term_memory):
            row = self._row(item)
            content = _short(row.get("content") or row.get("constraint") or row.get("text"), 700)
            if content:
                active_constraints.append({
                    "content": content,
                    "scope": row.get("scope") or row.get("scope_type") or "session",
                    "status": row.get("status") or "active",
                    "source_refs": row.get("source_refs") or row.get("source_ref") or [],
                })
                if row.get("source_ref"):
                    refs.append(row["source_ref"])

        # Detect explicit temporary requirements before semantic compression.
        # This is a candidate signal only; ContextBuilder validates/persists it.
        for constraint in self._explicit_constraints(query):
            active_constraints.append({
                "content": constraint,
                "scope": "session", "status": "active",
                "source_refs": [{"source_type": "current_user_query", "content_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()}],
            })

        artifacts_rows = [self._row(item) for item in artifacts if self._row(item)]
        trajectory_rows = [self._row(item) for item in trajectory if self._row(item)]
        keywords = sorted(query_terms, key=lambda value: (-len(value), value))[:16]
        topic = _short(query, 180)
        directions = [
            _short(row.get("topic") or row.get("research_directions") or row.get("description"), 240)
            for row in summaries
            if row.get("topic") or row.get("research_directions") or row.get("description")
        ][:8]
        ambiguities = self._ambiguities(query, active_constraints)
        hints = list(dict.fromkeys([
            f"复用与当前查询相关的历史 Run：{row.get('run_id')}" for row in reusable if row.get("run_id")
        ] + ["区分已验证事实、用户提供信息、推断和待验证信息", "核对当前有效临时要求后再拆解任务"]))

        pack = PlanningContextPack(
            resolved_query=query,
            intent="continuation_research" if reusable or history_rows else "new_research",
            topic=topic,
            keywords=keywords,
            research_directions=directions,
            active_constraints=active_constraints,
            relevant_history=history_rows[:12],
            prior_run_trajectory=trajectory_rows[:10],
            existing_artifacts=artifacts_rows[:8],
            reusable_findings=reusable[:8],
            ambiguities=ambiguities,
            query_expansion_hints=hints[:8],
            source_refs=self._unique_refs(refs),
        )
        pack.content = self._render(pack)
        budget = self.guard.estimate([pack.content, query])
        semantic_used = False
        if bool(budget["triggered"]) and self.llm is not None:
            semantic_input = self.guard.fit_text(
                pack.content,
                target_tokens=max(256, int(self.guard.input_budget * 0.60)),
            )
            semantic = _invoke_semantic_compressor(
                self.llm,
                system_prompt=PLANNER_COMPRESSION_SYSTEM_PROMPT,
                schema=PLANNING_CONTEXT_PACK_SCHEMA,
                payload={
                    "query": query,
                    "candidate_context": semantic_input,
                    "source_refs": pack.source_refs,
                },
            )
            if semantic:
                if isinstance(semantic.get("topic"), str) and semantic["topic"].strip():
                    pack.topic = _short(semantic["topic"], 180)
                if isinstance(semantic.get("keywords"), list):
                    pack.keywords = [str(item) for item in semantic["keywords"] if str(item).strip()][:16]
                if isinstance(semantic.get("research_directions"), list):
                    pack.research_directions = [str(item) for item in semantic["research_directions"] if str(item).strip()][:8]
                if isinstance(semantic.get("ambiguities"), list):
                    pack.ambiguities = [item for item in semantic["ambiguities"] if isinstance(item, dict)][:8]
                if isinstance(semantic.get("query_expansion_hints"), list):
                    pack.query_expansion_hints = [str(item) for item in semantic["query_expansion_hints"] if str(item).strip()][:8]
                # Query, constraints, history and source refs stay under
                # deterministic control; the model cannot silently replace them.
                pack.content = self._render(pack)
                semantic_used = True
                budget = self.guard.estimate([pack.content, query])
        rounds = 0
        while bool(budget["triggered"]) and int(budget["full_request_input_tokens"]) > int(budget["target_tokens"]) and rounds < self.guard.max_rounds:
            pack.content = self.guard.fit_text(pack.content, target_tokens=int(budget["target_tokens"]))
            budget = self.guard.estimate([pack.content, query])
            rounds += 1
        budget["compression_rounds"] = rounds
        budget["max_compression_rounds"] = self.guard.max_rounds
        budget["semantic_llm_used"] = semantic_used
        pack.budget = budget
        return pack

    @staticmethod
    def _row(item: Any) -> dict[str, Any]:
        if isinstance(item, ContextSegment):
            row = item.source_ref.copy()
            row.update({"content": item.content, "source_ref": item.source_ref})
            return row
        if hasattr(item, "model_dump"):
            return dict(item.model_dump())
        if isinstance(item, dict):
            return dict(item)
        return {"content": _text(item)} if item is not None else {}

    @staticmethod
    def _related(row: dict[str, Any], query_terms: set[str]) -> bool:
        text = _text(row.get("topic") or row.get("summary_text") or row.get("content") or row.get("summary"))
        return bool(query_terms & _terms(text)) or not query_terms

    @staticmethod
    def _ambiguities(query: str, constraints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        markers = ("这个", "该", "它", "上述", "前者", "后者", "继续", "再", "this", "it", "that")
        if any(marker in query.lower() for marker in markers):
            return [{"type": "historical_reference", "text": "当前查询包含可能依赖历史上下文的指代或省略", "candidates": []}]
        return []

    @staticmethod
    def _explicit_constraints(query: str) -> list[str]:
        patterns = (
            r"(?:这次|本轮|当前(?:阶段|版本)?)[：:，,]?[^。；;。]{0,12}(只[^。；;]{2,80})",
            r"((?:不要|暂时不要|优先|仅使用|限定在|限制为)[^。；;]{2,90})",
        )
        found: list[str] = []
        for pattern in patterns:
            found.extend(match.group(1).strip() for match in re.finditer(pattern, query, flags=re.IGNORECASE))
        return list(dict.fromkeys(found))[:5]

    @staticmethod
    def _unique_refs(refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        output = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            key = (str(ref.get("source_id") or ref.get("segment_id") or ref.get("message_id") or ""), str(ref.get("content_hash") or ""))
            if key in seen or not any(key):
                continue
            seen.add(key)
            output.append(ref)
        return output

    @staticmethod
    def _render(pack: PlanningContextPack) -> str:
        return "\n\n".join([
            "[PLANNING_CONTEXT_PACK]",
            f"resolved_query: {pack.resolved_query}",
            f"intent: {pack.intent}; topic: {pack.topic}",
            "active_constraints:\n" + "\n".join(f"- {item['content']} ({item['scope']})" for item in pack.active_constraints) if pack.active_constraints else "active_constraints: none",
            "relevant_history:\n" + "\n".join(f"- {item.get('content', '')}" for item in pack.relevant_history[:8]) if pack.relevant_history else "relevant_history: none",
            "reusable_findings:\n" + "\n".join(f"- Run {item.get('run_id')}: {_short(item.get('topic'), 240)}" for item in pack.reusable_findings[:6]) if pack.reusable_findings else "reusable_findings: none",
            "ambiguities:\n" + "\n".join(f"- {item.get('text')}" for item in pack.ambiguities) if pack.ambiguities else "ambiguities: none detected",
            "query_expansion_hints:\n" + "\n".join(f"- {item}" for item in pack.query_expansion_hints),
            "source_refs: " + json.dumps(pack.source_refs, ensure_ascii=False),
        ])


class EvidenceContextCompressor:
    """Deduplicate and compress evidence while retaining audit identifiers."""

    def __init__(self, *, guard: ContextWindowGuard | None = None, llm: Any = None, enabled: bool = True) -> None:
        self.guard = guard or ContextWindowGuard()
        self.llm = llm
        self.enabled = enabled

    def compress(
        self,
        *,
        research_objective: str,
        plan: Any = None,
        task_records: Iterable[Any] = (),
        retrieval_results: Iterable[Any] = (),
        evidence_ledger: Iterable[Any] = (),
        current_constraints: Iterable[Any] = (),
        existing_summary: Any = None,
    ) -> EvidenceContextPack:
        objective = _text(research_objective)
        blocks: list[dict[str, Any]] = []
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        refs: list[dict[str, Any]] = []
        for item in list(retrieval_results) + list(evidence_ledger):
            row = self._row(item)
            content = _text(row.get("evidence") or row.get("content") or row.get("summary") or row.get("text"))
            if not content:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source_id = _text(row.get("source_id") or metadata.get("source_id") or row.get("url") or row.get("source") or "unknown")
            content_hash = _text(row.get("content_hash") or metadata.get("content_hash")) or hashlib.sha256(content.encode("utf-8")).hexdigest()
            result_id = _text(row.get("result_id") or row.get("evidence_id")) or None
            key = (source_id, content_hash)
            ref = {
                "source_type": "evidence_record",
                "source_id": source_id,
                "result_id": result_id,
                "content_hash": content_hash,
                "title": row.get("title") or metadata.get("title"),
                "url": row.get("url") or metadata.get("url"),
                "published_at": row.get("published_at") or metadata.get("published_at"),
            }
            refs.append(ref)
            if key in groups:
                groups[key]["duplicate_result_ids"].append(result_id)
                continue
            block = {
                "evidence_id": result_id,
                "source_id": source_id,
                "content_hash": content_hash,
                "title": ref["title"],
                "url": ref["url"],
                "published_at": ref["published_at"],
                "retrieved_at": row.get("retrieved_at") or metadata.get("retrieved_at"),
                "source_quality": metadata.get("source_quality") or metadata.get("domain") or row.get("provider"),
                "content": _short(content, 850),
                "duplicate_result_ids": [],
                "status": "compressed" if len(content) > 850 else "uncompressed",
            }
            groups[key] = block
            blocks.append(block)

        blocks.sort(key=lambda block: (0 if block.get("url") else 1, str(block.get("source_id"))))
        findings = [
            {
                "claim": block["content"],
                "classification": "unverified_fact",
                "source_refs": [self._ref_for(block)],
            }
            for block in blocks[:12]
        ]
        task_coverage = self._task_coverage(plan, task_records, blocks)
        failed = [
            self._row(record)
            for record in task_records
            if any(token in _text(self._row(record).get("error") or self._row(record).get("status")).lower() for token in ("fail", "error", "timeout", "失败"))
        ]
        contradictions = self._contradictions(blocks)
        gaps = ["研究目标尚无可追溯证据"] if not blocks else []
        if not task_coverage and plan is not None:
            gaps.append("计划任务与当前证据之间缺少覆盖映射")
        checks = ["逐条核对关键数字、时间范围、适用法域和统计口径", "检查不同来源是否存在矛盾或不可合并的口径"]
        if failed:
            checks.append("检查失败检索路径是否造成研究覆盖缺口")
        pack = EvidenceContextPack(
            research_objective=objective,
            task_coverage=task_coverage,
            evidence_blocks=blocks,
            deduplicated_evidence_groups=list(groups.values()),
            key_findings=findings,
            contradictions=contradictions,
            source_quality_issues=[block for block in blocks if not block.get("url") or not block.get("source_quality")],
            failed_searches=[self._failure_row(row) for row in failed],
            evidence_gaps=gaps,
            unresolved_questions=["当前证据是否足以支撑所有验收标准？"] if blocks else ["需要补充可验证的原始来源"],
            suggested_reflection_checks=checks,
            source_refs=self._unique_refs(refs),
        )
        pack.content = self._render(pack)
        budget = self.guard.estimate([pack.content, objective])
        semantic_used = False
        if bool(budget["triggered"]) and self.llm is not None:
            semantic_input = self.guard.fit_text(
                pack.content,
                target_tokens=max(256, int(self.guard.input_budget * 0.60)),
            )
            semantic = _invoke_semantic_compressor(
                self.llm,
                system_prompt=REFLECTION_COMPRESSION_SYSTEM_PROMPT,
                schema=EVIDENCE_CONTEXT_PACK_SCHEMA,
                payload={
                    "research_objective": objective,
                    "candidate_evidence_context": semantic_input,
                    "source_refs": pack.source_refs,
                },
            )
            if semantic:
                if isinstance(semantic.get("suggested_reflection_checks"), list):
                    pack.suggested_reflection_checks = list(dict.fromkeys(
                        pack.suggested_reflection_checks
                        + [str(item) for item in semantic["suggested_reflection_checks"] if str(item).strip()]
                    ))[:8]
                if isinstance(semantic.get("evidence_gaps"), list):
                    pack.evidence_gaps = list(dict.fromkeys(
                        pack.evidence_gaps
                        + [str(item) for item in semantic["evidence_gaps"] if str(item).strip()]
                    ))[:8]
                if isinstance(semantic.get("unresolved_questions"), list):
                    pack.unresolved_questions = list(dict.fromkeys(
                        pack.unresolved_questions
                        + [str(item) for item in semantic["unresolved_questions"] if str(item).strip()]
                    ))[:8]
                # Only accept model-produced findings that cite a source/hash
                # already present in the deterministic evidence ledger.
                allowed_refs = {
                    (str(ref.get("source_id") or ""), str(ref.get("content_hash") or ""))
                    for ref in pack.source_refs
                }
                semantic_findings = []
                for finding in semantic.get("key_findings", []) if isinstance(semantic.get("key_findings"), list) else []:
                    if not isinstance(finding, dict) or not finding.get("claim"):
                        continue
                    finding_refs = finding.get("source_refs") or []
                    if finding_refs and all(
                        (str(ref.get("source_id") or ""), str(ref.get("content_hash") or "")) in allowed_refs
                        for ref in finding_refs if isinstance(ref, dict)
                    ):
                        semantic_findings.append(finding)
                if semantic_findings:
                    pack.key_findings = semantic_findings[:12]
                pack.content = self._render(pack)
                semantic_used = True
                budget = self.guard.estimate([pack.content, objective])
        rounds = 0
        while bool(budget["triggered"]) and int(budget["full_request_input_tokens"]) > int(budget["target_tokens"]) and rounds < self.guard.max_rounds:
            pack.content = self.guard.fit_text(pack.content, target_tokens=int(budget["target_tokens"]))
            budget = self.guard.estimate([pack.content, objective])
            rounds += 1
        budget["compression_rounds"] = rounds
        budget["max_compression_rounds"] = self.guard.max_rounds
        budget["semantic_llm_used"] = semantic_used
        pack.budget = budget
        return pack

    @staticmethod
    def _row(item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return dict(item.model_dump())
        if isinstance(item, dict):
            return dict(item)
        return {"content": _text(item)} if item is not None else {}

    @staticmethod
    def _ref_for(block: dict[str, Any]) -> dict[str, Any]:
        return {key: block.get(key) for key in ("source_id", "evidence_id", "content_hash", "url") if block.get(key)}

    @staticmethod
    def _failure_row(row: dict[str, Any]) -> dict[str, Any]:
        return {key: row.get(key) for key in ("task_id", "record_id", "error", "status", "reason") if row.get(key) is not None}

    @staticmethod
    def _task_coverage(plan: Any, records: Iterable[Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        task_ids: list[str] = []
        if plan is not None:
            raw = plan.model_dump() if hasattr(plan, "model_dump") else plan
            nodes = raw.get("task_graph", {}).get("nodes", []) if isinstance(raw, dict) else []
            task_ids.extend(str(node.get("task_id")) for node in nodes if isinstance(node, dict) and node.get("task_id"))
        task_ids.extend(str(PlanningContextCompressor._row(item).get("task_id")) for item in records if PlanningContextCompressor._row(item).get("task_id"))
        return [{"task_id": task_id, "evidence_count": len(blocks), "covered": bool(blocks)} for task_id in list(dict.fromkeys(task_ids))]

    @staticmethod
    def _contradictions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Deterministic signal only; it intentionally does not declare a true
        # contradiction without a semantic model.
        by_keyword: dict[str, list[str]] = {}
        for block in blocks:
            for word in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", block.get("content", ""))[:80]:
                by_keyword.setdefault(word.lower(), []).append(str(block.get("source_id")))
        return [
            {"term": term, "source_ids": ids, "status": "needs_review"}
            for term, ids in by_keyword.items()
            if len(set(ids)) > 1 and len(ids) >= 3
        ][:8]

    @staticmethod
    def _unique_refs(refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        output = []
        for ref in refs:
            key = (str(ref.get("source_id") or ref.get("result_id") or ""), str(ref.get("content_hash") or ""))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            output.append(ref)
        return output

    @staticmethod
    def _render(pack: EvidenceContextPack) -> str:
        blocks = "\n".join(
            f"- [{block.get('evidence_id') or 'evidence'}] {block.get('content')} "
            f"(source_id={block.get('source_id')}, url={block.get('url')}, published_at={block.get('published_at')}, hash={block.get('content_hash')})"
            for block in pack.evidence_blocks[:16]
        ) or "- none"
        return "\n\n".join([
            "[EVIDENCE_CONTEXT_PACK]",
            f"research_objective: {pack.research_objective}",
            "evidence_blocks:\n" + blocks,
            "contradictions:\n" + ("\n".join(f"- {item}" for item in pack.contradictions) or "- none detected; semantic review still required"),
            "failed_searches:\n" + ("\n".join(f"- {item}" for item in pack.failed_searches) or "- none recorded"),
            "evidence_gaps:\n" + ("\n".join(f"- {item}" for item in pack.evidence_gaps) or "- none identified"),
            "suggested_reflection_checks:\n" + "\n".join(f"- {item}" for item in pack.suggested_reflection_checks),
            "source_refs: " + json.dumps(pack.source_refs, ensure_ascii=False),
        ])


__all__ = [
    "COMPRESSION_STATUSES", "ContextSegment", "PlanningContextPack", "EvidenceContextPack",
    "PlanningContextCompressor", "EvidenceContextCompressor",
]
