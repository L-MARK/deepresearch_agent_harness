"""Bounded, provenance-aware query refinement."""
from __future__ import annotations
import re
from typing import Any
from pydantic import BaseModel, Field

class QueryRefinementContext(BaseModel):
    original_query: str
    resolved_query: str
    current_session_messages: list[dict[str, Any]] = Field(default_factory=list)
    session_summary: dict[str, Any] | None = None
    historical_recall: list[dict[str, Any]] = Field(default_factory=list)
    clarification_history: list[dict[str, Any]] = Field(default_factory=list)
    domain_context: str | None = None
    user_preferences: dict[str, Any] = Field(default_factory=dict)

class QueryRefinementResult(BaseModel):
    refined_query: str
    confidence: float = 0.0
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    applied_constraints: list[dict[str, Any]] = Field(default_factory=list)
    used_message_ids: list[str] = Field(default_factory=list)
    used_session_ids: list[str] = Field(default_factory=list)
    used_clarification_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

QUERY_REFINEMENT_PROMPT = """你是研究问题解析与重写器。仅融合已提供且有来源标记的信息，生成自包含、无关键指代、可交给 Research Planner 的 refined_query。不要回答问题、不要拆分任务、不要把历史对话当作研究证据；当前用户输入和澄清答案优先于当前会话、长期偏好和跨会话历史。若关键歧义无法确认，不要猜测。"""

class QueryRefiner:
    _historical = re.compile(r"(之前|上次|上回|历史|继续.*(研究|项目)|那个项目)")
    def refine(self, *, original_query: str, resolved_query: str, current_session_context=None,
               session_summary=None, historical_recall=None, clarification_history=None, domain_context=None,
               user_preferences=None) -> QueryRefinementResult:
        messages = current_session_context or []
        recalls = historical_recall or []
        clarifications = [x for x in (clarification_history or []) if str(x.get("answer") or "").strip()]
        used_messages = [str(x.get("message_id")) for x in messages if x.get("message_id")]
        used_clarifications = [str(x.get("clarification_id")) for x in clarifications if x.get("clarification_id")]
        constraints = [{"type": x.get("ambiguity_type", "clarification"), "value": str(x["answer"]).strip(), "source": "clarification"} for x in clarifications]
        base = (resolved_query.strip() or original_query.strip())
        ambiguities, used_sessions = [], []
        if recalls and self._historical.search(original_query):
            if len(recalls) == 1:
                candidate = recalls[0]; used_sessions = [str(candidate.get("session_id"))]
                title = str(candidate.get("title") or "").strip()
                if title and title not in base: base = f"围绕 {title}：{base}"
            else: ambiguities.append("跨 Session 召回了多个可能的研究对象，无法确认用户所指对象")
        if clarifications: base += "\n\n已确认约束：" + "；".join(str(x["answer"]).strip() for x in clarifications)
        if not original_query.strip(): ambiguities.append("缺少当前用户问题")
        return QueryRefinementResult(refined_query=base, confidence=0.35 if ambiguities else (0.95 if clarifications else 0.9), unresolved_ambiguities=ambiguities, applied_constraints=constraints, used_message_ids=used_messages, used_session_ids=used_sessions, used_clarification_ids=used_clarifications)
