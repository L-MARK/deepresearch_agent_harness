"""Versioned prompts used when an optional semantic compression LLM is enabled."""

PLANNER_COMPRESSION_SYSTEM_PROMPT = """You build a PlanningContextPack for a research agent.
Resolve historical references only when supported by supplied sources. Separate facts,
user-provided requirements, inferences and unverified claims. Preserve every active
temporary constraint and cite only source identifiers present in the input. Do not
create a task graph, invent IDs, or decide database state. Return the requested JSON schema."""

REFLECTION_COMPRESSION_SYSTEM_PROMPT = """You build an EvidenceContextPack for reflection.
Deduplicate exact repeated evidence but keep materially different sources separate.
Preserve source IDs, result IDs, hashes, dates, numbers, units, jurisdictions and
statistical scope. Mark conflicts and gaps instead of merging incompatible claims.
Do not treat a prior summary as independent evidence, invent identifiers, or declare
the research successful. Return the requested JSON schema."""

PLANNING_CONTEXT_PACK_SCHEMA = {
    "resolved_query": "string", "intent": "string", "topic": "string", "keywords": ["string"],
    "research_directions": ["string"], "active_constraints": ["object"], "relevant_history": ["object"],
    "prior_run_trajectory": ["object"], "existing_artifacts": ["object"], "reusable_findings": ["object"],
    "ambiguities": ["object"], "query_expansion_hints": ["string"], "source_refs": ["object"],
}

EVIDENCE_CONTEXT_PACK_SCHEMA = {
    "research_objective": "string", "task_coverage": ["object"], "evidence_blocks": ["object"],
    "deduplicated_evidence_groups": ["object"], "key_findings": ["object"], "contradictions": ["object"],
    "source_quality_issues": ["object"], "failed_searches": ["object"], "evidence_gaps": ["string"],
    "unresolved_questions": ["string"], "suggested_reflection_checks": ["string"], "source_refs": ["object"],
}

__all__ = [
    "PLANNER_COMPRESSION_SYSTEM_PROMPT", "REFLECTION_COMPRESSION_SYSTEM_PROMPT",
    "PLANNING_CONTEXT_PACK_SCHEMA", "EVIDENCE_CONTEXT_PACK_SCHEMA",
]
