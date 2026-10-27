import pytest
import pytest_asyncio

from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from deepresearch_agent.context import (
    ContextWindowGuard,
    EvidenceContextCompressor,
    PlanningContextCompressor,
    RunTopicSummaryService,
)
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.memory import MemoryService
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    AuditRepository,
    MemoryRepository,
    MessageRepository,
    RunRepository,
    SessionRepository,
    TopicSummaryRepository,
)


@pytest_asyncio.fixture
async def database(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{(tmp_path / 'compression.db').as_posix()}")
    await database.create_schema()
    yield database
    await database.close()


def test_context_window_guard_rechecks_full_request_budget():
    guard = ContextWindowGuard(
        context_window=100, reserved_output_tokens=10, safety_margin=5,
        trigger_ratio=0.80, target_ratio=0.70,
    )
    low = guard.estimate(["short system", "short query"])
    high = guard.estimate(["important " * 100])
    assert low["triggered"] is False
    assert high["triggered"] is True
    assert high["target_tokens"] < high["full_request_input_tokens"]


class _FakeCompressionLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": self.response})()


def test_optional_semantic_compressor_uses_stage_specific_prompt_and_keeps_refs():
    llm = _FakeCompressionLLM(
        '{"topic":"semantic topic","keywords":["policy"],'
        '"research_directions":["market impact"],"ambiguities":[],'
        '"query_expansion_hints":["compare official sources"]}'
    )
    guard = ContextWindowGuard(
        context_window=200, provider_input_limit=200, reserved_output_tokens=0,
        safety_margin=0, trigger_ratio=0.80, target_ratio=0.70,
    )
    pack = PlanningContextCompressor(guard=guard, llm=llm).compress(
        original_query="policy " * 200,
        current_session_messages=[{"message_id": "m1", "role": "user", "content": "policy history"}],
    )
    assert llm.prompts
    assert "PlanningContextPack" in llm.prompts[0]
    assert pack.topic == "semantic topic"
    assert any(ref.get("source_type") == "current_user_query" for ref in pack.source_refs)


def test_planning_pack_preserves_constraints_and_filters_unrelated_history():
    pack = PlanningContextCompressor().compress(
        original_query="这次只分析中国市场的新能源汽车政策，并继续上次报告",
        current_session_messages=[
            {"message_id": "m1", "role": "user", "content": "上次报告分析了新能源汽车政策"},
            {"message_id": "m2", "role": "assistant", "content": "今天天气不错"},
        ],
        topic_summaries=[{"run_id": "run-1", "topic": "中国新能源汽车政策", "summary_text": "已完成政策时间线"}],
    )
    assert pack.active_constraints
    assert any("中国市场" in item["content"] for item in pack.active_constraints)
    assert any(item.get("message_id") == "m1" for item in pack.relevant_history)
    assert all(item.get("message_id") != "m2" for item in pack.relevant_history)
    assert "[PLANNING_CONTEXT_PACK]" in pack.content


def test_evidence_pack_deduplicates_exact_source_but_keeps_distinct_sources():
    pack = EvidenceContextCompressor().compress(
        research_objective="比较政策影响",
        retrieval_results=[
            {"result_id": "r1", "source_id": "official-a", "content_hash": "h1", "evidence": "关键数字 10%", "url": "https://a"},
            {"result_id": "r2", "source_id": "official-a", "content_hash": "h1", "evidence": "关键数字 10%", "url": "https://a"},
            {"result_id": "r3", "source_id": "official-b", "content_hash": "h1", "evidence": "关键数字 10%", "url": "https://b"},
        ],
    )
    assert len(pack.evidence_blocks) == 2
    assert "r2" in pack.evidence_blocks[0]["duplicate_result_ids"] or "r2" in pack.evidence_blocks[1]["duplicate_result_ids"]
    assert {item["source_id"] for item in pack.evidence_blocks} == {"official-a", "official-b"}
    assert all(item.get("content_hash") for item in pack.source_refs)


@pytest.mark.asyncio
async def test_topic_summary_is_idempotent_and_session_memory_is_scoped(database):
    sessions = SessionRepository(database)
    messages = MessageRepository(database)
    runs = RunRepository(database)
    session = await sessions.create(SessionCreate(title="summary"))
    message, run, _ = await runs.create_for_user_message(
        MessageCreate(session_id=session.session_id, role="user", content="这次只分析中国市场", client_message_id="summary-1"),
        RunCreate(session_id=session.session_id, trigger_message_id="assigned", source_mode=SourceMode.WEB, workflow_mode=WorkflowMode.DEEP_RESEARCH),
    )
    summary_service = RunTopicSummaryService(TopicSummaryRepository(database))
    pack = PlanningContextCompressor().compress(original_query=message.content)
    _, first_changed = await summary_service.update_pre_plan(
        session_id=session.session_id, run_id=run.run_id, user_query=message.content, pack=pack,
    )
    _, second_changed = await summary_service.update_pre_plan(
        session_id=session.session_id, run_id=run.run_id, user_query=message.content, pack=pack,
    )
    assert first_changed is True
    assert second_changed is False

    memory = MemoryService(MemoryRepository(database), AuditRepository(database))
    await memory.create_candidate(
        target="session", scope="session", session_id=session.session_id,
        content="本轮只使用官方数据", kind="constraint", provenance_refs=[message.message_id], activate=True,
    )
    scoped = await memory.build_session_snapshot(session.session_id)
    assert "本轮只使用官方数据" in scoped["rendered"]
    other = await sessions.create(SessionCreate(title="other"))
    other_scoped = await memory.build_session_snapshot(other.session_id)
    assert other_scoped["rendered"] == ""
