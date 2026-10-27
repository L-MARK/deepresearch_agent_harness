"""Persistence helpers for compressed context and per-Run topic summaries."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import select, update

from deepresearch_agent.persistence.database import Database
from deepresearch_agent.persistence.models import SessionTopicSummaryModel

from .utils import json_text, new_id, utc_now_iso


class TopicSummaryRepository:
    """CRUD with a database-enforced one-summary-per-session/run invariant."""

    def __init__(self, database: Database):
        self.database = database

    async def get_for_run(self, session_id: str, run_id: str) -> Optional[SessionTopicSummaryModel]:
        async with self.database.sessions() as session:
            statement = select(SessionTopicSummaryModel).where(
                SessionTopicSummaryModel.session_id == session_id,
                SessionTopicSummaryModel.run_id == run_id,
            )
            return (await session.execute(statement)).scalar_one_or_none()

    async def list_for_session(self, session_id: str, *, limit: int = 50) -> list[SessionTopicSummaryModel]:
        async with self.database.sessions() as session:
            statement = (
                select(SessionTopicSummaryModel)
                .where(SessionTopicSummaryModel.session_id == session_id)
                .order_by(SessionTopicSummaryModel.created_at.desc())
                .limit(max(1, min(limit, 200)))
            )
            return list((await session.execute(statement)).scalars())

    async def upsert(
        self,
        *,
        session_id: str,
        run_id: str,
        user_query: str,
        summary: dict[str, Any],
        summary_text: str,
        source_refs: list[dict[str, Any]],
    ) -> tuple[SessionTopicSummaryModel, bool]:
        """Insert or update atomically. Returns (row, changed)."""
        now = utc_now_iso()
        encoded_summary = json_text(summary)
        encoded_refs = json_text(source_refs)
        async with self.database.transaction() as session:
            statement = select(SessionTopicSummaryModel).where(
                SessionTopicSummaryModel.session_id == session_id,
                SessionTopicSummaryModel.run_id == run_id,
            )
            row = (await session.execute(statement)).scalar_one_or_none()
            if row is None:
                row = SessionTopicSummaryModel(
                    summary_id=new_id("summary"), session_id=session_id, run_id=run_id,
                    user_query=user_query, summary_json=encoded_summary,
                    summary_text=summary_text, source_refs_json=encoded_refs,
                    summary_char_count=len(summary_text), created_at=now, updated_at=now,
                )
                session.add(row)
                return row, True

            try:
                same_summary = json.loads(row.summary_json or "{}") == json.loads(encoded_summary)
                same_refs = json.loads(row.source_refs_json or "[]") == json.loads(encoded_refs)
            except json.JSONDecodeError:
                same_summary = row.summary_json == encoded_summary
                same_refs = row.source_refs_json == encoded_refs
            if row.user_query == user_query and same_summary and row.summary_text == summary_text and same_refs:
                return row, False
            await session.execute(
                update(SessionTopicSummaryModel)
                .where(SessionTopicSummaryModel.summary_id == row.summary_id)
                .values(
                    user_query=user_query,
                    summary_json=encoded_summary,
                    summary_text=summary_text,
                    source_refs_json=encoded_refs,
                    summary_char_count=len(summary_text),
                    updated_at=now,
                )
            )
            row.user_query = user_query
            row.summary_json = encoded_summary
            row.summary_text = summary_text
            row.source_refs_json = encoded_refs
            row.summary_char_count = len(summary_text)
            row.updated_at = now
            return row, True

    @staticmethod
    def decode(row: SessionTopicSummaryModel) -> dict[str, Any]:
        try:
            summary = json.loads(row.summary_json or "{}")
        except json.JSONDecodeError:
            summary = {}
        try:
            refs = json.loads(row.source_refs_json or "[]")
        except json.JSONDecodeError:
            refs = []
        summary.setdefault("summary_id", row.summary_id)
        summary.setdefault("session_id", row.session_id)
        summary.setdefault("run_id", row.run_id)
        summary.setdefault("summary_text", row.summary_text)
        summary["source_refs"] = refs
        return summary


__all__ = ["TopicSummaryRepository"]
