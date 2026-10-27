"""Add idempotent per-Run topic summaries used by context compression."""

from alembic import op
import sqlalchemy as sa


revision = "20261027_0006"
down_revision = "20261008_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "session_topic_summaries" in inspector.get_table_names():
        return
    op.create_table(
        "session_topic_summaries",
        sa.Column("summary_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("summary_json", sa.Text(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("source_refs_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("summary_char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.String(length=40), nullable=False),
        sa.Column("updated_at", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
        sa.PrimaryKeyConstraint("summary_id"),
        sa.UniqueConstraint("session_id", "run_id", name="uq_session_topic_summary_run"),
    )
    op.create_index(
        "ix_topic_summaries_session_created", "session_topic_summaries", ["session_id", "created_at"]
    )
    op.create_index("ix_topic_summaries_run", "session_topic_summaries", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_topic_summaries_run", table_name="session_topic_summaries")
    op.drop_index("ix_topic_summaries_session_created", table_name="session_topic_summaries")
    op.drop_table("session_topic_summaries")
