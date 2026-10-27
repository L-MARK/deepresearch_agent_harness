"""允许一个 Run 持久化多条澄清控制消息。

Revision ID: 20261008_0005
Revises: 20260921_0004
"""

from alembic import op


revision = "20261008_0005"
down_revision = "20260921_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    # 旧索引会阻止同一 Run 同时保存 clarification_request 和最终报告。
    bind.exec_driver_sql("DROP INDEX IF EXISTS uq_messages_assistant_run")
    bind.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_messages_run_role "
        "ON messages (run_id, role) WHERE run_id IS NOT NULL"
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP INDEX IF EXISTS ix_messages_run_role")
    bind.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_assistant_run "
        "ON messages (run_id) WHERE role = 'assistant' AND run_id IS NOT NULL"
    )
