"""add graph edges table

Revision ID: 1f2e3d4c5b6a
Revises: c23286132406
Create Date: 2026-03-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1f2e3d4c5b6a"
down_revision: Union[str, None] = "c23286132406"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_engine_id", sa.String(length=36), nullable=False),
        sa.Column("target_engine_id", sa.String(length=36), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("signal_breakdown", sa.JSON(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_graph_edges_id", "graph_edges", ["id"], unique=False)
    op.create_index("ix_graph_edges_source_engine_id", "graph_edges", ["source_engine_id"], unique=False)
    op.create_index("ix_graph_edges_target_engine_id", "graph_edges", ["target_engine_id"], unique=False)
    op.create_index(
        "idx_graph_edges_pair",
        "graph_edges",
        ["source_engine_id", "target_engine_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_graph_edges_pair", table_name="graph_edges")
    op.drop_index("ix_graph_edges_target_engine_id", table_name="graph_edges")
    op.drop_index("ix_graph_edges_source_engine_id", table_name="graph_edges")
    op.drop_index("ix_graph_edges_id", table_name="graph_edges")
    op.drop_table("graph_edges")
