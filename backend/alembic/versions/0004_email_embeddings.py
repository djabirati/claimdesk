"""email embeddings for similar past cases (pgvector)

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The pgvector extension ships with the pgvector/pgvector image used by docker compose
    # and CI; it only has to be enabled once per database.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "email_embeddings",
        sa.Column("email_id", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=80), nullable=False),
        sa.Column("embedding", Vector(dim=768), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["email_id"], ["emails.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("email_id"),
    )
    op.create_index(
        "ix_email_embeddings_embedding",
        "email_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_email_embeddings_embedding", table_name="email_embeddings")
    op.drop_table("email_embeddings")
    # The extension is left in place: dropping it would fail if anything else used it.
