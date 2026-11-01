from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.enums import EmailStatus

# JSONB on Postgres (indexable, compact), plain JSON elsewhere (SQLite in unit tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Order(Base):
    """A customer order from the (fictional) retailer. Tools will read these in step 2."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. ML-104233
    customer_name: Mapped[str] = mapped_column(String(120))
    customer_email: Mapped[str] = mapped_column(String(255), index=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    carrier: Mapped[str] = mapped_column(String(40))
    tracking_number: Mapped[str] = mapped_column(String(40), unique=True)
    status: Mapped[str] = mapped_column(String(20))  # in_transit, delivered, returned
    shipped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Email(Base):
    """An incoming email and the latest validated triage result."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    sender: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(20), default=EmailStatus.RECEIVED, index=True)
    triage: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    flags: Mapped[list[str]] = mapped_column(JSONType, default=list)

    runs: Mapped[list["TriageRun"]] = relationship(
        back_populates="email", order_by="TriageRun.id", cascade="all, delete-orphan"
    )


class TriageRun(Base):
    """One triage attempt on one email: what was asked, what came back, what it cost."""

    __tablename__ = "triage_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(80))
    prompt_version: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))
    attempts: Mapped[int]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    latency_ms: Mapped[int]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    flags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    raw_output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    email: Mapped[Email] = relationship(back_populates="runs")
