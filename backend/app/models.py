from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.enums import ActionStatus, EmailStatus, JobStatus

# JSONB on Postgres (indexable, compact), plain JSON elsewhere (SQLite in unit tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Order(Base):
    """A customer order from the (fictional) retailer. The agent's tools read these."""

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
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="email", order_by="Job.id", cascade="all, delete-orphan"
    )
    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="email", order_by="AgentRun.id", cascade="all, delete-orphan"
    )
    actions: Mapped[list["ProposedAction"]] = relationship(
        back_populates="email", order_by="ProposedAction.id", cascade="all, delete-orphan"
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


class Job(Base):
    """One piece of background work on one email, waiting in the queue or done.

    The queue is this table: no separate broker to run. See app/jobs.py.
    """

    __tablename__ = "jobs"
    # The worker's query filters on status and sorts on run_after.
    __table_args__ = (Index("ix_jobs_status_run_after", "status", "run_after"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.QUEUED)
    attempts: Mapped[int] = mapped_column(default=0)
    max_attempts: Mapped[int]
    # The job is not picked before this time (used to space out retries).
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(120))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    email: Mapped[Email] = relationship(back_populates="jobs")


class AgentRun(Base):
    """One run of the agent on one email: every model call, every tool call, the cost."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(80))
    prompt_version: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))
    steps: Mapped[int]  # number of model calls
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    latency_ms: Mapped[int]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    # The agent's last message, for the person who reviews its proposals.
    note: Mapped[str | None] = mapped_column(Text)
    flags: Mapped[list[str]] = mapped_column(JSONType, default=list)
    # One entry per tool call: step, tool, arguments, result, duration.
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    email: Mapped[Email] = relationship(back_populates="agent_runs")
    actions: Mapped[list["ProposedAction"]] = relationship(
        back_populates="agent_run", order_by="ProposedAction.id"
    )


class ProposedAction(Base):
    """An action the agent proposed. Nothing happens until a person approves it."""

    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20), default=ActionStatus.PROPOSED, index=True)
    order_id: Mapped[str | None] = mapped_column(String(16))
    # What would be done: amount for a refund, message for a reply...
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict)
    rationale: Mapped[str | None] = mapped_column(Text)  # the agent's reason
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(Text)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    email: Mapped[Email] = relationship(back_populates="actions")
    agent_run: Mapped[AgentRun] = relationship(back_populates="actions")
