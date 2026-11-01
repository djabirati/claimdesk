import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.enums import Category, RequestedAction, Urgency

ORDER_ID_PATTERN = re.compile(r"^ML-?(\d{6})$", re.IGNORECASE)


class TriageOutput(BaseModel):
    """The structured answer the LLM must return for one email.

    This class is sent to the model as the response schema, then used again to
    validate what comes back. Constraints that the provider's schema format may not
    support (patterns, lengths) are enforced in validators, so the model never sees
    them but invalid answers are still rejected.
    """

    category: Category = Field(description="Main reason for the email.")
    urgency: Urgency = Field(description="How quickly a person should look at it.")
    order_id: str | None = Field(
        default=None,
        description="Order id exactly as written in the email (format ML-XXXXXX), or null.",
    )
    tracking_number: str | None = Field(
        default=None, description="Tracking number exactly as written in the email, or null."
    )
    requested_action: RequestedAction = Field(description="What the customer asks for.")
    summary: str = Field(description="One sentence in French summarising the problem.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the category, 0 to 1.")

    @field_validator("order_id")
    @classmethod
    def order_id_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        match = ORDER_ID_PATTERN.match(value.strip())
        if not match:
            raise ValueError("order_id must look like ML-XXXXXX (6 digits)")
        # Customers write "ml104233" or "ML-104233": store one canonical form.
        return f"ML-{match.group(1)}"

    @field_validator("tracking_number")
    @classmethod
    def tracking_upper(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @field_validator("summary")
    @classmethod
    def summary_length(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must not be empty")
        if len(value) > 280:
            raise ValueError("summary must be 280 characters or fewer")
        return value


# ---------- API schemas ----------


class EmailIn(BaseModel):
    sender: str = Field(min_length=3, max_length=255)
    subject: str = Field(max_length=500)
    body: str = Field(min_length=1, max_length=20_000)


class TriageRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model: str
    prompt_version: str
    status: str
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Decimal
    flags: list[str]
    error: str | None
    created_at: datetime | None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    status: str
    attempts: int
    max_attempts: int
    run_after: datetime
    last_error: str | None
    created_at: datetime | None
    finished_at: datetime | None


class AgentRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model: str
    prompt_version: str
    status: str
    steps: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: Decimal
    note: str | None
    flags: list[str]
    trace: list[dict[str, Any]]
    error: str | None
    created_at: datetime | None


class ActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email_id: int
    agent_run_id: int
    kind: str
    status: str
    order_id: str | None
    payload: dict[str, Any]
    rationale: str | None
    created_at: datetime | None
    decided_by: str | None
    decided_at: datetime | None
    decision_note: str | None
    executed_at: datetime | None
    result: dict[str, Any] | None


class DecisionIn(BaseModel):
    # No authentication yet: the reviewer's name is given, not proven (see the README).
    reviewer: str = Field(min_length=2, max_length=120)
    note: str | None = Field(default=None, max_length=1000)


class EmailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sender: str
    subject: str
    body: str
    received_at: datetime | None
    status: str
    triage: TriageOutput | None
    flags: list[str]


class EmailDetail(EmailOut):
    runs: list[TriageRunOut]
    jobs: list[JobOut]
    agent_runs: list[AgentRunOut]
    actions: list[ActionOut]
