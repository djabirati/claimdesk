from decimal import Decimal

from sqlalchemy.orm import Session

from app.config import Settings
from app.enums import EmailStatus, RunStatus
from app.llm.base import LLMClient
from app.models import Email, TriageRun
from app.triage.classifier import classify
from app.triage.prompts import PROMPT_VERSION


def estimate_cost(input_tokens: int, output_tokens: int, settings: Settings) -> Decimal:
    cost = (
        input_tokens * settings.llm_input_price_per_mtok
        + output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000
    return Decimal(str(round(cost, 6)))


def triage_email(session: Session, email: Email, llm: LLMClient, settings: Settings) -> TriageRun:
    """Run triage on a stored email, record the run, update the email. Commits."""
    outcome = classify(
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        llm=llm,
        max_attempts=settings.triage_max_attempts,
    )

    run = TriageRun(
        email=email,
        model=llm.model,
        prompt_version=PROMPT_VERSION,
        status=outcome.status,
        attempts=outcome.attempts,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        latency_ms=outcome.latency_ms,
        cost_usd=estimate_cost(outcome.input_tokens, outcome.output_tokens, settings),
        flags=outcome.flags,
        raw_output=outcome.raw_output,
        error=outcome.error,
    )
    session.add(run)

    if outcome.status is RunStatus.SUCCESS and outcome.result is not None:
        email.triage = outcome.result.model_dump(mode="json")
        email.flags = outcome.flags
        email.status = EmailStatus.NEEDS_REVIEW if outcome.needs_review else EmailStatus.TRIAGED
    else:
        email.status = EmailStatus.FAILED

    session.commit()
    session.refresh(run)
    return run
