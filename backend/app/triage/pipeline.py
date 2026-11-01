from sqlalchemy.orm import Session

from app.config import Settings
from app.enums import EmailStatus, RunStatus
from app.llm.base import LLMClient
from app.llm.cost import estimate_cost
from app.models import Email, TriageRun
from app.triage.classifier import classify
from app.triage.prompts import PROMPT_VERSION

__all__ = ["estimate_cost", "triage_email"]


def triage_email(session: Session, email: Email, llm: LLMClient, settings: Settings) -> TriageRun:
    """Run triage on a stored email and record the run. Commits.

    A successful run updates the email. A failed run leaves the email status as it is:
    the worker decides whether to try again later or to give up (see app/worker.py).
    """
    sender, subject, body = email.sender, email.subject, email.body
    # End the transaction opened by reading the email before the model call, which can
    # take tens of seconds. Left open, it would keep the connection "idle in transaction".
    session.commit()

    outcome = classify(
        sender=sender,
        subject=subject,
        body=body,
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

    session.commit()
    session.refresh(run)
    return run
