"""Triage one email: deterministic extraction, one LLM call (plus a retry), deterministic checks.

The model proposes; the code decides. The model never decides on its own whether an email
needs a human: that comes from the checks below, which can be tested without any LLM.
"""

import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.enums import CLAIM_CATEGORIES, RunStatus, Urgency
from app.llm.base import LLMClient, LLMError
from app.schemas import TriageOutput
from app.triage.extract import Extracted, extract
from app.triage.prompts import SYSTEM_PROMPT, build_user_prompt

LOW_CONFIDENCE_THRESHOLD = 0.6

# Flags that send the email to a person instead of letting it continue automatically.
REVIEW_FLAGS = {
    "order_id_not_in_email",
    "tracking_not_in_email",
    "missing_identifiers",
    "multiple_order_ids",
    "low_confidence",
    "high_urgency",
}


@dataclass
class TriageOutcome:
    status: RunStatus
    result: TriageOutput | None
    flags: list[str] = field(default_factory=list)
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw_output: str | None = None
    error: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.status is not RunStatus.SUCCESS or bool(REVIEW_FLAGS & set(self.flags))


def apply_guards(result: TriageOutput, extracted: Extracted) -> tuple[TriageOutput, list[str]]:
    """Compare the model's answer with what pattern matching found in the email.

    Returns a corrected copy of the result and the list of flags raised.
    """
    flags: list[str] = []
    order_id = result.order_id
    tracking = result.tracking_number

    # An identifier that is not in the email was invented: drop it rather than act on it.
    if order_id and order_id not in extracted.order_ids:
        flags.append("order_id_not_in_email")
        order_id = None
    if tracking and tracking not in extracted.tracking_numbers:
        flags.append("tracking_not_in_email")
        tracking = None

    # The model missed an identifier that pattern matching found without ambiguity.
    if order_id is None and len(extracted.order_ids) == 1:
        order_id = extracted.order_ids[0]
        flags.append("order_id_filled_from_pattern")
    if tracking is None and len(extracted.tracking_numbers) == 1:
        tracking = extracted.tracking_numbers[0]
        flags.append("tracking_filled_from_pattern")

    if len(extracted.order_ids) > 1:
        flags.append("multiple_order_ids")
    if result.category in CLAIM_CATEGORIES and order_id is None and tracking is None:
        flags.append("missing_identifiers")
    if result.confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("low_confidence")
    if result.urgency is Urgency.HIGH:
        flags.append("high_urgency")

    corrected = result.model_copy(update={"order_id": order_id, "tracking_number": tracking})
    return corrected, flags


def _short_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}" for err in exc.errors()
    )


def classify(
    *, sender: str, subject: str, body: str, llm: LLMClient, max_attempts: int = 2
) -> TriageOutcome:
    extracted = extract(f"{subject}\n{body}")
    prompt = build_user_prompt(sender=sender, subject=subject, body=body, extracted=extracted)
    outcome = TriageOutcome(status=RunStatus.INVALID_OUTPUT, result=None)
    start = time.perf_counter()

    next_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        try:
            response = llm.generate_json(
                system=SYSTEM_PROMPT, prompt=next_prompt, schema=TriageOutput
            )
        except LLMError as exc:
            outcome.status = RunStatus.ERROR
            outcome.error = str(exc)
            break

        outcome.input_tokens += response.input_tokens
        outcome.output_tokens += response.output_tokens
        outcome.raw_output = response.text

        try:
            result = TriageOutput.model_validate_json(response.text)
        except ValidationError as exc:
            outcome.error = _short_errors(exc)
            # Tell the model exactly what was wrong; one retry fixes most format slips.
            next_prompt = (
                f"{prompt}\n\nYour previous answer was rejected: {outcome.error}. "
                "Return valid JSON that follows the schema."
            )
            continue

        outcome.result, outcome.flags = apply_guards(result, extracted)
        outcome.status = RunStatus.SUCCESS
        outcome.error = None
        break

    outcome.latency_ms = round((time.perf_counter() - start) * 1000)
    return outcome
