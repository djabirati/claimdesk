"""Similar past cases: how people decided on claims that look like this one.

Every email the agent works on gets an embedding (a vector of its text). To help with a new
claim, the agent can ask for the closest past emails that a person already decided on, and
see what was decided: which actions were executed, which were rejected.

What a past case shows is deliberately narrow: the category, the request, and the
decisions (action, outcome, amount). Never the text of the other email, the other
customer's name, or any free text written by the agent or a reviewer. So a past case
cannot leak another customer's data into this run, and cannot carry instructions written
by someone else into the prompt.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.enums import ActionStatus
from app.llm.base import LLMClient
from app.models import Email, EmailEmbedding, ProposedAction

DECIDED = (ActionStatus.EXECUTED, ActionStatus.REJECTED)
MAX_EMBEDDED_CHARS = 8000


class SimilarCasesUnavailable(Exception):
    """The search cannot run: no vector for this email, or no pgvector database."""


def embedding_text(email: Email) -> str:
    return f"{email.subject}\n\n{email.body}"[:MAX_EMBEDDED_CHARS]


def ensure_embedding(session: Session, email: Email, llm: LLMClient) -> EmailEmbedding:
    """Embed the email unless it already has a vector. Commits.

    Calls the provider: raises LLMError if the call fails. The transaction opened by
    reading the email is ended first, so none stays open during the call.
    """
    existing = session.get(EmailEmbedding, email.id)
    if existing is not None:
        return existing
    text = embedding_text(email)
    session.commit()
    vector = llm.embed(text)
    row = EmailEmbedding(email_id=email.id, model=llm.embedding_model, embedding=vector)
    session.add(row)
    session.commit()
    return row


def find_similar_cases(
    session: Session, email: Email, *, limit: int, min_similarity: float
) -> list[dict[str, Any]]:
    """The closest past emails with at least one action decided by a person."""
    if session.get_bind().dialect.name != "postgresql":
        raise SimilarCasesUnavailable("Similar case search needs PostgreSQL with pgvector.")
    own = session.get(EmailEmbedding, email.id)
    if own is None:
        raise SimilarCasesUnavailable("This email has no embedding yet.")

    distance = EmailEmbedding.embedding.cosine_distance(own.embedding)
    decided = select(ProposedAction.email_id).where(ProposedAction.status.in_(DECIDED))
    rows = session.execute(
        select(EmailEmbedding.email_id, distance.label("distance"))
        .where(
            EmailEmbedding.email_id != email.id,
            EmailEmbedding.model == own.model,  # vectors of different models do not compare
            EmailEmbedding.email_id.in_(decided),
        )
        .order_by(distance)
        .limit(limit)
    ).all()

    cases = []
    for email_id, dist in rows:
        similarity = round(1 - float(dist), 3)
        if similarity < min_similarity:
            break  # rows are sorted: the next ones are further away
        past = session.get_one(Email, email_id)
        triage = past.triage or {}
        cases.append(
            {
                "similarity": similarity,
                "category": triage.get("category"),
                "requested_action": triage.get("requested_action"),
                "decisions": [
                    {
                        "action": action.kind,
                        "decision": action.status,
                        **(
                            {"amount_eur": action.payload["amount_eur"]}
                            if "amount_eur" in action.payload
                            else {}
                        ),
                    }
                    for action in past.actions
                    if action.status in DECIDED
                ],
            }
        )
    return cases
