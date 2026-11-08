"""Embed the claims stored before similar-case search existed. Safe to run several times.

The agent embeds every email it works on. Emails handled before step 2c have no vector,
so they can never come up as similar cases; this script gives them one.

Usage (from backend/, with GEMINI_API_KEY set):  python -m scripts.backfill_embeddings
"""

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.enums import CLAIM_CATEGORIES
from app.llm.base import LLMError
from app.llm.factory import build_llm_client
from app.models import Email, EmailEmbedding
from app.similar import ensure_embedding


def main() -> None:
    llm = build_llm_client(get_settings())
    with SessionLocal() as session:
        missing = list(
            session.scalars(
                select(Email)
                .outerjoin(EmailEmbedding)
                .where(EmailEmbedding.email_id.is_(None), Email.triage.is_not(None))
                .order_by(Email.id)
            )
        )
        claims = [e for e in missing if (e.triage or {}).get("category") in CLAIM_CATEGORIES]
        done = 0
        for email in claims:
            try:
                ensure_embedding(session, email, llm)
                done += 1
            except LLMError as exc:
                print(f"email {email.id}: {exc}")
        print(f"Embedded {done} of {len(claims)} claim(s) without a vector.")


if __name__ == "__main__":
    main()
