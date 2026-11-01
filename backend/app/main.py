from fastapi import FastAPI

from app.api import actions, emails

app = FastAPI(
    title="ClaimDesk",
    description="Triage of delivery dispute emails with an LLM, an agent that proposes "
    "actions, and human approval before anything runs.",
    version="0.1.0",
)
app.include_router(emails.router)
app.include_router(actions.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
