from fastapi import FastAPI

from app.api import emails

app = FastAPI(
    title="ClaimDesk",
    description="Triage of delivery dispute emails with an LLM, deterministic guards and logging.",
    version="0.1.0",
)
app.include_router(emails.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
