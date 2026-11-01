"""Run the triage on the 24 sample emails with the real model and compare with expected labels.

This is a first look, not the evaluation suite: one run per email, exact-match scoring.
Step 3 adds repeated runs, a regression gate in CI and LLM-as-judge for free-text fields.

Usage (from backend/, with GEMINI_API_KEY in .env):  python -m scripts.try_samples
"""

import json
from pathlib import Path

from app.config import get_settings
from app.llm.factory import build_llm_client
from app.triage.classifier import classify
from app.triage.pipeline import estimate_cost

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples" / "emails.jsonl"
FIELDS = ["category", "requested_action", "urgency", "order_id", "tracking_number"]


def main() -> None:
    settings = get_settings()
    llm = build_llm_client(settings)
    samples = [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines()]

    correct = dict.fromkeys(FIELDS + ["needs_review"], 0)
    counted = dict.fromkeys(FIELDS + ["needs_review"], 0)
    tokens_in = tokens_out = 0

    for sample in samples:
        outcome = classify(
            sender=sample["sender"],
            subject=sample["subject"],
            body=sample["body"],
            llm=llm,
            max_attempts=settings.triage_max_attempts,
        )
        tokens_in += outcome.input_tokens
        tokens_out += outcome.output_tokens
        expected = sample["expected"]
        got = outcome.result.model_dump(mode="json") if outcome.result else {}

        misses = []
        for name in FIELDS:
            counted[name] += 1
            if got.get(name) == expected[name]:
                correct[name] += 1
            else:
                misses.append(f"{name}: expected {expected[name]!r}, got {got.get(name)!r}")
        if "needs_review" in expected:
            counted["needs_review"] += 1
            if outcome.needs_review == expected["needs_review"]:
                correct["needs_review"] += 1
            else:
                misses.append(f"needs_review: expected {expected['needs_review']}")

        mark = "OK " if not misses else "ERR"
        print(f"{mark} {sample['id']} [{outcome.status}] flags={outcome.flags}")
        for miss in misses:
            print(f"      {miss}")

    print("\nAccuracy per field:")
    for name in correct:
        print(f"  {name:18} {correct[name]}/{counted[name]}")
    cost = estimate_cost(tokens_in, tokens_out, settings)
    print(f"\nTokens: {tokens_in} in, {tokens_out} out. Estimated cost: ${cost}")


if __name__ == "__main__":
    main()
