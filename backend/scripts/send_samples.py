"""Send the sample emails to the running API, to fill the inbox and the approval queue.

The worker then triages each email, and runs the agent on the claims. Watch the queue:
    curl localhost:8000/actions?status=proposed

Usage (from backend/, with the API on localhost:8000):
    python -m scripts.send_samples            # all 24 emails
    python -m scripts.send_samples --limit 3  # the first 3
"""

import argparse
import json
from pathlib import Path

import httpx

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples" / "emails.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="send only the first N emails")
    parser.add_argument("--api", default="http://localhost:8000", help="base URL of the API")
    args = parser.parse_args()

    lines = SAMPLES.read_text(encoding="utf-8").splitlines()[: args.limit]
    with httpx.Client(base_url=args.api, timeout=10) as client:
        for line in lines:
            sample = json.loads(line)
            response = client.post(
                "/emails",
                json={k: sample[k] for k in ("sender", "subject", "body")},
            )
            response.raise_for_status()
            print(f"{sample['id']} -> email {response.json()['id']}: {sample['subject']}")
    print(f"\nSent {len(lines)} email(s). The worker triages them, then the agent runs on claims.")


if __name__ == "__main__":
    main()
