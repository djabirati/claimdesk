# ClaimDesk

![CI](https://github.com/djabirati/claimdesk/actions/workflows/ci.yml/badge.svg)

An LLM-powered triage service for delivery-dispute emails, built for an after-sales team.
Every incoming email is classified, its identifiers are extracted and checked against the
text, and anything uncertain goes to a person. For each claim, an agent checks the order and
the carrier tracking, looks at how people decided on similar past claims (pgvector), then
proposes the actions (refund, replacement, carrier claim, reply): a person approves each one
before it runs. Every model call and tool call is logged with its cost. A read-only MCP
server gives the same data and tools to any MCP client, such as Claude Desktop.

> **Status: step 2 of 5.** Triage pipeline, job queue and worker, agent with human approval,
> similar past cases and MCP server are done. Evals and the dashboard are next (see
> [Roadmap](#roadmap)). The agent's quality has not been measured yet: that is step 3.

## Why this project

In a previous job at a retail group, I built a workflow that classified carrier-dispute
emails for the logistics team and fed a dashboard of the carriers causing the most disputes.
ClaimDesk rebuilds that idea in public, with what production use taught me to care about:
answers validated in code rather than trusted, identifiers that cannot be invented, a log of
every run, and (next) evaluations that block a regression before it ships.

All data in this repository is fictional: the retailer (Maison Lumen), the carriers, the
customers and the orders.

## How it works

### Receiving an email: a queue and a worker

```mermaid
flowchart LR
    A[POST /emails] -->|"one transaction"| Q[("Postgres<br/>email + job")]
    A -->|"202, status received"| C[Caller]
    W1[Worker] -->|"claim: FOR UPDATE SKIP LOCKED"| Q
    W2[Worker] -->|claim| Q
    W1 --> T[Triage]
    T -->|success| D[job done]
    T -->|"temporary provider error"| R["retry later<br/>15 s, 1 min, 4 min"]
    T -->|"rejected request,<br/>invalid output twice"| F[job failed]
```

1. **The API does not call the model.** `POST /emails` stores the email and a `triage` job
   in the same transaction, then answers `202` with status `received`. A slow or failing
   model call never blocks the caller, and an email is never stored without its job.
2. **Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`.** Several workers can run
   at once: each job goes to one of them, and none waits on a row another is claiming.
   The claim is committed right away, and the triage ends its read transaction before the
   model call, so no transaction stays open while waiting for the model.
3. **Only temporary provider errors are retried** (network, timeout, quota, server error),
   with a growing delay. Each model call has a timeout (90 s), kept below the lock timeout
   (15 min). A rejected request (invalid key, unknown model) fails at once, and so does an
   answer that was invalid twice: retrying would give the same result.
4. **Abandoned jobs come back.** A job still `running` after the lock timeout (worker crashed
   or killed) goes back to the queue; its attempt counts, so it cannot loop forever. A job can
   therefore run twice, and the worker skips an email that is already triaged.
5. **A claim gets a second job.** When the triage finds a claim (damaged, late, lost, wrong
   item, refund request), an `agent` job is queued in the same commit that marks the triage
   done. Spam, thank-you notes and other emails stop at the triage.

### Triage of one email

```mermaid
flowchart LR
    A[Job] --> B["Deterministic extraction<br/>order ids, tracking numbers, amounts"]
    B --> C["LLM call<br/>JSON response schema"]
    C --> D{Valid?}
    D -- no --> R["Retry once with<br/>the validation error"] --> D
    D -- yes --> E["Guards<br/>compare answer with extraction"]
    E --> F[("Postgres<br/>email + run log")]
    E --> G{Review flags?}
    G -- yes --> H[needs_review]
    G -- no --> I[triaged]
```

1. **Extraction first.** Regexes find order ids (`ML-104201`, also written `ml104201`),
   tracking numbers and amounts before any model call.
2. **One structured call.** The model gets the email inside delimiters, the extracted values
   as hints, and must answer in a JSON schema (category, urgency, requested action, ids,
   summary, confidence).
3. **Validation in code.** The answer is parsed with Pydantic. If it is invalid, the model
   gets one retry with the exact validation error.
4. **Guards.** An order id or tracking number that does not appear in the email is dropped
   and flagged. Several order ids, a claim without any identifier, low confidence or high
   urgency send the email to review. The model never decides on its own that no human is needed.
5. **Run log.** Each run stores the model, prompt version, attempts, tokens, latency,
   estimated cost, flags, and the raw output when it fails.

### The agent: it checks, it proposes, a person decides

```mermaid
flowchart LR
    J[Agent job] --> E["Embed the email<br/>(once, pgvector)"]
    E --> M["Model call<br/>with 8 tools"]
    M -->|tool calls| T{"Tool"}
    T -->|read| R["get_order, list_customer_orders,<br/>track_parcel (sender's orders only),<br/>find_similar_cases (decisions only)"]
    T -->|propose| P["propose_refund, _replacement,<br/>_carrier_claim, _reply<br/>(validated, kept in memory)"]
    R --> M
    P --> M
    M -->|"text only: note<br/>for the reviewer"| S[("Postgres<br/>run + trace + proposals")]
    S --> Q["GET /actions?status=proposed"]
    Q -->|approve| X["executed (simulated)"]
    Q -->|reject| N[never runs]
```

1. **A loop written by hand.** One step is one model call with the tool declarations. The
   loop runs the tool calls the model asks for, sends the results back, and stops when the
   model answers with text only (its note for the reviewer) or after 8 steps.
2. **Two kinds of tools, none that acts.** Read tools look up orders, the (simulated)
   carrier tracking, and similar past cases. Propose tools record a proposal. Executing an approved action is done by
   `app/actions.py`, which the agent never imports; a test checks it.
3. **Least privilege.** Every tool only sees the orders of the email's sender. An injected
   "refund order ML-104206" from another address gets "not found".
4. **Validation in code, errors back to the model.** A refund above the order total, a second
   reply, an unknown tool or a wrong argument returns an error to the model, which can correct
   its call. The sample email that says "refund 500 euros" for a 27.80 EUR lamp ends with a
   proposal of 27.80 EUR at most.
5. **All or nothing.** Proposals are saved at the end of the run, with the run, in one
   transaction. A run stopped by a provider error keeps its log but no proposal, so the retry
   cannot duplicate a refund.
6. **A person decides.** `POST /actions/{id}/approve` executes the action (simulated: nothing
   is paid, shipped or sent) and `reject` closes it. Approval is a compare-and-set
   (`UPDATE ... WHERE status = 'proposed'`): two people clicking at once execute it once.
7. **Trace.** Each agent run stores its steps, tokens, cost, flags (`step_limit`,
   `no_reply_proposed`, `tool_errors`, `no_embedding`) and one trace entry per tool call:
   arguments, result, duration.

### Similar past cases: learning from people's decisions

Before its first step, the agent embeds the email (Gemini Embedding 2, 768 dimensions) and
stores the vector in Postgres with pgvector, behind an HNSW index. The `find_similar_cases`
tool returns the closest past emails **that a person already decided on**, with what was
decided:

```json
{"similarity": 0.91, "category": "damaged", "requested_action": "replacement",
 "decisions": [{"action": "replacement", "decision": "executed"},
               {"action": "refund", "decision": "rejected", "amount_eur": "89.90"}]}
```

- **Decisions only.** A past case never carries the other email's text, the other
  customer's name or address, or any free text written by the agent or a reviewer. So it
  cannot leak another customer's data into this run, nor bring someone else's instructions
  into the prompt.
- **Only decided cases, only close ones.** Proposals still waiting are left out, and so are
  cases below a cosine similarity of 0.6, or embedded by another model.
- **A hint, not a rule.** The prompt says so: the facts of the claim and the policy come first.
- **Never blocking.** If the embedding call fails, the agent works without similar cases and
  the run gets a `no_embedding` flag.

Emails handled before this feature have no vector: `python -m scripts.backfill_embeddings`
embeds them.

### MCP server: the same data for other assistants

`python -m app.mcp_server` serves ClaimDesk over MCP (stdio). It is **read-only**: list the
inbox, read one email with the agent's note and proposals, read the approval queue, and run
the agent's read tools (`get_order`, `list_customer_orders`, `track_parcel`,
`find_similar_cases`) with the same code and the same limits. Each customer tool takes an
`email_id` and only sees that sender's orders. No tool proposes, approves or executes: an MCP
client is a model, and approving stays a human act, in the app. It reads the database only,
so it needs no model key.

To use it from Claude Desktop, add this to `claude_desktop_config.json` (use the absolute
path of your clone, with Postgres running from `docker compose up`):

```json
{
  "mcpServers": {
    "claimdesk": {
      "command": "/path/to/claimdesk/backend/.venv/bin/python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "DATABASE_URL": "postgresql+psycopg://claimdesk:claimdesk@localhost:5432/claimdesk"
      }
    }
  }
}
```

Then ask, for example: "Which actions are waiting for approval, and what did the agent
check before proposing them?"

## Design decisions

The short version is below; the reasoning is in [docs/decisions.md](docs/decisions.md).

| Decision | Why |
|---|---|
| Deterministic extraction before the LLM | A regex cannot invent an order id. Its output is also the reference used to check the model. |
| The code decides `needs_review`, not the model | Review rules can be tested without any LLM and do not change when the prompt changes. |
| Small `LLMClient` interface | The triage logic never imports a provider SDK. Gemini today, another model by adding one class. |
| A fake LLM client for tests | The test suite runs in CI on every push, for free, and is deterministic. |
| Prompt untrusted, delimiters sanitised | An email cannot close the `<email>` block and add instructions outside it. |
| Postgres + JSONB, migrations with Alembic | One database for orders, emails and run logs; CI checks that migrations match the models. |
| The job queue is a Postgres table | No broker to run. The email and its job are saved in one transaction. |
| Retry temporary provider errors only, with backoff | A quota or network error often passes; an unknown model or an invalid answer comes back the same. |
| No temperature on Gemini 3 | Google advises against it. Answers may vary, which the evaluation suite measures. |
| Agent loop written by hand, no framework | One readable file, tested with the fake client; the step limit and the trace are explicit. |
| The agent proposes, a person approves | The model can be wrong or manipulated; the only path to an action goes through a person. |
| Tools scoped to the sender's orders | An email cannot make the agent read or act on another customer's order. |
| Similar cases return decisions, never text | Another customer's data and words never enter the prompt. |
| pgvector in the same Postgres, HNSW index | No extra vector database to run; the vectors sit next to the decisions they are joined with. |
| A read-only MCP server reusing the agent's tools | One implementation, one set of limits, and no AI client can approve an action. |

## Roadmap

- [x] **Step 1: triage pipeline.** Extraction, structured LLM call, validation and retry,
  guards, run log with cost, REST API, tests, CI with Postgres.
- [x] **Step 2: agent and human approval.**
  - [x] **2a.** Job queue in Postgres and background worker, with retries and recovery of
    abandoned jobs.
  - [x] **2b.** An agent that calls read-only tools (order lookup, carrier tracking), and
    actions (refund, replacement, carrier claim, reply) that it can only *propose*: a person
    approves them before they run.
  - [x] **2c.** Similar past cases with pgvector (decisions only), and a read-only MCP
    server that reuses the agent's tools.
- [ ] **Step 3: evaluations.** Golden dataset, deterministic checks first, LLM-as-judge with a
  written rubric for free text, pass rate over repeated runs, prompt-injection cases, and a
  CI gate that fails on regression. Tracing with Langfuse.
- [ ] **Step 4: dashboard.** React interface: inbox, run details, approval queue, metrics.
- [ ] **Step 5: deployment.** Hosted demo with a public URL.

## Quick start

Requirements: Docker, and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
cp .env.example .env          # then put your GEMINI_API_KEY in .env
docker compose up --build     # Postgres, migrations, API on http://localhost:8000, worker
```

Then open http://localhost:8000/docs, or send an email:

```bash
curl -X POST localhost:8000/emails -H "Content-Type: application/json" -d '{
  "sender": "camille.roux@example.com",
  "subject": "Vase cassé",
  "body": "Bonjour, ma commande ML-104201 est arrivée cassée. Pouvez-vous en renvoyer un ?"
}'
```

The answer has status `received` and a queued job. A few seconds later the worker has
triaged it, and since it is a claim, the agent has proposed actions:

```bash
curl localhost:8000/emails/<id>              # status, triage, runs, jobs, agent run, actions
curl "localhost:8000/actions?status=proposed" # the approval queue
curl -X POST localhost:8000/actions/<action id>/approve \
  -H "Content-Type: application/json" -d '{"reviewer": "Djabi"}'
```

To fill the queue with the 24 sample emails: `cd backend && python -m scripts.send_samples`.
More workers: `docker compose up --scale worker=3`.

### Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d db       # from the repo root, or use your own Postgres
alembic upgrade head
python -m scripts.seed
uvicorn app.main:app --reload # terminal 1: the API
python -m app.worker          # terminal 2: the worker (needs GEMINI_API_KEY)
```

Checks (the same as CI):

```bash
ruff check . && ruff format --check . && mypy app scripts tests && pytest
```

Run the 24 sample emails through the real model and compare with the expected labels:

```bash
python -m scripts.try_samples
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/emails` | Store an email and queue it for triage. Returns `202` with status `received`. |
| `GET` | `/emails?status=needs_review` | List emails, optionally filtered by status. |
| `GET` | `/emails/{id}` | One email with its triage runs, jobs, agent runs (with trace) and proposed actions. |
| `GET` | `/actions?status=proposed` | The approval queue, oldest first. |
| `GET` | `/actions/{id}` | One proposed action. |
| `POST` | `/actions/{id}/approve` | Approve and execute (simulated). Body: `reviewer`, optional `note`. `409` if already decided. |
| `POST` | `/actions/{id}/reject` | Reject: the action never runs. Same body. `409` if already decided. |
| `GET` | `/health` | Liveness check. |

There is no authentication yet: the reviewer's name is given, not proven.

## Project layout

```
backend/
  app/
    api/         FastAPI routes: emails, actions (approval queue)
    llm/         LLMClient interface (JSON and tool calling), Gemini client, fake client
    triage/      extraction, prompt, classifier and guards, pipeline
    agent/       the agent: loop, tools, prompt
    actions.py   approve, reject, execute (simulated): the human side
    similar.py   embeddings and similar past cases (pgvector)
    mcp_server.py  read-only MCP server (python -m app.mcp_server)
    carrier.py   simulated carrier tracking API
    jobs.py      the Postgres job queue: enqueue, claim, retry, recover abandoned jobs
    worker.py    background worker (python -m app.worker): triage and agent jobs
    models.py    SQLAlchemy tables: orders, emails, triage_runs, jobs, agent_runs, actions,
                 email_embeddings
    schemas.py   Pydantic models, including the LLM response schema
  alembic/       database migrations
  scripts/       seed data, sample run, send the samples to the API, embed old emails
  tests/         unit, API, worker, agent, Gemini client (local stand-in server), MCP
                 (in memory and over stdio), and Postgres tests (migrations, queue locking,
                 concurrent approvals, pgvector search)
data/
  seed/          fictional orders
  samples/       24 fictional emails with expected labels
docs/            design decisions
```

## Tech stack

Python 3.12, FastAPI, Pydantic, SQLAlchemy 2, Alembic, PostgreSQL with pgvector (HNSW),
Google Gemini (`google-genai`: Gemini 3.8 Flash, Gemini Embedding 2), MCP Python SDK,
pytest, ruff, mypy, Docker, GitHub Actions.

Built with AI-assisted development (Claude Code).
