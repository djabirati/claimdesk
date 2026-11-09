"""The agent loop, written by hand: no framework between the model and the tools.

One step is one model call. The model answers with tool calls, the loop runs them and
sends the results back, until the model answers with text only (its note for the
reviewer) or the step limit is reached. Each tool call is logged in the run's trace.

Proposals are kept in memory during the run and saved at the end, with the run, in one
transaction. A run that fails half-way (provider error) saves its log but no proposal, so
a retry starts clean and never duplicates a refund.
"""

import time
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.agent.prompts import AGENT_PROMPT_VERSION, AGENT_SYSTEM_PROMPT, build_agent_prompt
from app.agent.tools import TOOL_SPECS, ToolContext, call_tool
from app.config import Settings
from app.enums import ActionKind, AgentRunStatus
from app.jobs import utcnow
from app.llm.base import LLMClient, LLMError, Message, ToolResult, ToolResults, UserMessage
from app.llm.cost import estimate_cost
from app.models import AgentRun, Email, ProposedAction


def run_agent(
    session: Session,
    email: Email,
    llm: LLMClient,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> AgentRun:
    """Run the agent on a triaged email, then save the run and its proposals. Commits."""
    now = now or utcnow()
    ctx = ToolContext(session=session, email=email, now=now)
    messages: list[Message] = [
        UserMessage(
            build_agent_prompt(
                sender=email.sender,
                subject=email.subject,
                body=email.body,
                triage=email.triage or {},
                flags=list(email.flags or []),
                now=now,
            )
        )
    ]
    trace: list[dict[str, Any]] = []
    status = AgentRunStatus.STEP_LIMIT
    note: str | None = None
    error: str | None = None
    steps = input_tokens = output_tokens = 0
    start = time.perf_counter()

    for step in range(1, settings.agent_max_steps + 1):
        # No transaction stays open while waiting for the model (see decision 7).
        session.commit()
        steps = step
        try:
            turn = llm.generate_with_tools(
                system=AGENT_SYSTEM_PROMPT, messages=messages, tools=TOOL_SPECS
            )
        except LLMError as exc:
            status = AgentRunStatus.ERROR if exc.retryable else AgentRunStatus.REJECTED
            error = str(exc)
            break
        input_tokens += turn.input_tokens
        output_tokens += turn.output_tokens
        messages.append(turn)

        if not turn.tool_calls:
            status = AgentRunStatus.COMPLETED
            note = turn.text or None
            break

        results = []
        for call in turn.tool_calls:
            result, duration_ms = call_tool(ctx, call)
            trace.append(
                {
                    "step": step,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": result,
                    "ms": duration_ms,
                }
            )
            results.append(ToolResult(call=call, content=result))
        messages.append(ToolResults(results))

    flags = []
    if status is AgentRunStatus.STEP_LIMIT:
        flags.append("step_limit")
    if status in (AgentRunStatus.COMPLETED, AgentRunStatus.STEP_LIMIT):
        if not ctx.already_proposed(ActionKind.REPLY):
            flags.append("no_reply_proposed")
        if any("error" in entry["result"] for entry in trace):
            flags.append("tool_errors")

    run = AgentRun(
        email=email,
        model=llm.model,
        prompt_version=AGENT_PROMPT_VERSION,
        status=status,
        steps=steps,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round((time.perf_counter() - start) * 1000),
        cost_usd=estimate_cost(input_tokens, output_tokens, settings),
        note=note,
        flags=flags,
        trace=trace,
        error=error,
    )
    session.add(run)
    # A stopped run (step limit) keeps its proposals: they all wait for a person anyway,
    # and the flag tells the reviewer the agent did not finish.
    if status in (AgentRunStatus.COMPLETED, AgentRunStatus.STEP_LIMIT):
        for proposal in ctx.proposals:
            session.add(
                ProposedAction(
                    email=email,
                    agent_run=run,
                    kind=proposal.kind,
                    order_id=proposal.order_id,
                    payload=proposal.payload,
                    rationale=proposal.rationale,
                )
            )
    session.commit()
    session.refresh(run)
    return run
