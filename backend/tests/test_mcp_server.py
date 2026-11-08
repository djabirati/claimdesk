"""The MCP server, through a real MCP client: in memory, and once over stdio."""

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.tools import TOOLS
from app.config import Settings
from app.db import Base
from app.enums import ActionKind, ActionStatus, EmailStatus
from app.mcp_server import AGENT_READ_TOOLS, create_server
from app.models import AgentRun, Email, ProposedAction
from tests.conftest import add_seed_orders

Factory = sessionmaker[Session]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def server_factory(session_factory: Factory) -> Factory:
    with session_factory() as session:
        add_seed_orders(session)
    return session_factory


def add_email_with_proposals(factory: Factory) -> int:
    """camille.roux's broken vase, triaged, with a run that proposed a refund and a reply."""
    with factory() as session:
        email = Email(
            sender="camille.roux@example.com",
            subject="Vase cassé",
            body="Ma commande ML-104201 est arrivée cassée.",
            status=EmailStatus.TRIAGED,
            triage={"category": "damaged", "urgency": "normal"},
            flags=[],
        )
        run = AgentRun(
            email=email,
            model="fake",
            prompt_version="agent-v2",
            status="completed",
            steps=2,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            cost_usd=0,
            note="Commande vérifiée, je propose un remboursement.",
            flags=[],
            trace=[],
        )
        session.add_all(
            [
                email,
                run,
                ProposedAction(
                    email=email,
                    agent_run=run,
                    kind=ActionKind.REFUND,
                    order_id="ML-104201",
                    payload={"amount_eur": "89.90"},
                    rationale="Vase cassé.",
                ),
                ProposedAction(
                    email=email,
                    agent_run=run,
                    kind=ActionKind.REPLY,
                    payload={"message": "Bonjour..."},
                    status=ActionStatus.REJECTED,
                ),
            ]
        )
        session.commit()
        return email.id


def payload(result: CallToolResult) -> Any:
    assert not result.is_error, result.content
    texts = [json.loads(c.text) for c in result.content if c.type == "text"]
    return texts[0] if len(texts) == 1 else texts


@pytest.mark.anyio
async def test_only_read_tools_are_exposed(server_factory: Factory) -> None:
    server = create_server(server_factory, Settings(_env_file=None))  # type: ignore[call-arg]

    async with Client(server) as client:
        tools = (await client.list_tools()).tools

    names = {tool.name for tool in tools}
    assert names == {"list_emails", "get_email", "list_pending_actions", *AGENT_READ_TOOLS}
    # Every read tool of the agent is offered, and nothing that proposes or decides.
    assert set(AGENT_READ_TOOLS) == {t.name for t in TOOLS if t.effect == "read"}
    assert not any(n.startswith(("propose", "approve", "reject", "execute")) for n in names)
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)


@pytest.mark.anyio
async def test_order_tools_only_see_the_orders_of_the_email_sender(
    server_factory: Factory,
) -> None:
    email_id = add_email_with_proposals(server_factory)
    server = create_server(server_factory, Settings(_env_file=None))  # type: ignore[call-arg]

    async with Client(server) as client:
        own = payload(
            await client.call_tool("get_order", {"email_id": email_id, "order_id": "ML-104201"})
        )
        other = payload(
            await client.call_tool("get_order", {"email_id": email_id, "order_id": "ML-104206"})
        )
        orders = payload(await client.call_tool("list_customer_orders", {"email_id": email_id}))

    assert own["total_eur"] == "89.90"
    assert other == {"error": "No order with this id for this customer."}
    assert [o["order_id"] for o in orders["orders"]] == ["ML-104201"]


@pytest.mark.anyio
async def test_an_email_shows_its_triage_the_agent_note_and_the_proposals(
    server_factory: Factory,
) -> None:
    email_id = add_email_with_proposals(server_factory)
    server = create_server(server_factory, Settings(_env_file=None))  # type: ignore[call-arg]

    async with Client(server) as client:
        email = payload(await client.call_tool("get_email", {"email_id": email_id}))
        pending = payload(await client.call_tool("list_pending_actions", {}))
        inbox = payload(await client.call_tool("list_emails", {"status": "triaged"}))

    assert email["category"] == "damaged"
    assert email["agent_note"] == "Commande vérifiée, je propose un remboursement."
    assert [(a["kind"], a["status"]) for a in email["actions"]] == [
        ("refund", "proposed"),
        ("reply", "rejected"),
    ]
    # The queue holds only what still waits for a person.
    assert pending["kind"] == "refund"
    assert inbox["email_id"] == email_id


@pytest.mark.anyio
async def test_an_unknown_email_is_a_tool_error(server_factory: Factory) -> None:
    server = create_server(server_factory, Settings(_env_file=None))  # type: ignore[call-arg]

    async with Client(server) as client:
        result = await client.call_tool("get_email", {"email_id": 999})

    assert result.is_error
    assert "No email with id 999" in result.content[0].text  # type: ignore[union-attr]


@pytest.fixture
def sqlite_file(tmp_path: Path) -> Iterator[str]:
    url = f"sqlite:///{tmp_path / 'claimdesk.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        add_seed_orders(session)
    engine.dispose()
    yield url


@pytest.mark.anyio
async def test_the_stdio_entry_point_serves_the_tools(sqlite_file: str, tmp_path: Path) -> None:
    # What Claude Desktop does: start `python -m app.mcp_server` from any directory and talk
    # over stdin/stdout. Anything printed on stdout besides the protocol would break this.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env={"PATH": os.environ.get("PATH", ""), "DATABASE_URL": sqlite_file},
        cwd=str(tmp_path),
    )

    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        inbox = await client.call_tool("list_emails", {})

    assert "get_order" in {tool.name for tool in tools}
    assert not inbox.is_error


@pytest.mark.anyio
async def test_each_call_sees_the_orders_of_the_email_it_names(server_factory: Factory) -> None:
    camille = add_email_with_proposals(server_factory)
    with server_factory() as session:
        hugo = Email(
            sender="hugo.lambert@example.com",
            subject="Colis en retard",
            body="Toujours rien.",
            status=EmailStatus.TRIAGED,
            flags=[],
        )
        session.add(hugo)
        session.commit()
        hugo_id = hugo.id
    server = create_server(server_factory, Settings(_env_file=None))  # type: ignore[call-arg]

    async with Client(server) as client:
        hugos = payload(await client.call_tool("list_customer_orders", {"email_id": hugo_id}))
        camilles = payload(await client.call_tool("list_customer_orders", {"email_id": camille}))

    assert [o["order_id"] for o in hugos["orders"]] == ["ML-104206"]
    assert [o["order_id"] for o in camilles["orders"]] == ["ML-104201"]
