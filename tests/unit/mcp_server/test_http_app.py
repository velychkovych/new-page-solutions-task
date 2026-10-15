"""The streamable-HTTP app: /health, /mcp and the DNS-rebinding guard (CONTRACTS §2.1).

Requests go through httpx.ASGITransport into the real Starlette app, in-process.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import respx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from tests.unit.mcp_server.conftest import ServerFactory, make_settings

from helpdesk.mcp_server.server import HEALTH_PATH, MCP_PATH

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


@asynccontextmanager
async def running(server: FastMCP[Any], host_header: str) -> AsyncIterator[httpx.AsyncClient]:
    app = server.streamable_http_app()
    # ASGITransport does not run the lifespan, so start the session manager here.
    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=f"http://{host_header}"
        ) as client,
    ):
        yield client


async def test_health(make_server: ServerFactory) -> None:
    app = make_server().streamable_http_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mcp:8001"
    ) as client:
        response = await client.get(HEALTH_PATH)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_bound_to_all_interfaces_accepts_compose_host(make_server: ServerFactory) -> None:
    server = make_server(make_settings(mcp_host="0.0.0.0"))  # noqa: S104 - compose binding
    async with running(server, "mcp:8001") as client:
        response = await client.post(MCP_PATH, json=INITIALIZE, headers=MCP_HEADERS)
    assert response.status_code == 200
    # json_response=False: answers come as an SSE stream, so mid-call progress can flow.
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_bound_to_loopback_keeps_dns_rebinding_guard(make_server: ServerFactory) -> None:
    server = make_server(make_settings(mcp_host="127.0.0.1"))
    async with running(server, "mcp:8001") as client:
        rejected = await client.post(MCP_PATH, json=INITIALIZE, headers=MCP_HEADERS)
        allowed = await client.post(
            MCP_PATH, json=INITIALIZE, headers={**MCP_HEADERS, "Host": "127.0.0.1:8001"}
        )
    assert rejected.status_code == 421
    assert allowed.status_code == 200


async def test_streamable_http_round_trip(
    make_server: ServerFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json={"items": [], "next_cursor": None}),
        ]
    )
    server = make_server(make_settings(mcp_host="0.0.0.0"))  # noqa: S104 - compose binding
    progress: list[str | None] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append(message)

    app = server.streamable_http_app()
    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mcp:8001"
        ) as http,
        streamable_http_client("http://mcp:8001/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool(
            "search_tickets", {"query": "vpn"}, progress_callback=on_progress
        )

    assert {tool.name for tool in tools.tools} == {
        "search_tickets",
        "get_ticket",
        "create_ticket",
        "add_comment",
    }
    assert result.isError is False
    assert progress == ["Retrying (attempt 2 of 3)"]


@pytest.mark.parametrize("path", ["/", "/admin/chaos", "/tickets"])
async def test_only_mcp_and_health_are_served(make_server: ServerFactory, path: str) -> None:
    app = make_server().streamable_http_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8001"
    ) as client:
        response = await client.get(path)
    assert response.status_code == 404
