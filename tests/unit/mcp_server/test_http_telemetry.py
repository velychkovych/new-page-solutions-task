"""App-start telemetry for the MCP server (CONTRACTS §6), in-process.

The app's ASGI lifespan is driven the way uvicorn drives it, and requests go through
httpx.ASGITransport and the real MCP client. Spans are read from <tmp>/traces/mcp.jsonl
after the lifespan has ended (its exit flushes them).
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from opentelemetry import propagate
from opentelemetry.context import Context
from starlette.applications import Starlette
from starlette.requests import Request
from tests.unit.mcp_server.conftest import TOKEN, ServerFactory, make_settings

from helpdesk.mcp_server import __main__ as entry
from helpdesk.mcp_server.request_trace import SCOPE_KEY, tool_span_parent
from helpdesk.mcp_server.server import HEALTH_PATH, build_http_app
from helpdesk.shared import trace_attributes as attr
from helpdesk.shared.settings import Settings

TICKET = {
    "id": "INC-0003",
    "title": "VPN keeps dropping",
    "description": "The VPN disconnects every few minutes.",
    "status": "open",
    "priority": "high",
    "category": "network",
    "reporter_id": "alice",
    "assignee_team": None,
    "created_at": "2026-10-01T09:00:00Z",
    "updated_at": "2026-10-01T09:00:00Z",
    "comments": [],
}
# Distinct callers, so a span that joins the wrong request's trace is caught.
SESSION_TRACE, SESSION_SPAN = "11111111111111111111111111111111", "1111111111111111"
CALL_TRACE, CALL_SPAN = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


def telemetry_settings(tmp_path: Path) -> Settings:
    # Compose binding, so `Host: mcp:8001` passes FastMCP's DNS-rebinding guard.
    return make_settings(
        mcp_host="0.0.0.0",  # noqa: S104
        traces_dir=tmp_path / "traces",
        otel_exporter_otlp_endpoint=None,
    )


def records(settings: Settings) -> list[dict[str, Any]]:
    path = settings.traces_dir / "mcp.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@asynccontextmanager
async def lifespan_running(app: Starlette) -> AsyncIterator[None]:
    """Run the ASGI lifespan protocol in its own task, as uvicorn does."""
    to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}
    task = asyncio.create_task(app(scope, to_app.get, from_app.put))
    await to_app.put({"type": "lifespan.startup"})
    started = await from_app.get()
    assert started["type"] == "lifespan.startup.complete", started
    try:
        yield
    finally:
        await to_app.put({"type": "lifespan.shutdown"})
        stopped = await from_app.get()
        await task
        assert stopped["type"] == "lifespan.shutdown.complete", stopped


def traceparent_per_call(call_traceparent: str | None) -> Any:
    """An httpx request hook: the tools/call request gets `call_traceparent` (or none),
    every other MCP request (initialize, notifications) the session's traceparent."""

    async def hook(request: httpx.Request) -> None:
        if b'"tools/call"' in request.content:
            if call_traceparent is not None:
                request.headers["traceparent"] = call_traceparent
        else:
            request.headers["traceparent"] = f"00-{SESSION_TRACE}-{SESSION_SPAN}-01"

    return hook


async def call_get_ticket(app: Starlette, call_traceparent: str | None) -> None:
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://mcp:8001",
            event_hooks={"request": [traceparent_per_call(call_traceparent)]},
        ) as http,
        streamable_http_client("http://mcp:8001/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
        assert result.isError is False
        assert (await http.get(HEALTH_PATH)).status_code == 200


async def test_app_start_sets_up_mcp_telemetry_and_the_tool_span_joins_the_callers_trace(
    make_server: ServerFactory, mock_api: respx.Router, tmp_path: Path
) -> None:
    mock_api.get("/tickets/INC-0003").respond(200, json=TICKET)
    settings = telemetry_settings(tmp_path)
    app = build_http_app(make_server(settings, global_tracer=True), settings)

    async with lifespan_running(app):
        await call_get_ticket(app, f"00-{CALL_TRACE}-{CALL_SPAN}-01")

    spans = records(settings)
    assert spans, "no spans were written"
    assert {span["service"] for span in spans} == {"mcp"}
    (tool,) = [span for span in spans if span["name"] == attr.SPAN_TOOL]
    assert tool["attributes"][attr.TOOL_NAME] == "get_ticket"
    # The tool span sits under the server span of the request that carried the call,
    # which itself continues the caller's trace.
    (server,) = [span for span in spans if span["span_id"] == tool["parent_span_id"]]
    assert server["name"] == "POST /mcp"
    assert server["parent_span_id"] == CALL_SPAN
    assert tool["trace_id"] == server["trace_id"] == CALL_TRACE
    assert server["start_time"] <= tool["start_time"] <= tool["end_time"] <= server["end_time"]
    assert not [span for span in spans if HEALTH_PATH in span["name"]]
    trace_file = (settings.traces_dir / "mcp.jsonl").read_text(encoding="utf-8")
    assert TOKEN not in trace_file


async def test_without_a_traceparent_the_tool_span_stays_in_its_own_requests_trace(
    make_server: ServerFactory, mock_api: respx.Router, tmp_path: Path
) -> None:
    """The session task was started by the initialize request; the tool span must not
    inherit that request's trace."""
    mock_api.get("/tickets/INC-0003").respond(200, json=TICKET)
    settings = telemetry_settings(tmp_path)
    app = build_http_app(make_server(settings, global_tracer=True), settings)

    async with lifespan_running(app):
        await call_get_ticket(app, None)

    spans = records(settings)
    (tool,) = [span for span in spans if span["name"] == attr.SPAN_TOOL]
    (server,) = [span for span in spans if span["span_id"] == tool["parent_span_id"]]
    assert server["name"] == "POST /mcp"
    assert server["parent_span_id"] is None
    assert tool["trace_id"] == server["trace_id"] != SESSION_TRACE


async def test_spans_are_recorded_only_while_the_app_runs(
    make_server: ServerFactory, mock_api: respx.Router, tmp_path: Path
) -> None:
    mock_api.get("/tickets/INC-0003").respond(200, json=TICKET)
    settings = telemetry_settings(tmp_path)
    server = make_server(settings, global_tracer=True)
    app = build_http_app(server, settings)

    # Served without its lifespan (the session manager is started by hand): no setup.
    async with server.session_manager.run():
        await call_get_ticket(app, f"00-{CALL_TRACE}-{CALL_SPAN}-01")
    assert not (settings.traces_dir / "mcp.jsonl").exists()


async def test_serve_runs_the_app_with_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built: list[Starlette] = []
    served: list[uvicorn.Config] = []

    def recording_build_http_app(*args: Any) -> Starlette:
        built.append(build_http_app(*args))
        return built[-1]

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            served.append(config)

        async def serve(self) -> None:
            return None

    monkeypatch.setattr(entry, "build_http_app", recording_build_http_app)
    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)
    await entry.serve(telemetry_settings(tmp_path))
    (config,) = served
    assert [config.app] == built
    # A failing lifespan (and so a failed telemetry setup) stops the server.
    assert config.lifespan == "on"


# --- choosing the tool span's parent ----------------------------------------------------


def http_request(headers: dict[str, str], scope_extra: dict[str, Any] | None = None) -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        **(scope_extra or {}),
    }
    return Request(scope)


def test_parent_is_the_context_captured_for_the_request() -> None:
    captured = Context({"marker": 1})
    request = http_request(
        {"traceparent": f"00-{CALL_TRACE}-{CALL_SPAN}-01"}, {SCOPE_KEY: captured}
    )
    assert tool_span_parent(request) is captured


def test_parent_falls_back_to_the_callers_traceparent() -> None:
    request = http_request({"traceparent": f"00-{CALL_TRACE}-{CALL_SPAN}-01"})
    parent = tool_span_parent(request)
    assert parent is not None
    carrier: dict[str, str] = {}
    propagate.inject(carrier, context=parent)
    assert carrier["traceparent"] == f"00-{CALL_TRACE}-{CALL_SPAN}-01"


def test_without_a_traceparent_the_parent_is_empty_not_the_current_context() -> None:
    parent = tool_span_parent(http_request({}))
    assert parent is not None
    carrier: dict[str, str] = {}
    propagate.inject(carrier, context=parent)
    assert "traceparent" not in carrier


@pytest.mark.parametrize("request_value", [None, object()])
def test_non_http_transports_use_the_current_context(request_value: object) -> None:
    assert tool_span_parent(request_value) is None
