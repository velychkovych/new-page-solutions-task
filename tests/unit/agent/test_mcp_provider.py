"""The MCP tool provider against the real MCP server app, in-process (CONTRACTS §2).

The provider runs the adapter's real streamable-HTTP client; its httpx client talks to the
server's Starlette app through httpx.ASGITransport, and respx stubs the mock behind it.
Nothing opens a socket.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import respx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from starlette.applications import Starlette

from helpdesk.agent.graph import ensure_no_injected_args
from helpdesk.agent.mcp_provider import McpToolProvider, call_timeout_s, model_visible_schema
from helpdesk.agent.tool_provider import ToolTransportError
from helpdesk.mcp_server.mock_client import MockApiClient, create_mock_http_client
from helpdesk.mcp_server.redaction import SecretRedactor
from helpdesk.mcp_server.retry import RetryPolicy
from helpdesk.mcp_server.server import build_server
from helpdesk.shared.errors import ErrorCode, parse_tool_error
from helpdesk.shared.settings import Settings
from helpdesk.shared.tickets import IDEMPOTENCY_KEY_HEADER
from helpdesk.shared.tools import (
    INJECTED_ARGS,
    AddCommentArgs,
    AddCommentInput,
    CreateTicketArgs,
    CreateTicketInput,
    GetTicketArgs,
    SearchTicketsArgs,
    ToolName,
)

MOCK_URL = "http://mock.test"
MCP_URL = "http://mcp:8001/mcp"
TOKEN = "tok-provider-test-2b7e9c41"
NOW = "2026-10-01T09:00:00Z"
TICKET: dict[str, Any] = {
    "id": "INC-0003",
    "title": "VPN keeps dropping",
    "description": "The VPN disconnects every few minutes.",
    "status": "open",
    "priority": "high",
    "category": "network",
    "reporter_id": "alice",
    "assignee_team": None,
    "created_at": NOW,
    "updated_at": NOW,
    "comments": [],
}
SEARCH: dict[str, Any] = {
    "items": [
        {k: TICKET[k] for k in ("id", "title", "status", "priority", "category", "updated_at")}
    ],
    "next_cursor": None,
}
KEY = "6f1c2a9e-0d4b-4c51-9a77-1f0b8e3c2d10"

type ClientFactory = Callable[..., httpx.AsyncClient]


def server_settings() -> Settings:
    # 0.0.0.0 as in compose, so `Host: mcp:8001` passes FastMCP's DNS-rebinding guard.
    return Settings(
        _env_file=None,
        mock_url=MOCK_URL,
        mock_api_token=TOKEN,
        mcp_host="0.0.0.0",  # noqa: S104
        otel_exporter_otlp_endpoint=None,
    )


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def mock_api() -> respx.Router:
    return respx.Router(base_url=MOCK_URL, assert_all_called=False)


@pytest.fixture
async def mcp_app(mock_api: respx.Router) -> AsyncIterator[Starlette]:
    """The real FastMCP app; retries do not wait, so retry tests stay fast."""
    settings = server_settings()
    http = create_mock_http_client(
        settings, TOKEN, transport=httpx.MockTransport(mock_api.async_handler)
    )
    client = MockApiClient(
        http, RetryPolicy.from_settings(settings), sleep=_no_sleep, uniform=lambda low, _: low
    )
    server = build_server(
        settings,
        client,
        redactor=SecretRedactor([TOKEN]),
        tracer=TracerProvider().get_tracer("tests.mcp_provider"),
    )
    app = server.streamable_http_app()
    # ASGITransport does not run the lifespan, so the session manager is started here, in
    # a task of its own: its anyio cancel scope must be left in the task that entered it,
    # and pytest-asyncio tears fixtures down in another task.
    started, stop = asyncio.Event(), asyncio.Event()

    async def run_session_manager() -> None:
        async with http, server.session_manager.run():
            started.set()
            await stop.wait()

    runner = asyncio.create_task(run_session_manager())
    await started.wait()
    try:
        yield app
    finally:
        stop.set()
        await runner


def asgi_clients(
    app: Starlette, created: list[httpx.AsyncClient] | None = None, **client_kwargs: Any
) -> ClientFactory:
    """The adapter's McpHttpClientFactory, with the ASGI app as the transport."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            headers=headers,
            timeout=timeout,
            auth=auth,
            **client_kwargs,
        )
        if created is not None:
            created.append(client)
        return client

    return factory


@pytest.fixture
def provider(mcp_app: Starlette) -> McpToolProvider:
    return McpToolProvider(MCP_URL, timeout_s=20.0, httpx_client_factory=asgi_clients(mcp_app))


def _visible(model: type[Any]) -> tuple[dict[str, Any], set[str]]:
    schema = model.model_json_schema(by_alias=True)
    return schema["properties"], set(schema.get("required", []))


# --- list_tools --------------------------------------------------------------------------


async def test_list_tools_returns_the_four_tools_with_descriptions(
    provider: McpToolProvider,
) -> None:
    definitions = await provider.list_tools()
    assert sorted(d.name for d in definitions) == sorted(t.value for t in ToolName)
    search = next(d for d in definitions if d.name == ToolName.SEARCH_TICKETS)
    assert "Always call before proposing a new ticket, to find duplicates." in search.description


async def test_list_tools_strips_injected_args_from_properties_and_required(
    provider: McpToolProvider,
) -> None:
    definitions = {d.name: d for d in await provider.list_tools()}
    ensure_no_injected_args(list(definitions.values()))
    for tool, visible_model in (
        (ToolName.CREATE_TICKET, CreateTicketArgs),
        (ToolName.ADD_COMMENT, AddCommentArgs),
    ):
        schema = definitions[tool].input_schema
        properties, required = _visible(visible_model)
        assert schema["properties"] == properties
        assert isinstance(schema["required"], list)
        assert set(schema["required"]) == required
        for injected in INJECTED_ARGS[tool]:
            assert injected not in json.dumps(schema["properties"])
            assert injected not in schema["required"]


async def test_list_tools_leaves_read_tool_schemas_whole(provider: McpToolProvider) -> None:
    definitions = {d.name: d for d in await provider.list_tools()}
    reads = ((ToolName.SEARCH_TICKETS, SearchTicketsArgs), (ToolName.GET_TICKET, GetTicketArgs))
    for tool, model in reads:
        properties, required = _visible(model)
        schema = definitions[tool].input_schema
        assert schema["properties"] == properties
        listed = schema.get("required") or []
        assert isinstance(listed, list)
        assert set(listed) == required


def test_model_visible_schema_does_not_change_the_given_schema() -> None:
    full = CreateTicketInput.model_json_schema(by_alias=True)
    before = json.dumps(full, sort_keys=True)
    visible = model_visible_schema(ToolName.CREATE_TICKET, full)
    assert json.dumps(full, sort_keys=True) == before
    assert "reporter_id" in full["properties"]
    assert isinstance(visible["properties"], dict)
    assert "reporter_id" not in visible["properties"]
    # A tool with nothing injected is returned as it is.
    comment = AddCommentInput.model_json_schema(by_alias=True)
    assert model_visible_schema("some_other_tool", comment) == comment


async def test_list_tools_follows_pagination() -> None:
    """tools/list in pages: every page is collected (the adapter follows nextCursor)."""
    tools = [
        {"name": f"tool_{i}", "description": f"d{i}", "inputSchema": {"type": "object"}}
        for i in range(3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "paged", "version": "0"},
            }
        else:
            cursor = (message.get("params") or {}).get("cursor")
            page = int(cursor or 0)
            result = {"tools": [tools[page]]}
            if page + 1 < len(tools):
                result["nextCursor"] = str(page + 1)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": message["id"], "result": result},
            headers={"mcp-session-id": "s1"},
        )

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers)

    paged = McpToolProvider(MCP_URL, timeout_s=5.0, httpx_client_factory=factory)
    assert [d.name for d in await paged.list_tools()] == ["tool_0", "tool_1", "tool_2"]


# --- call: results ------------------------------------------------------------------------


async def test_call_returns_the_tool_result_text(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(200, json=SEARCH)
    result = await provider.call("search_tickets", {"query": "vpn"})
    assert result.is_error is False
    assert json.loads(result.text)["items"][0]["id"] == "INC-0003"
    assert route.call_count == 1
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_tool_error_keeps_the_exact_contract_text(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets/INC-0099").respond(404, json={"error": "not_found"})
    result = await provider.call("get_ticket", {"ticket_id": "INC-0099"})
    assert result.is_error is True
    assert result.text.startswith("ERROR NOT_FOUND: ")
    parsed = parse_tool_error(result.text)
    assert parsed is not None
    assert parsed.code is ErrorCode.NOT_FOUND


async def test_invalid_arguments_come_back_as_a_tool_error_not_an_exception(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    route = mock_api.get(url__regex=r".*")
    result = await provider.call("get_ticket", {"ticket_id": "INC-7"})
    assert result.is_error is True
    assert result.text.startswith("ERROR INVALID_ARGUMENT: ")
    assert route.call_count == 0


async def test_write_sends_injected_args_and_the_key(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    route = mock_api.post("/tickets").respond(201, json=TICKET)
    args = {
        "title": "VPN keeps dropping",
        "description": "The VPN disconnects every few minutes.",
        "category": "network",
        "priority": "high",
        "reporter_id": "alice",
        "idempotency_key": KEY,
    }
    result = await provider.call("create_ticket", args)
    assert result.is_error is False
    assert json.loads(result.text)["id"] == "INC-0003"
    request = route.calls.last.request
    assert request.headers[IDEMPOTENCY_KEY_HEADER] == KEY
    assert json.loads(request.content)["reporter_id"] == "alice"


async def test_retries_happen_once_in_the_server_only(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    """Retries exhausted: the server's contract error comes back after exactly its own
    attempts; the provider adds none."""
    route = mock_api.get("/tickets").respond(503, json={"error": "upstream_error"})
    result = await provider.call("search_tickets", {"query": "vpn"})
    assert result.is_error is True
    assert result.text.startswith("ERROR UPSTREAM_UNAVAILABLE: ")
    assert route.call_count == server_settings().retry_max_attempts


# --- call: progress -----------------------------------------------------------------------


async def test_progress_messages_reach_the_callers_callback(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "0"}),
            httpx.Response(200, json=SEARCH),
        ]
    )
    seen: list[str] = []

    async def on_progress(message: str) -> None:
        seen.append(message)

    result = await provider.call("search_tickets", {"query": "vpn"}, on_progress=on_progress)
    assert result.is_error is False
    assert seen == ["Retrying (attempt 2 of 3)", "Retrying (attempt 3 of 3)"]


async def test_progress_goes_only_to_the_call_that_asked_for_it(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json=SEARCH),
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json=SEARCH),
        ]
    )
    first: list[str] = []
    second: list[str] = []

    async def record_first(message: str) -> None:
        first.append(message)

    async def record_second(message: str) -> None:
        second.append(message)

    await provider.call("search_tickets", {"query": "vpn"}, on_progress=record_first)
    await provider.call("search_tickets", {"query": "vpn"}, on_progress=record_second)
    # And a call without a callback works the same (no progress token is sent).
    mock_api.get("/tickets").respond(200, json=SEARCH)
    assert (await provider.call("search_tickets", {"query": "vpn"})).is_error is False
    assert first == ["Retrying (attempt 2 of 3)"]
    assert second == ["Retrying (attempt 2 of 3)"]


async def test_a_failing_progress_callback_does_not_fail_the_call(
    provider: McpToolProvider, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json=SEARCH),
        ]
    )

    async def broken(message: str) -> None:
        raise RuntimeError("listener gone")

    result = await provider.call("search_tickets", {"query": "vpn"}, on_progress=broken)
    assert result.is_error is False


# --- call: transport failures --------------------------------------------------------------


def _failing_factory(handler: Callable[[httpx.Request], httpx.Response]) -> ClientFactory:
    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=headers)

    return factory


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, text="Internal Server Error")


@pytest.mark.parametrize("handler", [_refuse, _server_error], ids=["refused", "http-500"])
async def test_unreachable_server_is_a_transport_error(
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    broken = McpToolProvider(MCP_URL, timeout_s=5.0, httpx_client_factory=_failing_factory(handler))
    with pytest.raises(ToolTransportError):
        await broken.call("search_tickets", {"query": "vpn"})
    with pytest.raises(ToolTransportError):
        await broken.list_tools()


async def test_transport_error_message_holds_no_arguments() -> None:
    broken = McpToolProvider(MCP_URL, timeout_s=5.0, httpx_client_factory=_failing_factory(_refuse))
    with pytest.raises(ToolTransportError) as caught:
        await broken.call("add_comment", {"ticket_id": "INC-0003", "body": "secret-ish text"})
    assert "secret-ish text" not in str(caught.value)
    assert "add_comment" in str(caught.value)


async def test_a_server_that_never_answers_is_cut_off_by_the_client_timeout(
    mcp_app: Starlette, mock_api: respx.Router
) -> None:
    never = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        await never.wait()
        return httpx.Response(200, json=SEARCH)

    mock_api.get("/tickets").mock(side_effect=hang)
    impatient = McpToolProvider(MCP_URL, timeout_s=0.3, httpx_client_factory=asgi_clients(mcp_app))
    async with asyncio.timeout(10):
        with pytest.raises(ToolTransportError):
            await impatient.call("search_tickets", {"query": "vpn"})


# --- timeouts and trace context ------------------------------------------------------------


def test_client_timeout_outlasts_the_servers_deadline() -> None:
    settings = Settings(_env_file=None, tool_deadline_s=15.0, tool_attempt_timeout_s=5.0)
    assert call_timeout_s(settings) == 20.0
    assert call_timeout_s(settings) > settings.tool_deadline_s


async def test_each_call_sends_its_requests_in_the_callers_trace_context(
    mcp_app: Starlette, mock_api: respx.Router
) -> None:
    """A fresh session per call: the tools/call request is sent while the caller's span is
    current (what httpx instrumentation reads for `traceparent`), not a session opener's."""
    mock_api.get("/tickets").respond(200, json=SEARCH)
    seen: list[int] = []

    async def record(request: httpx.Request) -> None:
        if b'"tools/call"' in request.content:
            seen.append(trace.get_current_span().get_span_context().trace_id)

    created: list[httpx.AsyncClient] = []
    traced = McpToolProvider(
        MCP_URL,
        timeout_s=20.0,
        httpx_client_factory=asgi_clients(mcp_app, created, event_hooks={"request": [record]}),
    )
    tracer = TracerProvider().get_tracer("tests.caller")
    expected: list[int] = []
    for name in ("first", "second"):
        with tracer.start_as_current_span(name) as span:
            expected.append(span.get_span_context().trace_id)
            await traced.call("search_tickets", {"query": "vpn"})
    assert seen == expected
    assert expected[0] != expected[1]
    assert len(created) == 2  # one session (and client) per call
    assert all(client.is_closed for client in created)


# --- call: no request after the tool result ------------------------------------------------


def _method(request: httpx.Request) -> str | None:
    if request.method != "POST" or not request.content:
        return None
    message = json.loads(request.content)
    return message.get("method") if isinstance(message, dict) else None


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """The ASGI app, except for the JSON-RPC methods `override` answers itself."""

    def __init__(
        self,
        app: Starlette,
        override: Callable[[httpx.Request, str | None], httpx.Response | None],
        seen: list[str | None],
    ) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._override = override
        self._seen = seen

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        method = _method(request)
        self._seen.append(method)
        response = self._override(request, method)
        if response is not None:
            return response
        return await self._inner.handle_async_request(request)


def _scripted_clients(
    app: Starlette,
    override: Callable[[httpx.Request, str | None], httpx.Response | None],
    seen: list[str | None],
) -> ClientFactory:
    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=_ScriptedTransport(app, override, seen), headers=headers, timeout=timeout
        )

    return factory


WRITE_ARGS: dict[str, Any] = {
    "title": "VPN keeps dropping",
    "description": "The VPN disconnects every few minutes.",
    "category": "network",
    "priority": "high",
    "reporter_id": "alice",
    "idempotency_key": KEY,
}


async def test_a_committed_write_is_not_lost_to_a_failing_follow_up_request(
    mcp_app: Starlette, mock_api: respx.Router
) -> None:
    """Nothing is sent after tools/call answers, so no later failure can turn a saved
    write into an unconfirmed one."""
    writes = mock_api.post("/tickets").respond(201, json=TICKET)
    seen: list[str | None] = []

    def refuse_tools_list(request: httpx.Request, method: str | None) -> httpx.Response | None:
        if method == "tools/list":
            raise httpx.ConnectError("connection reset", request=request)
        return None

    provider = McpToolProvider(
        MCP_URL,
        timeout_s=20.0,
        httpx_client_factory=_scripted_clients(mcp_app, refuse_tools_list, seen),
    )
    result = await provider.call("create_ticket", WRITE_ARGS)
    assert result.is_error is False
    assert json.loads(result.text)["id"] == "INC-0003"
    assert writes.call_count == 1


async def test_a_call_sends_no_tools_list(mcp_app: Starlette, mock_api: respx.Router) -> None:
    mock_api.get("/tickets").respond(200, json=SEARCH)
    seen: list[str | None] = []
    provider = McpToolProvider(
        MCP_URL,
        timeout_s=20.0,
        httpx_client_factory=_scripted_clients(mcp_app, lambda *_: None, seen),
    )
    assert (await provider.call("search_tickets", {"query": "vpn"})).is_error is False
    assert "tools/call" in seen
    assert "tools/list" not in seen


# --- timeouts on every request of a session ------------------------------------------------


class _NeverEnds(httpx.AsyncByteStream):
    """An SSE body that never sends an event: httpx's read timeout does not end it, since
    the mcp transport swallows read errors on an SSE stream."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        return None


def _stall_on(stalled: str) -> Callable[[httpx.Request, str | None], httpx.Response | None]:
    def override(request: httpx.Request, method: str | None) -> httpx.Response | None:
        if method == stalled:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_NeverEnds()
            )
        return None

    return override


@pytest.mark.parametrize("stalled", ["initialize", "tools/list"])
async def test_a_stalled_request_is_cut_off_by_the_client_timeout(
    mcp_app: Starlette, mock_api: respx.Router, stalled: str
) -> None:
    mock_api.get("/tickets").respond(200, json=SEARCH)
    seen: list[str | None] = []
    impatient = McpToolProvider(
        MCP_URL,
        timeout_s=0.3,
        httpx_client_factory=_scripted_clients(mcp_app, _stall_on(stalled), seen),
    )
    async with asyncio.timeout(5):
        with pytest.raises(ToolTransportError):
            await impatient.list_tools()
        if stalled == "initialize":
            with pytest.raises(ToolTransportError):
                await impatient.call("search_tickets", {"query": "vpn"})
