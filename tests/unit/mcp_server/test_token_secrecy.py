"""The mock token goes in the Authorization header and nowhere else (CONTRACTS §2.6)."""

import json
import logging
from typing import Any

import httpcore
import httpx
import pytest
import respx
from mcp.shared.memory import create_connected_server_and_client_session
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
from tests.unit.mcp_server.conftest import TOKEN, FakeClock, SessionFactory, make_settings

from helpdesk.mcp_server.mock_client import MockApiClient, create_mock_http_client
from helpdesk.mcp_server.redaction import REDACTED, SecretRedactor, truncate_utf8
from helpdesk.mcp_server.retry import RetryPolicy
from helpdesk.mcp_server.server import build_server
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
DEBUG_LOGGERS = ("", "httpx", "httpcore", "mcp", "helpdesk")


@pytest.fixture
def debug_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    for name in DEBUG_LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)
    return caplog


def all_span_values(exporter: InMemorySpanExporter) -> str:
    values: list[str] = []
    for span in exporter.get_finished_spans():
        values += [f"{k}={v}" for k, v in (span.attributes or {}).items()]
        for event in span.events:
            values += [f"{k}={v}" for k, v in (event.attributes or {}).items()]
    return "\n".join(values)


async def test_token_absent_from_logs_spans_and_results(
    mcp_session: SessionFactory,
    mock_api: respx.Router,
    span_exporter: InMemorySpanExporter,
    debug_logs: pytest.LogCaptureFixture,
) -> None:
    route = mock_api.get("/tickets/INC-0003").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=TICKET)]
    )
    mock_api.get("/tickets").respond(401, json={"error": "unauthorized"})
    async with mcp_session() as session:
        ok = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
        failed = await session.call_tool("search_tickets", {"query": "vpn"})

    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert debug_logs.records, "expected DEBUG logs to be captured"
    assert TOKEN not in debug_logs.text
    assert TOKEN not in all_span_values(span_exporter)
    for result in (ok, failed):
        assert TOKEN not in result.model_dump_json()


async def test_token_echoed_by_upstream_is_redacted(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    """Backstop: even a buggy upstream that echoes the token cannot leak it to the agent."""
    leaky = {**TICKET, "description": f"Authorization: Bearer {TOKEN}"}
    mock_api.get("/tickets/INC-0003").respond(200, json=leaky)
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    text = result.content[0].text  # type: ignore[union-attr]
    assert TOKEN not in text
    assert json.loads(text)["description"] == f"Authorization: Bearer {REDACTED}"
    assert TOKEN not in all_span_values(span_exporter)


class RecordingStream(httpcore.AsyncMockStream):
    """httpcore's in-memory stream that also keeps the bytes the client wrote."""

    def __init__(self, buffer: list[bytes], written: list[bytes]) -> None:
        super().__init__(buffer)
        self._written = written

    # The signature is httpcore's network-stream interface.
    async def write(self, buffer: bytes, timeout: float | None = None) -> None:  # noqa: ASYNC109
        self._written.append(buffer)


class RecordingBackend(httpcore.AsyncMockBackend):
    """No socket: every connection reads the scripted response from memory."""

    def __init__(self, response: bytes) -> None:
        super().__init__([response])
        self._response = response
        self.written: list[bytes] = []

    async def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        return RecordingStream([self._response], self.written)


def in_memory_httpcore_transport(backend: RecordingBackend) -> httpx.AsyncHTTPTransport:
    """httpx's real transport and httpcore's real HTTP/1.1 connection code (and its DEBUG
    logging), over an in-memory network backend instead of a socket. httpx has no public
    hook for the backend, so the pool is swapped in directly."""
    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=backend)
    return transport


async def test_token_absent_from_httpcore_debug_logs(
    debug_logs: pytest.LogCaptureFixture, tracer: Tracer, clock: FakeClock
) -> None:
    body = json.dumps(TICKET).encode()
    backend = RecordingBackend(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    settings = make_settings()
    transport = in_memory_httpcore_transport(backend)
    async with create_mock_http_client(settings, TOKEN, transport=transport) as http:
        client = MockApiClient(http, RetryPolicy.from_settings(settings), sleep=clock.sleep)
        server = build_server(settings, client, redactor=SecretRedactor([TOKEN]), tracer=tracer)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    assert result.isError is False
    assert f"Authorization: Bearer {TOKEN}".encode() in b"".join(backend.written)
    assert any(r.name.startswith("httpcore") for r in debug_logs.records)
    assert TOKEN not in debug_logs.text


def test_http_client_repr_masks_token() -> None:
    http = create_mock_http_client(make_settings(), TOKEN)
    assert TOKEN not in repr(http.headers)


def test_settings_repr_masks_token() -> None:
    settings: Settings = make_settings()
    assert TOKEN not in repr(settings)


def test_redactor_replaces_every_secret() -> None:
    redact = SecretRedactor(["abc", "abcdef", ""])
    assert redact("x abcdef y abc z") == f"x {REDACTED} y {REDACTED} z"


def test_truncate_utf8_never_splits_a_character() -> None:
    assert truncate_utf8("héllo", 2) == "h"
    assert truncate_utf8("héllo", 3) == "hé"
    assert truncate_utf8("short", 100) == "short"
