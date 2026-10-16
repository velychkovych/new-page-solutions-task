"""The `mcp.tool` span: one per call, with the CONTRACTS §6 attributes."""

import json

import httpx
import respx
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from tests.unit.mcp_server.conftest import SessionFactory, make_settings

from helpdesk.shared import trace_attributes as attr

TICKET_JSON = {
    "id": "INC-0003",
    "title": "VPN keeps dropping",
    "description": "x" * 3000,
    "status": "open",
    "priority": "high",
    "category": "network",
    "reporter_id": "alice",
    "assignee_team": None,
    "created_at": "2026-10-01T09:00:00Z",
    "updated_at": "2026-10-01T09:00:00Z",
    "comments": [],
}


def only_tool_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = [s for s in exporter.get_finished_spans() if s.name == attr.SPAN_TOOL]
    assert len(spans) == 1
    return spans[0]


async def test_successful_call_span(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    mock_api.get("/tickets/INC-0003").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json=TICKET_JSON),
        ]
    )
    async with mcp_session() as session:
        await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    span = only_tool_span(span_exporter)
    attributes = dict(span.attributes or {})
    assert attributes[attr.TOOL_NAME] == "get_ticket"
    assert json.loads(str(attributes[attr.TOOL_ARGS])) == {"ticket_id": "INC-0003"}
    assert attributes[attr.TOOL_ATTEMPTS] == 2
    assert attr.TOOL_ERROR_CODE not in attributes
    assert span.status.status_code is not StatusCode.ERROR
    retry_events = [e for e in span.events if e.name == attr.EVENT_RETRY]
    assert [dict(e.attributes or {})[attr.RETRY_ATTEMPT] for e in retry_events] == [1, 2]


async def test_result_attribute_is_truncated(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    mock_api.get("/tickets/INC-0003").respond(200, json=TICKET_JSON)
    async with mcp_session(make_settings(trace_result_max_bytes=256)) as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    recorded = str(dict(only_tool_span(span_exporter).attributes or {})[attr.TOOL_RESULT])
    assert len(recorded.encode()) == 256
    full = result.content[0].text  # type: ignore[union-attr]
    assert full.startswith(recorded)  # the tool result itself is not truncated
    assert len(full) > 3000


async def test_error_call_span(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    mock_api.get("/tickets/INC-0099").respond(404, json={"error": "not_found"})
    async with mcp_session() as session:
        await session.call_tool("get_ticket", {"ticket_id": "INC-0099"})

    span = only_tool_span(span_exporter)
    attributes = dict(span.attributes or {})
    assert attributes[attr.TOOL_ERROR_CODE] == "NOT_FOUND"
    assert attributes[attr.TOOL_ATTEMPTS] == 1
    assert str(attributes[attr.TOOL_RESULT]).startswith("ERROR NOT_FOUND: ")
    assert span.status.status_code is StatusCode.ERROR


async def test_unexpected_failure_span_has_error_code(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    mock_api.get("/tickets/INC-0003").mock(side_effect=RuntimeError("RAW-UPSTREAM"))
    async with mcp_session() as session:
        await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    span = only_tool_span(span_exporter)
    attributes = dict(span.attributes or {})
    assert attributes[attr.TOOL_ERROR_CODE] == "INTERNAL"
    assert str(attributes[attr.TOOL_RESULT]).startswith("ERROR INTERNAL: ")
    assert "RAW-UPSTREAM" not in str(attributes[attr.TOOL_RESULT])
    # The exception is caught, so it is not recorded on the span with its message.
    assert not span.events
    assert span.status.status_code is StatusCode.ERROR


async def test_invalid_arguments_span_has_zero_attempts(
    mcp_session: SessionFactory, span_exporter: InMemorySpanExporter
) -> None:
    async with mcp_session() as session:
        await session.call_tool("get_ticket", {"ticket_id": "INC-7"})

    attributes = dict(only_tool_span(span_exporter).attributes or {})
    assert attributes[attr.TOOL_ERROR_CODE] == "INVALID_ARGUMENT"
    assert attributes[attr.TOOL_ATTEMPTS] == 0


async def test_injected_args_are_recorded_for_debugging(
    mcp_session: SessionFactory, mock_api: respx.Router, span_exporter: InMemorySpanExporter
) -> None:
    mock_api.post("/tickets/INC-0003/comments").respond(
        201,
        json={
            "ticket_id": "INC-0003",
            "comment": {
                "id": "c-1",
                "author_id": "alice",
                "body": "hi",
                "created_at": "2026-10-01T09:00:00Z",
            },
        },
    )
    arguments = {
        "ticket_id": "INC-0003",
        "body": "hi",
        "author_id": "alice",
        "idempotency_key": "k-1",
    }
    async with mcp_session() as session:
        await session.call_tool("add_comment", arguments)

    recorded = dict(only_tool_span(span_exporter).attributes or {})[attr.TOOL_ARGS]
    assert json.loads(str(recorded)) == arguments
