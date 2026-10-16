"""Tool behaviour through the real FastMCP request path (in-memory MCP session).

respx stubs the mock API; the retry layer runs with a fake clock and sleep. The error
texts themselves are pinned in test_tool_errors.py.
"""

import json
from typing import Any

import httpx
import pytest
import respx
from mcp.types import CallToolResult, TextContent
from tests.unit.mcp_server.conftest import TOKEN, FakeClock, SessionFactory

from helpdesk.shared.errors import ErrorCode, parse_tool_error
from helpdesk.shared.tickets import (
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAY_HEADER,
    CommentCreated,
    SearchResult,
    Ticket,
)

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
        {
            "id": "INC-0003",
            "title": "VPN keeps dropping",
            "status": "open",
            "priority": "high",
            "category": "network",
            "updated_at": NOW,
        }
    ],
    "next_cursor": "b2Zmc2V0OjU",
}
COMMENT_CREATED: dict[str, Any] = {
    "ticket_id": "INC-0003",
    "comment": {"id": "c-1", "author_id": "alice", "body": "Still dropping.", "created_at": NOW},
}
CREATE_ARGS: dict[str, Any] = {
    "title": "Printer on floor 3 jams",
    "description": "Paper jams on every print job.",
    "category": "hardware",
    "priority": "medium",
    "reporter_id": "bob",
    "idempotency_key": "6f1c2a9e-0d4b-4c51-9a77-1f0b8e3c2d10",
}
COMMENT_ARGS: dict[str, Any] = {
    "ticket_id": "INC-0003",
    "body": "Still dropping.",
    "author_id": "alice",
    "idempotency_key": "0e7d5a4c-3b2a-4f19-8e6d-5c4b3a291807",
}


def text_of(result: CallToolResult) -> str:
    assert len(result.content) == 1
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


def error_code(result: CallToolResult) -> ErrorCode:
    assert result.isError is True
    parsed = parse_tool_error(text_of(result))
    assert parsed is not None
    return parsed.code


# --- success paths and request mapping ------------------------------------------------


async def test_search_maps_arguments_to_mock_query(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(200, json=SEARCH)
    async with mcp_session() as session:
        result = await session.call_tool(
            "search_tickets",
            {
                "query": "vpn drops",
                "status": ["open", "in_progress"],
                "cursor": "b2Zmc2V0OjA",
                "page_size": 7,
            },
        )
    assert result.isError is False
    request = route.calls.last.request
    assert request.url.params.multi_items() == [
        ("q", "vpn drops"),
        ("limit", "7"),
        ("status", "open"),
        ("status", "in_progress"),
        ("cursor", "b2Zmc2V0OjA"),
    ]


async def test_search_defaults_send_only_q_and_limit(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(200, json=SEARCH)
    async with mcp_session() as session:
        await session.call_tool("search_tickets", {"query": "vpn"})
    assert route.calls.last.request.url.params.multi_items() == [("q", "vpn"), ("limit", "5")]


async def test_search_result_is_compact_search_result_json(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").respond(200, json=SEARCH)
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    text = text_of(result)
    assert text == SearchResult.model_validate(SEARCH).model_dump_json()
    assert ", " not in text
    assert ": " not in text
    assert json.loads(text)["next_cursor"] == "b2Zmc2V0OjU"


async def test_get_ticket_returns_ticket_json(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets/INC-0003").respond(200, json=TICKET)
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert result.isError is False
    assert Ticket.model_validate_json(text_of(result)) == Ticket.model_validate(TICKET)


async def test_create_ticket_forwards_body_and_key(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.post("/tickets").respond(201, json=TICKET)
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert result.isError is False
    request = route.calls.last.request
    assert request.headers[IDEMPOTENCY_KEY_HEADER] == CREATE_ARGS["idempotency_key"]
    assert json.loads(request.content) == {
        "title": CREATE_ARGS["title"],
        "description": CREATE_ARGS["description"],
        "category": "hardware",
        "priority": "medium",
        "reporter_id": "bob",
    }


async def test_add_comment_forwards_body_and_key(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.post("/tickets/INC-0003/comments").respond(201, json=COMMENT_CREATED)
    async with mcp_session() as session:
        result = await session.call_tool("add_comment", COMMENT_ARGS)
    assert CommentCreated.model_validate_json(text_of(result)).comment.id == "c-1"
    request = route.calls.last.request
    assert request.headers[IDEMPOTENCY_KEY_HEADER] == COMMENT_ARGS["idempotency_key"]
    assert json.loads(request.content) == {"author_id": "alice", "body": "Still dropping."}


async def test_token_is_sent_as_bearer_header(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets/INC-0003").respond(200, json=TICKET)
    async with mcp_session() as session:
        await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TOKEN}"


async def test_reads_send_no_idempotency_key(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(200, json=SEARCH)
    async with mcp_session() as session:
        await session.call_tool("search_tickets", {"query": "vpn"})
    assert IDEMPOTENCY_KEY_HEADER not in route.calls.last.request.headers


# --- retries ---------------------------------------------------------------------------


async def test_write_retry_reuses_the_same_key(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.post("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.ConnectError("refused"),
            httpx.Response(201, json=TICKET),
        ]
    )
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert result.isError is False
    keys = [call.request.headers[IDEMPOTENCY_KEY_HEADER] for call in route.calls]
    assert keys == [CREATE_ARGS["idempotency_key"]] * 3


async def test_lost_response_then_replay_succeeds(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    """Fail-after-commit: the write landed, the 503 hid it, and the retry gets the replay."""
    route = mock_api.post("/tickets/INC-0003/comments").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(201, json=COMMENT_CREATED, headers={IDEMPOTENT_REPLAY_HEADER: "true"}),
        ]
    )
    async with mcp_session() as session:
        result = await session.call_tool("add_comment", COMMENT_ARGS)
    assert result.isError is False
    assert route.call_count == 2


async def test_429_retry_after_is_honoured(
    mcp_session: SessionFactory, mock_api: respx.Router, clock: FakeClock
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "1"}),
            httpx.Response(200, json=SEARCH),
        ]
    )
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert result.isError is False
    assert clock.sleeps == [1.0]


@pytest.mark.parametrize(
    "first",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")],
    ids=["connect-error", "read-timeout"],
)
async def test_connect_error_and_read_timeout_are_retried(
    first: Exception, mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets/INC-0003").mock(
        side_effect=[first, httpx.Response(200, json=TICKET)]
    )
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert result.isError is False
    assert route.call_count == 2


async def test_retries_stop_at_3_and_report_unavailable(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(503, json={"error": "upstream_error"})
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert error_code(result) is ErrorCode.UPSTREAM_UNAVAILABLE
    assert route.call_count == 3


async def test_exhausted_write_says_not_confirmed(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.post("/tickets").respond(500, json={"error": "upstream_error"})
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert error_code(result) is ErrorCode.UPSTREAM_UNAVAILABLE
    assert "could not confirm whether the change was applied" in text_of(result)
    assert route.call_count == 3


async def test_deadline_reports_timeout(
    mcp_session: SessionFactory, mock_api: respx.Router, clock: FakeClock
) -> None:
    def slow_failure(request: httpx.Request) -> httpx.Response:
        clock.now += 16.0  # the mock took longer than the whole deadline
        return httpx.Response(503, json={"error": "upstream_error"})

    route = mock_api.get("/tickets").mock(side_effect=slow_failure)
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert error_code(result) is ErrorCode.TIMEOUT
    assert route.call_count == 1


async def test_retry_progress_is_reported_when_client_asks(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").mock(
        side_effect=[
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(503, json={"error": "upstream_error"}),
            httpx.Response(200, json=SEARCH),
        ]
    )
    progress: list[tuple[float, float | None, str | None]] = []

    async def on_progress(value: float, total: float | None, message: str | None) -> None:
        progress.append((value, total, message))

    async with mcp_session() as session:
        result = await session.call_tool(
            "search_tickets", {"query": "vpn"}, progress_callback=on_progress
        )
    assert result.isError is False
    assert progress == [
        (1, 3, "Retrying (attempt 2 of 3)"),
        (2, 3, "Retrying (attempt 3 of 3)"),
    ]
