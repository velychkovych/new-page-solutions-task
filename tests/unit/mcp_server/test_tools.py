"""Tool behaviour through the real FastMCP request path (in-memory MCP session).

respx stubs the mock API; the retry layer runs with a fake clock and sleep.
"""

import json
from typing import Any

import httpx
import pytest
import respx
from mcp.types import CallToolResult, TextContent
from tests.unit.mcp_server.conftest import TOKEN, FakeClock, SessionFactory

from helpdesk.mcp_server.errors import WRITE_NOT_CONFIRMED_NEXT_STEP
from helpdesk.shared.errors import ErrorCode, format_tool_error, parse_tool_error
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


def assert_error(result: CallToolResult, expected: str) -> None:
    assert result.isError is True
    assert text_of(result) == expected
    assert parse_tool_error(expected) is not None


def unavailable(write: bool, attempts: int = 3) -> str:
    return format_tool_error(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        f"The ticketing system is unavailable after {attempts} attempt(s)",
        WRITE_NOT_CONFIRMED_NEXT_STEP if write else None,
    )


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
    assert_error(result, unavailable(write=False))
    assert route.call_count == 3


async def test_exhausted_write_says_not_confirmed(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.post("/tickets").respond(500, json={"error": "upstream_error"})
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert_error(result, unavailable(write=True))


async def test_deadline_reports_timeout(
    mcp_session: SessionFactory, mock_api: respx.Router, clock: FakeClock
) -> None:
    def slow_failure(request: httpx.Request) -> httpx.Response:
        clock.now += 16.0  # the mock took longer than the whole deadline
        return httpx.Response(503, json={"error": "upstream_error"})

    route = mock_api.get("/tickets").mock(side_effect=slow_failure)
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert_error(
        result,
        format_tool_error(ErrorCode.TIMEOUT, "The ticketing system did not answer within 15 s"),
    )
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


# --- error mapping ---------------------------------------------------------------------


async def test_404_is_not_found_without_retry(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets/INC-0099").respond(404, json={"error": "not_found"})
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0099"})
    assert_error(result, format_tool_error(ErrorCode.NOT_FOUND, "Ticket INC-0099 does not exist"))
    assert route.call_count == 1


async def test_comment_on_missing_ticket_is_not_found(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.post("/tickets/INC-0003/comments").respond(404, json={"error": "not_found"})
    async with mcp_session() as session:
        result = await session.call_tool("add_comment", COMMENT_ARGS)
    assert_error(result, format_tool_error(ErrorCode.NOT_FOUND, "Ticket INC-0003 does not exist"))


@pytest.mark.parametrize(
    ("status", "body"),
    [(401, {"error": "unauthorized"}), (403, {"error": "forbidden"})],
)
async def test_401_403_are_auth(
    status: int, body: dict[str, str], mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(status, json=body)
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    expected = format_tool_error(
        ErrorCode.AUTH,
        f"The ticketing system rejected the MCP server's credentials (HTTP {status})",
    )
    assert_error(result, expected)
    assert "administrator" in expected
    assert route.call_count == 1


async def test_409_is_conflict(mcp_session: SessionFactory, mock_api: respx.Router) -> None:
    route = mock_api.post("/tickets").respond(409, json={"error": "idempotency_conflict"})
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert_error(
        result,
        format_tool_error(
            ErrorCode.CONFLICT, "This idempotency key was already used for a different change"
        ),
    )
    assert route.call_count == 1


async def test_400_invalid_cursor_is_invalid_argument(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets").respond(400, json={"error": "invalid_cursor"})
    async with mcp_session() as session:
        result = await session.call_tool(
            "search_tickets", {"query": "printer", "cursor": "b2Zmc2V0OjU"}
        )
    assert_error(
        result,
        format_tool_error(
            ErrorCode.INVALID_ARGUMENT,
            "The cursor is not valid for this query and status filter",
            "Search again without a cursor, or with the query and status the cursor came from",
        ),
    )


@pytest.mark.parametrize(
    ("status", "body", "what"),
    [
        (400, {"error": "weird"}, "The ticketing system rejected the request (HTTP 400)"),
        (400, b"not json", "The ticketing system rejected the request (HTTP 400)"),
        (
            422,
            {"error": "invalid_request", "detail": "title: RAW-UPSTREAM-DETAIL"},
            "The ticketing system rejected the arguments as invalid (HTTP 422)",
        ),
    ],
)
async def test_400_422_are_invalid_argument_without_raw_body(
    status: int,
    body: dict[str, str] | bytes,
    what: str,
    mcp_session: SessionFactory,
    mock_api: respx.Router,
) -> None:
    response = (
        httpx.Response(status, content=body)
        if isinstance(body, bytes)
        else httpx.Response(status, json=body)
    )
    route = mock_api.get("/tickets").mock(return_value=response)
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert_error(result, format_tool_error(ErrorCode.INVALID_ARGUMENT, what))
    assert "RAW-UPSTREAM-DETAIL" not in text_of(result)
    assert route.call_count == 1


async def test_bad_idempotency_key_rejection_is_invalid_argument(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.post("/tickets").respond(400, json={"error": "invalid_idempotency_key"})
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)
    assert_error(
        result,
        format_tool_error(
            ErrorCode.INVALID_ARGUMENT,
            "The ticketing system rejected the idempotency key",
            "Do not retry; tell the user the change was not applied",
        ),
    )


async def test_unexpected_status_is_unavailable_without_retry(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = mock_api.get("/tickets").respond(501, text="Not Implemented")
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn"})
    assert_error(
        result,
        format_tool_error(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "The ticketing system returned an unexpected response (HTTP 501)",
        ),
    )
    assert route.call_count == 1


async def test_malformed_success_body_is_unavailable_and_not_echoed(
    mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    mock_api.get("/tickets/INC-0003").respond(200, json={"id": "RAW-UPSTREAM-BODY"})
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert_error(
        result,
        format_tool_error(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "The ticketing system returned a malformed response (HTTP 200)",
        ),
    )
    assert "RAW-UPSTREAM-BODY" not in text_of(result)


@pytest.mark.parametrize(
    "error",
    [httpx.DecodingError("bad gzip RAW-UPSTREAM-BYTES"), httpx.TooManyRedirects("RAW-UPSTREAM")],
    ids=["decoding-error", "too-many-redirects"],
)
@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
async def test_other_http_errors_are_unavailable_without_retry(
    error: Exception, write: bool, mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    route = (mock_api.post("/tickets") if write else mock_api.get("/tickets/INC-0003")).mock(
        side_effect=error
    )
    async with mcp_session() as session:
        if write:
            result = await session.call_tool("create_ticket", CREATE_ARGS)
        else:
            result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert_error(result, unavailable(write=write, attempts=1))
    assert "RAW-UPSTREAM" not in text_of(result)
    assert route.call_count == 1


@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
async def test_unexpected_exception_still_gives_contract_text(
    write: bool, mcp_session: SessionFactory, mock_api: respx.Router
) -> None:
    """Without a fallback, FastMCP's low-level server would return str(exc) as the text."""
    boom = RuntimeError(f"RAW-UPSTREAM-BODY Bearer {TOKEN}")
    (mock_api.post("/tickets") if write else mock_api.get("/tickets/INC-0003")).mock(
        side_effect=boom
    )
    async with mcp_session() as session:
        if write:
            result = await session.call_tool("create_ticket", CREATE_ARGS)
        else:
            result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})
    assert_error(
        result,
        format_tool_error(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "The tool failed with an unexpected error",
            WRITE_NOT_CONFIRMED_NEXT_STEP if write else None,
        ),
    )
    assert "RAW-UPSTREAM" not in text_of(result)
    assert TOKEN not in text_of(result)


# --- argument validation through FastMCP -----------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "issue"),
    [
        ("get_ticket", {"ticket_id": "INC-7"}, "ticket_id: String should match pattern"),
        ("get_ticket", {"ticket_id": "INC-٠٠٠٣"}, "ticket_id: String should match pattern"),
        (
            "search_tickets",
            {"query": "vpn", "page_size": 50},
            "page_size: Input should be less than or equal to 20",
        ),
        (
            "search_tickets",
            {"query": "vpn", "page_size": 0},
            "page_size: Input should be greater than or equal to 1",
        ),
        ("search_tickets", {"query": "   "}, "query: String should have at least 1 character"),
        ("search_tickets", {"query": "vpn", "status": ["pending"]}, "status.0: Input should be"),
        ("search_tickets", {}, "query: Field required"),
        (
            "create_ticket",
            {k: v for k, v in CREATE_ARGS.items() if k != "idempotency_key"},
            "idempotency_key: Field required",
        ),
        (
            "create_ticket",
            {**CREATE_ARGS, "reporter_id": "bob:admin"},
            "reporter_id: String should match pattern",
        ),
        (
            "add_comment",
            {**COMMENT_ARGS, "extra": 1},
            "extra: Extra inputs are not permitted",
        ),
    ],
)
async def test_invalid_arguments_give_exact_invalid_argument_text(
    tool: str,
    arguments: dict[str, Any],
    issue: str,
    mcp_session: SessionFactory,
    mock_api: respx.Router,
) -> None:
    async with mcp_session() as session:
        result = await session.call_tool(tool, arguments)
    assert result.isError is True
    text = text_of(result)
    assert text.startswith(f"ERROR INVALID_ARGUMENT: Invalid arguments for {tool}: {issue}")
    assert text.endswith(". Next step: Fix the arguments and call the tool again.")
    assert "Error executing tool" not in text
    parsed = parse_tool_error(text)
    assert parsed is not None
    assert parsed.code is ErrorCode.INVALID_ARGUMENT
    assert mock_api.calls.call_count == 0  # the mock is never called with bad arguments


async def test_invalid_ticket_id_exact_text(mcp_session: SessionFactory) -> None:
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-7"})
    assert_error(
        result,
        "ERROR INVALID_ARGUMENT: Invalid arguments for get_ticket: ticket_id: String should "
        "match pattern '^INC-[0-9]{4}$'. Next step: Fix the arguments and call the tool again.",
    )


async def test_invalid_page_size_exact_text(mcp_session: SessionFactory) -> None:
    async with mcp_session() as session:
        result = await session.call_tool("search_tickets", {"query": "vpn", "page_size": 50})
    assert_error(
        result,
        "ERROR INVALID_ARGUMENT: Invalid arguments for search_tickets: page_size: Input should "
        "be less than or equal to 20. Next step: Fix the arguments and call the tool again.",
    )


async def test_many_validation_issues_are_summarised(mcp_session: SessionFactory) -> None:
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", {})
    text = text_of(result)
    assert "; and 1 more. Next step:" in text  # 6 required fields, 5 shown
    assert parse_tool_error(text) is not None
