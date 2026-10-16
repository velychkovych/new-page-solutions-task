"""Every error text the server can produce (CONTRACTS §2.4), through the real FastMCP path.

Each failure mode is one table row with its exact text, so a wording change is a reviewed
change here. Every row is also checked for the contract rules that do not depend on the
wording: one parseable code, the tool name and the key facts, the write outcome, and no
token, stack trace or upstream bytes.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx
from mcp.types import CallToolResult, TextContent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from tests.unit.mcp_server.conftest import TOKEN, FakeClock, SessionFactory

from helpdesk.mcp_server.errors import argument_rules
from helpdesk.mcp_server.mock_client import MockApiClient
from helpdesk.shared import trace_attributes as attr
from helpdesk.shared.errors import ErrorCode, ToolError, parse_tool_error
from helpdesk.shared.tickets import Ticket
from helpdesk.shared.tools import (
    INJECTED_ARGS,
    WRITE_TOOLS,
    AddCommentInput,
    CreateTicketInput,
    GetTicketArgs,
    SearchTicketsArgs,
    ToolName,
)

# Planted in upstream bodies and exception messages; must never reach an error text.
RAW = "RAW-UPSTREAM-7f3a"

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
CALLS: dict[ToolName, tuple[str, str, dict[str, Any]]] = {
    ToolName.SEARCH_TICKETS: ("GET", "/tickets", {"query": "vpn"}),
    ToolName.GET_TICKET: ("GET", "/tickets/INC-0003", {"ticket_id": "INC-0003"}),
    ToolName.CREATE_TICKET: ("POST", "/tickets", CREATE_ARGS),
    ToolName.ADD_COMMENT: ("POST", "/tickets/INC-0003/comments", COMMENT_ARGS),
}
INPUT_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.SEARCH_TICKETS: SearchTicketsArgs,
    ToolName.GET_TICKET: GetTicketArgs,
    ToolName.CREATE_TICKET: CreateTicketInput,
    ToolName.ADD_COMMENT: AddCommentInput,
}

READ_UNAVAILABLE = "Tell the user the ticketing system is unavailable and to try again later."
READ_TIMEOUT = "Tell the user the ticketing system timed out and to try again later."
READ_INTERNAL = (
    "Do not retry this call; tell the user the request failed because of an internal error"
    " and to contact an administrator."
)
WRITE_UNCONFIRMED = (
    "Tell the user the change could not be confirmed and may or may not have been applied;"
    " do not make up a ticket id."
)
WRITE_INTERNAL = (
    "Tell the user the change could not be confirmed and to contact an administrator;"
    " do not retry and do not make up a ticket id."
)
MISCONFIGURED = "Tell the user the helpdesk is misconfigured and to contact an administrator."
REJECTED = (
    "Do not repeat this call with the same arguments; tell the user the ticketing system"
    " rejected the request."
)
NOT_APPLIED_ADMIN = (
    "Do not retry; tell the user the change was not applied and to contact an administrator."
)
INTERNAL_CAUSE = (
    "an unexpected internal error occurred in the MCP server (a bug, not caused by the"
    " arguments or the ticketing system)"
)

type Outcome = Callable[[httpx.Request, FakeClock], httpx.Response]


def status(code: int, body: dict[str, Any] | bytes | None = None) -> Outcome:
    def respond(request: httpx.Request, clock: FakeClock) -> httpx.Response:
        if isinstance(body, bytes):
            return httpx.Response(code, content=body)
        return httpx.Response(code, json=body if body is not None else {"error": "x"})

    return respond


def raises(make: Callable[[], Exception]) -> Outcome:
    def fail(request: httpx.Request, clock: FakeClock) -> httpx.Response:
        raise make()

    return fail


def slower_than_the_deadline(request: httpx.Request, clock: FakeClock) -> httpx.Response:
    clock.now += 16.0
    return httpx.Response(503, json={"error": "upstream_error"})


@dataclass(frozen=True)
class Case:
    id: str
    tool: ToolName
    outcome: Outcome
    code: ErrorCode
    text: str
    calls: int = 1


def upstream_cases() -> list[Case]:
    s, g, c, a = (
        ToolName.SEARCH_TICKETS,
        ToolName.GET_TICKET,
        ToolName.CREATE_TICKET,
        ToolName.ADD_COMMENT,
    )
    unknown = "could not confirm whether the change was applied"
    not_applied = "failed and the change was not applied"
    unavailable = ErrorCode.UPSTREAM_UNAVAILABLE
    invalid = ErrorCode.INVALID_ARGUMENT
    return [
        Case(
            "get-404",
            g,
            status(404, {"error": "not_found"}),
            ErrorCode.NOT_FOUND,
            "ERROR NOT_FOUND: get_ticket failed: ticket INC-0003 does not exist. Next step: "
            "Tell the user ticket INC-0003 does not exist and ask them to check the id.",
        ),
        Case(
            "comment-404",
            a,
            status(404, {"error": "not_found"}),
            ErrorCode.NOT_FOUND,
            f"ERROR NOT_FOUND: add_comment {not_applied}: ticket INC-0003 does not exist. "
            "Next step: Tell the user ticket INC-0003 does not exist and ask them to check "
            "the id.",
        ),
        Case(
            "search-404-misrouted",
            s,
            status(404, {"error": "not_found"}),
            ErrorCode.NOT_FOUND,
            "ERROR NOT_FOUND: search_tickets failed: the ticketing system answered HTTP 404 for "
            "this request, so the MCP server is misconfigured. Next step: " + MISCONFIGURED,
        ),
        Case(
            "search-401",
            s,
            status(401, {"error": "unauthorized"}),
            ErrorCode.AUTH,
            "ERROR AUTH: search_tickets failed: the ticketing system rejected the MCP server's "
            "credentials (HTTP 401), so the MCP server is misconfigured. Next step: "
            + MISCONFIGURED,
        ),
        Case(
            "create-403",
            c,
            status(403, {"error": "forbidden"}),
            ErrorCode.AUTH,
            f"ERROR AUTH: create_ticket {not_applied}: the ticketing system rejected the MCP "
            "server's credentials (HTTP 403), so the MCP server is misconfigured. Next step: "
            + MISCONFIGURED,
        ),
        Case(
            "create-409",
            c,
            status(409, {"error": "idempotency_conflict"}),
            ErrorCode.CONFLICT,
            f"ERROR CONFLICT: create_ticket {not_applied}: its idempotency key was already "
            "used for a different change (HTTP 409). Next step: Do not retry; tell the user "
            "the change was not applied.",
        ),
        Case(
            "search-400-invalid-cursor",
            s,
            status(400, {"error": "invalid_cursor"}),
            invalid,
            "ERROR INVALID_ARGUMENT: search_tickets failed: the cursor does not belong to this "
            "search: it is malformed, or it came from a search with a different query or "
            "status (HTTP 400). Next step: Call search_tickets again with the same query and "
            "status and no cursor.",
        ),
        Case(
            "create-400-invalid-key",
            c,
            status(400, {"error": "invalid_idempotency_key"}),
            invalid,
            f"ERROR INVALID_ARGUMENT: create_ticket {not_applied}: the ticketing system "
            "rejected the idempotency key (invalid_idempotency_key, HTTP 400). Next step: "
            + NOT_APPLIED_ADMIN,
        ),
        Case(
            "comment-400-missing-key",
            a,
            status(400, {"error": "missing_idempotency_key"}),
            invalid,
            f"ERROR INVALID_ARGUMENT: add_comment {not_applied}: the ticketing system "
            "rejected the idempotency key (missing_idempotency_key, HTTP 400). Next step: "
            + NOT_APPLIED_ADMIN,
        ),
        Case(
            "search-400-unknown-code",
            s,
            status(400, {"error": f"{RAW} {TOKEN}"}),
            invalid,
            "ERROR INVALID_ARGUMENT: search_tickets failed: the ticketing system rejected the "
            "request as invalid (HTTP 400). Next step: " + REJECTED,
        ),
        Case(
            "get-400-not-json",
            g,
            status(400, f"{RAW} not json".encode()),
            invalid,
            "ERROR INVALID_ARGUMENT: get_ticket failed: the ticketing system rejected the "
            "request as invalid (HTTP 400). Next step: " + REJECTED,
        ),
        Case(
            "search-422",
            s,
            status(422, {"error": "invalid_request", "detail": f"q: {RAW} {TOKEN}"}),
            invalid,
            "ERROR INVALID_ARGUMENT: search_tickets failed: the ticketing system rejected the "
            "arguments as invalid (HTTP 422). Next step: " + REJECTED,
        ),
        Case(
            "create-422",
            c,
            status(422, {"error": "invalid_request", "detail": RAW}),
            invalid,
            f"ERROR INVALID_ARGUMENT: create_ticket {not_applied}: the ticketing system "
            "rejected the arguments as invalid (HTTP 422). Next step: " + NOT_APPLIED_ADMIN,
        ),
        Case(
            "search-503-exhausted",
            s,
            status(503, {"error": "upstream_error"}),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: search_tickets failed: the ticketing system is "
            "unavailable (3 attempts; the last one got HTTP 503). Next step: " + READ_UNAVAILABLE,
            calls=3,
        ),
        Case(
            "get-429-exhausted",
            g,
            status(429, {"error": "rate_limited"}),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: get_ticket failed: the ticketing system is "
            "unavailable (3 attempts; the last one got HTTP 429). Next step: " + READ_UNAVAILABLE,
            calls=3,
        ),
        Case(
            "get-connect-error-exhausted",
            g,
            raises(lambda: httpx.ConnectError(f"{RAW} refused")),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: get_ticket failed: the ticketing system is "
            "unavailable (3 attempts; the last one could not connect). Next step: "
            + READ_UNAVAILABLE,
            calls=3,
        ),
        Case(
            "search-read-timeout-exhausted",
            s,
            raises(lambda: httpx.ReadTimeout(f"{RAW} slow")),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: search_tickets failed: the ticketing system is "
            "unavailable (3 attempts; the last one timed out). Next step: " + READ_UNAVAILABLE,
            calls=3,
        ),
        Case(
            "create-500-exhausted",
            c,
            status(500, {"error": "upstream_error"}),
            unavailable,
            f"ERROR UPSTREAM_UNAVAILABLE: create_ticket {unknown}: the ticketing system is "
            "unavailable (3 attempts; the last one got HTTP 500). Next step: " + WRITE_UNCONFIRMED,
            calls=3,
        ),
        Case(
            "comment-connect-error-exhausted",
            a,
            raises(lambda: httpx.ConnectError(f"{RAW} refused")),
            unavailable,
            f"ERROR UPSTREAM_UNAVAILABLE: add_comment {unknown}: the ticketing system is "
            "unavailable (3 attempts; the last one could not connect). Next step: "
            + WRITE_UNCONFIRMED,
            calls=3,
        ),
        Case(
            "search-501-unexpected",
            s,
            status(501, f"{RAW} Not Implemented".encode()),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: search_tickets failed: the ticketing system sent an "
            "unexpected HTTP 501 response (1 attempt). Next step: " + READ_UNAVAILABLE,
        ),
        Case(
            "comment-405-unexpected",
            a,
            status(405, {"error": "method_not_allowed"}),
            unavailable,
            f"ERROR UPSTREAM_UNAVAILABLE: add_comment {unknown}: the ticketing system sent an "
            "unexpected HTTP 405 response (1 attempt). Next step: " + WRITE_UNCONFIRMED,
        ),
        Case(
            "get-200-malformed",
            g,
            status(200, {"id": RAW}),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: get_ticket failed: the ticketing system sent an HTTP "
            "200 response in an unexpected format (1 attempt). Next step: " + READ_UNAVAILABLE,
        ),
        Case(
            "create-201-malformed",
            c,
            status(201, {"id": RAW}),
            unavailable,
            f"ERROR UPSTREAM_UNAVAILABLE: create_ticket {unknown}: the ticketing system sent "
            "an HTTP 201 response in an unexpected format (1 attempt). Next step: "
            + WRITE_UNCONFIRMED,
        ),
        Case(
            "get-decoding-error",
            g,
            raises(lambda: httpx.DecodingError(f"bad gzip {RAW}")),
            unavailable,
            "ERROR UPSTREAM_UNAVAILABLE: get_ticket failed: the ticketing system is "
            "unavailable (1 attempt, which failed with DecodingError). Next step: "
            + READ_UNAVAILABLE,
        ),
        Case(
            "create-too-many-redirects",
            c,
            raises(lambda: httpx.TooManyRedirects(RAW)),
            unavailable,
            f"ERROR UPSTREAM_UNAVAILABLE: create_ticket {unknown}: the ticketing system is "
            "unavailable (1 attempt, which failed with TooManyRedirects). Next step: "
            + WRITE_UNCONFIRMED,
        ),
        Case(
            "search-deadline",
            s,
            slower_than_the_deadline,
            ErrorCode.TIMEOUT,
            "ERROR TIMEOUT: search_tickets failed: the ticketing system did not answer within "
            "the 15 s deadline (1 attempt). Next step: " + READ_TIMEOUT,
        ),
        Case(
            "comment-deadline",
            a,
            slower_than_the_deadline,
            ErrorCode.TIMEOUT,
            f"ERROR TIMEOUT: add_comment {unknown}: the ticketing system did not answer "
            "within the 15 s deadline (1 attempt). Next step: " + WRITE_UNCONFIRMED,
        ),
        Case(
            "get-unexpected-exception",
            g,
            raises(lambda: RuntimeError(f"{RAW} Bearer {TOKEN}")),
            ErrorCode.INTERNAL,
            f"ERROR INTERNAL: get_ticket failed: {INTERNAL_CAUSE}. Next step: {READ_INTERNAL}",
        ),
        Case(
            "create-unexpected-exception",
            c,
            raises(lambda: KeyError(f"{RAW} Bearer {TOKEN}")),
            ErrorCode.INTERNAL,
            f"ERROR INTERNAL: create_ticket {unknown}: {INTERNAL_CAUSE}. Next step: "
            + WRITE_INTERNAL,
        ),
    ]


UPSTREAM_CASES = upstream_cases()


def text_of(result: CallToolResult) -> str:
    assert result.isError is True
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


def check_contract_rules(tool: ToolName, text: str) -> ToolError:
    """The §2.4 rules every error text must meet, whatever its wording."""
    parsed = parse_tool_error(text)
    assert parsed is not None, text
    assert text == text.strip()
    assert parsed.what.startswith(f"{tool.value} "), "the text names the tool"
    for leak in (TOKEN, RAW, "Traceback", 'File "', "Error executing tool"):
        assert leak not in text
    if tool in WRITE_TOOLS:
        # A write always says whether the change was applied.
        assert ("could not confirm whether the change was applied" in parsed.what) != (
            "the change was not applied" in parsed.what
        )
    return parsed


def check_key_facts(case: Case, parsed: ToolError) -> None:
    match case.code:
        case ErrorCode.NOT_FOUND if "INC-" in parsed.what:
            assert "INC-0003" in parsed.what
        case ErrorCode.UPSTREAM_UNAVAILABLE:
            assert "attempt" in parsed.what
            assert (
                "HTTP " in parsed.what
                or "could not connect" in parsed.what
                or ("timed out" in parsed.what or "failed with" in parsed.what)
            )
        case ErrorCode.TIMEOUT:
            assert "15 s" in parsed.what
        case ErrorCode.CONFLICT:
            assert "idempotency key" in parsed.what
            assert "not applied" in parsed.what
        case ErrorCode.AUTH:
            assert "misconfigured" in parsed.what
            assert "administrator" in parsed.next_step
        case ErrorCode.INTERNAL:
            assert "internal error" in parsed.what
            assert "administrator" in parsed.next_step
    unknown_outcome = case.code in {
        ErrorCode.TIMEOUT,
        ErrorCode.UPSTREAM_UNAVAILABLE,
        ErrorCode.INTERNAL,
    }
    if case.tool in WRITE_TOOLS and unknown_outcome:
        assert "could not confirm" in parsed.what
        assert "do not make up a ticket id" in parsed.next_step


@pytest.mark.parametrize("case", UPSTREAM_CASES, ids=[case.id for case in UPSTREAM_CASES])
async def test_each_failure_mode_gives_one_code_and_its_exact_text(
    case: Case,
    mcp_session: SessionFactory,
    mock_api: respx.Router,
    clock: FakeClock,
    span_exporter: InMemorySpanExporter,
) -> None:
    method, path, arguments = CALLS[case.tool]
    route = mock_api.route(method=method, path=path).mock(
        side_effect=lambda request: case.outcome(request, clock)
    )
    async with mcp_session() as session:
        result = await session.call_tool(case.tool.value, arguments)

    text = text_of(result)
    assert text == case.text
    parsed = check_contract_rules(case.tool, text)
    assert parsed.code is case.code
    check_key_facts(case, parsed)
    assert route.call_count == case.calls
    (span,) = [s for s in span_exporter.get_finished_spans() if s.name == attr.SPAN_TOOL]
    assert dict(span.attributes or {})[attr.TOOL_ERROR_CODE] == case.code.value


def test_every_failure_mode_is_covered_for_reads_and_writes() -> None:
    codes = {case.code for case in UPSTREAM_CASES}
    assert codes == set(ErrorCode)
    for code in (ErrorCode.UPSTREAM_UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.INTERNAL):
        tools = {case.tool for case in UPSTREAM_CASES if case.code is code}
        assert tools & WRITE_TOOLS, code
        assert tools - WRITE_TOOLS, code


# --- unexpected exceptions --------------------------------------------------------------


async def test_a_bug_after_the_upstream_call_is_internal(
    mcp_session: SessionFactory,
    mock_api: respx.Router,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bug in our own code, not in the transport: the handler returns the wrong type."""

    async def buggy_get_ticket(self: MockApiClient, *args: object) -> Ticket:
        return {"secret": TOKEN}  # type: ignore[return-value]

    monkeypatch.setattr(MockApiClient, "get_ticket", buggy_get_ticket)
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003"})

    text = text_of(result)
    assert (
        text == f"ERROR INTERNAL: get_ticket failed: {INTERNAL_CAUSE}. Next step: {READ_INTERNAL}"
    )
    assert mock_api.calls.call_count == 0


async def test_an_unexpected_exception_is_logged_with_its_traceback_but_no_secret(
    mcp_session: SessionFactory, mock_api: respx.Router, caplog: pytest.LogCaptureFixture
) -> None:
    mock_api.post("/tickets").mock(side_effect=RuntimeError(f"{RAW} Bearer {TOKEN}"))
    caplog.set_level(logging.ERROR, logger="helpdesk.mcp_server")
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", CREATE_ARGS)

    assert parse_tool_error(text_of(result)) is not None
    (record,) = [r for r in caplog.records if "unexpected error" in r.getMessage()]
    logged = record.getMessage()
    assert "create_ticket" in logged
    assert "Traceback" in logged
    assert "RuntimeError" in logged
    assert TOKEN not in caplog.text


# --- argument validation (the mock is never called) ------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        (
            "get_ticket",
            {"ticket_id": "INC-7"},
            "ERROR INVALID_ARGUMENT: get_ticket failed: ticket_id must be INC- followed by four "
            "digits, for example INC-0042. Next step: Fix ticket_id, then call get_ticket again.",
        ),
        (
            "get_ticket",
            {"ticket_id": "INC-٠٠٠٣"},
            "ERROR INVALID_ARGUMENT: get_ticket failed: ticket_id must be INC- followed by four "
            "digits, for example INC-0042. Next step: Fix ticket_id, then call get_ticket again.",
        ),
        (
            "get_ticket",
            {"ticket_id": None},
            "ERROR INVALID_ARGUMENT: get_ticket failed: ticket_id must be INC- followed by four "
            "digits, for example INC-0042. Next step: Fix ticket_id, then call get_ticket again.",
        ),
        (
            "search_tickets",
            {"query": "vpn", "page_size": 50},
            "ERROR INVALID_ARGUMENT: search_tickets failed: page_size must be a whole number "
            "from 1 to 20. Next step: Fix page_size, then call search_tickets again.",
        ),
        (
            "search_tickets",
            {"query": "vpn", "page_size": 0},
            "ERROR INVALID_ARGUMENT: search_tickets failed: page_size must be a whole number "
            "from 1 to 20. Next step: Fix page_size, then call search_tickets again.",
        ),
        # Strict: the advertised schema says integer, so no lax coercion.
        *(
            (
                "search_tickets",
                {"query": "vpn", "page_size": value},
                "ERROR INVALID_ARGUMENT: search_tickets failed: page_size must be a whole number "
                "from 1 to 20. Next step: Fix page_size, then call search_tickets again.",
            )
            for value in (True, "7", 7.5)
        ),
        (
            "search_tickets",
            {"query": "   "},
            "ERROR INVALID_ARGUMENT: search_tickets failed: query must be text of 1 to 200 "
            "characters. Next step: Fix query, then call search_tickets again.",
        ),
        (
            "search_tickets",
            {"query": 42},
            "ERROR INVALID_ARGUMENT: search_tickets failed: query must be text of 1 to 200 "
            "characters. Next step: Fix query, then call search_tickets again.",
        ),
        (
            "search_tickets",
            {},
            "ERROR INVALID_ARGUMENT: search_tickets failed: query is required and must be text "
            "of 1 to 200 characters. Next step: Fix query, then call search_tickets again.",
        ),
        (
            "search_tickets",
            {"query": "vpn", "status": ["pending", "open", "later"]},
            "ERROR INVALID_ARGUMENT: search_tickets failed: status must be null or a list whose "
            "items are each one of: open, in_progress, resolved, closed. Next step: Fix status, "
            "then call search_tickets again.",
        ),
        (
            "search_tickets",
            {"query": "vpn", "cursor": 5},
            "ERROR INVALID_ARGUMENT: search_tickets failed: cursor must be null or the "
            "next_cursor of an earlier search_tickets result for the same query and status. "
            "Next step: Fix cursor, then call search_tickets again.",
        ),
        (
            "search_tickets",
            {"page_size": 50, "query": "vpn", "limit": 5, "status": "open"},
            "ERROR INVALID_ARGUMENT: search_tickets failed: status must be null or a list whose "
            "items are each one of: open, in_progress, resolved, closed; page_size must be a "
            "whole number from 1 to 20; limit is not an argument of search_tickets. Next step: "
            "Fix status and page_size, remove limit, then call search_tickets again.",
        ),
        (
            "create_ticket",
            {**CREATE_ARGS, "category": "printers", "title": ""},
            "ERROR INVALID_ARGUMENT: create_ticket failed and the change was not applied: title "
            "must be text of 1 to 200 characters; category must be one of: network, hardware, "
            "software, access, other. Next step: Fix title and category, then call "
            "create_ticket again.",
        ),
        (
            "add_comment",
            {**COMMENT_ARGS, "extra": 1},
            "ERROR INVALID_ARGUMENT: add_comment failed and the change was not applied: extra "
            "is not an argument of add_comment. Next step: Remove extra, then call add_comment "
            "again.",
        ),
        # The agent injects these, so the model cannot fix them by calling again.
        (
            "create_ticket",
            {k: v for k, v in CREATE_ARGS.items() if k != "idempotency_key"},
            "ERROR INVALID_ARGUMENT: create_ticket failed and the change was not applied: "
            "idempotency_key is required and must be 1 to 255 printable ASCII characters "
            "without spaces. Next step: " + NOT_APPLIED_ADMIN,
        ),
        (
            "create_ticket",
            {**CREATE_ARGS, "reporter_id": "bob:admin"},
            "ERROR INVALID_ARGUMENT: create_ticket failed and the change was not applied: "
            "reporter_id must be 1 to 64 letters, digits, '_' or '-'. Next step: "
            + NOT_APPLIED_ADMIN,
        ),
    ],
)
async def test_invalid_arguments_give_exact_invalid_argument_text(
    tool: str,
    arguments: dict[str, Any],
    expected: str,
    mcp_session: SessionFactory,
    mock_api: respx.Router,
) -> None:
    async with mcp_session() as session:
        result = await session.call_tool(tool, arguments)
    text = text_of(result)
    assert text == expected
    parsed = check_contract_rules(ToolName(tool), text)
    assert parsed.code is ErrorCode.INVALID_ARGUMENT
    assert mock_api.calls.call_count == 0  # the mock is never called with bad arguments


async def test_a_long_unknown_argument_name_is_cut(mcp_session: SessionFactory) -> None:
    long_name = "x" * 500
    async with mcp_session() as session:
        result = await session.call_tool("get_ticket", {"ticket_id": "INC-0003", long_name: 1})
    text = text_of(result)
    assert f"{long_name[:64]}... is not an argument of get_ticket" in text
    assert long_name[:65] not in text
    check_contract_rules(ToolName.GET_TICKET, text)


async def test_many_validation_issues_are_summarised(mcp_session: SessionFactory) -> None:
    async with mcp_session() as session:
        result = await session.call_tool("create_ticket", {})
    text = text_of(result)
    assert "; and 1 more. Next step:" in text  # 6 required fields, 5 shown
    check_contract_rules(ToolName.CREATE_TICKET, text)


@pytest.mark.parametrize("tool", list(ToolName))
def test_every_argument_has_a_specific_rule(tool: ToolName) -> None:
    rules = argument_rules(INPUT_MODELS[tool])
    assert set(rules) == set(INPUT_MODELS[tool].model_fields)
    for name, rule in rules.items():
        assert rule not in {"text", "a valid value", "null or text"}, name
        assert "pattern ^" not in rule, name  # every shared pattern has a readable rule
    assert set(INJECTED_ARGS.get(tool, ())) <= set(rules)
