"""Ticket routes (CONTRACTS §1.3-§1.5)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import JSONResponse

from helpdesk.mock.chaos import ChaosEngine
from helpdesk.mock.cursor import decode_cursor, encode_cursor
from helpdesk.mock.db import transaction
from helpdesk.mock.errors import ApiErrorCode, error_response
from helpdesk.mock.idempotency import (
    IdempotentWriter,
    WriteOperation,
    WriteOutcome,
    request_fingerprint,
    validate_key,
)
from helpdesk.mock.repository import add_comment, create_ticket, get_ticket, search_tickets
from helpdesk.mock.search import SearchFilter
from helpdesk.shared.tickets import (
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAY_HEADER,
    SEARCH_LIMIT_DEFAULT,
    SEARCH_LIMIT_MAX,
    SEARCH_LIMIT_MIN,
    AddCommentRequest,
    CommentCreated,
    CreateTicketRequest,
    SearchResult,
    Ticket,
    TicketStatus,
)

type Clock = Callable[[], datetime]

IdempotencyKeyHeader = Annotated[str | None, Header(alias=IDEMPOTENCY_KEY_HEADER)]


@dataclass(frozen=True, slots=True)
class TicketServices:
    db_path: Path
    writer: IdempotentWriter
    chaos: ChaosEngine
    clock: Clock


def build_ticket_router(services: TicketServices) -> APIRouter:
    router = APIRouter()

    async def run_write(
        request: Request,
        body: CreateTicketRequest | AddCommentRequest,
        raw_key: str | None,
        operation: WriteOperation,
    ) -> JSONResponse:
        key = validate_key(raw_key)
        fingerprint = request_fingerprint(request.method, request.url.path, body)
        outcome = await services.writer.run(key, fingerprint, HTTPStatus.CREATED, operation)
        return _write_response(outcome, services.chaos)

    @router.get("/tickets")
    async def search(
        q: Annotated[str | None, Query(max_length=200)] = None,
        status: Annotated[list[TicketStatus] | None, Query()] = None,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=SEARCH_LIMIT_MIN, le=SEARCH_LIMIT_MAX)] = (
            SEARCH_LIMIT_DEFAULT
        ),
    ) -> SearchResult:
        search_filter = SearchFilter.build(q, status or ())
        offset = 0 if cursor is None else decode_cursor(cursor, search_filter)
        async with transaction(services.db_path) as conn:
            matches = await search_tickets(conn, search_filter)
        end = offset + limit
        return SearchResult(
            items=matches[offset:end],
            next_cursor=encode_cursor(end, search_filter) if end < len(matches) else None,
        )

    @router.get("/tickets/{ticket_id}")
    async def read(ticket_id: str) -> Ticket:
        # No pattern check on the path: a malformed id is simply not found (CONTRACTS §1.3).
        async with transaction(services.db_path) as conn:
            return await get_ticket(conn, ticket_id)

    @router.post("/tickets", status_code=HTTPStatus.CREATED)
    async def create(
        request: Request, body: CreateTicketRequest, idempotency_key: IdempotencyKeyHeader = None
    ) -> JSONResponse:
        return await run_write(
            request,
            body,
            idempotency_key,
            lambda conn: create_ticket(conn, body, services.clock()),
        )

    @router.post("/tickets/{ticket_id}/comments", status_code=HTTPStatus.CREATED)
    async def comment(
        request: Request,
        ticket_id: str,
        body: AddCommentRequest,
        idempotency_key: IdempotencyKeyHeader = None,
    ) -> JSONResponse:
        async def operation(conn: aiosqlite.Connection) -> CommentCreated:
            created = await add_comment(conn, ticket_id, body, services.clock())
            return CommentCreated(ticket_id=ticket_id, comment=created)

        return await run_write(request, body, idempotency_key, operation)

    return router


def _write_response(outcome: WriteOutcome, chaos: ChaosEngine) -> JSONResponse:
    if not outcome.executed:
        return JSONResponse(
            status_code=outcome.status_code,
            content=outcome.body,
            headers={IDEMPOTENT_REPLAY_HEADER: "true"},
        )
    if chaos.fail_after_commit():
        # The write and its key are committed; only the response is lost. A retry with the
        # same key gets the stored 201 as a replay.
        return error_response(HTTPStatus.SERVICE_UNAVAILABLE, ApiErrorCode.UPSTREAM_ERROR)
    return JSONResponse(status_code=outcome.status_code, content=outcome.body)
