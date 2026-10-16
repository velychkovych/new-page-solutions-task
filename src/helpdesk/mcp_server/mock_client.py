"""HTTP client for the mock ticketing API (CONTRACTS.md §1, §2.2, §2.6).

Every request goes through `call_with_retry`, the only retry layer. Failures leave this
module as `ToolCallError` (see `.errors`); success values are the shared wire models.
"""

import asyncio
import random
import time
from typing import Literal

import httpx
from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from helpdesk.mcp_server.errors import (
    deadline_exceeded,
    error_for_status,
    malformed_response,
    upstream_unavailable,
)
from helpdesk.mcp_server.retry import (
    Clock,
    DeadlineExceededError,
    OnRetry,
    RetryPolicy,
    RetryResult,
    Sleep,
    Uniform,
    UpstreamUnavailableError,
    call_with_retry,
)
from helpdesk.shared import trace_attributes as attr
from helpdesk.shared.settings import Settings
from helpdesk.shared.tickets import (
    IDEMPOTENCY_KEY_HEADER,
    AddCommentRequest,
    CommentCreated,
    CreateTicketRequest,
    SearchResult,
    Ticket,
)
from helpdesk.shared.tools import (
    AddCommentInput,
    CreateTicketInput,
    GetTicketArgs,
    SearchTicketsArgs,
)

# Matches httpx's accepted item type exactly (list is invariant).
type QueryItems = list[tuple[str, str | int | float | bool | None]]

# Jitter only spreads retries out; it has no security role.
_jitter = random.Random()  # noqa: S311


def create_mock_http_client(
    settings: Settings, token: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """The one place the bearer token is attached. httpx masks Authorization in reprs and
    logs no headers, so the token stays out of logs."""
    return httpx.AsyncClient(
        base_url=settings.mock_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=settings.tool_attempt_timeout_s,
        transport=transport,
    )


class MockApiClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        policy: RetryPolicy,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        uniform: Uniform = _jitter.uniform,
    ) -> None:
        self._http = http
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._uniform = uniform

    async def search_tickets(self, args: SearchTicketsArgs, on_retry: OnRetry) -> SearchResult:
        params: QueryItems = [("q", args.query), ("limit", args.page_size)]
        params.extend(("status", status.value) for status in args.status or ())
        if args.cursor is not None:
            params.append(("cursor", args.cursor))
        result = await self._send("GET", "/tickets", on_retry=on_retry, params=params)
        return _parse(result, SearchResult)

    async def get_ticket(self, args: GetTicketArgs, on_retry: OnRetry) -> Ticket:
        result = await self._send("GET", f"/tickets/{args.ticket_id}", on_retry=on_retry)
        return _parse(result, Ticket, ticket_id=args.ticket_id)

    async def create_ticket(self, args: CreateTicketInput, on_retry: OnRetry) -> Ticket:
        body = CreateTicketRequest(
            title=args.title,
            description=args.description,
            category=args.category,
            priority=args.priority,
            reporter_id=args.reporter_id,
        )
        result = await self._send(
            "POST",
            "/tickets",
            on_retry=on_retry,
            body=body,
            idempotency_key=args.idempotency_key,
        )
        return _parse(result, Ticket)

    async def add_comment(self, args: AddCommentInput, on_retry: OnRetry) -> CommentCreated:
        body = AddCommentRequest(author_id=args.author_id, body=args.body)
        result = await self._send(
            "POST",
            f"/tickets/{args.ticket_id}/comments",
            on_retry=on_retry,
            body=body,
            idempotency_key=args.idempotency_key,
        )
        return _parse(result, CommentCreated, ticket_id=args.ticket_id)

    async def _send(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        on_retry: OnRetry,
        params: QueryItems | None = None,
        body: BaseModel | None = None,
        idempotency_key: str | None = None,
    ) -> RetryResult:
        # The key the agent sent is forwarded unchanged on every attempt, so a retried
        # write is a replay at the mock, never a second write.
        headers = {} if idempotency_key is None else {IDEMPOTENCY_KEY_HEADER: idempotency_key}
        json_body = None if body is None else body.model_dump(mode="json")

        async def attempt(timeout_s: float) -> httpx.Response:
            return await self._http.request(
                method, path, params=params, json=json_body, headers=headers, timeout=timeout_s
            )

        span = trace.get_current_span()
        try:
            result = await call_with_retry(
                attempt,
                policy=self._policy,
                clock=self._clock,
                sleep=self._sleep,
                uniform=self._uniform,
                on_retry=on_retry,
            )
        except DeadlineExceededError as exc:
            span.set_attribute(attr.TOOL_ATTEMPTS, exc.attempts)
            raise deadline_exceeded(exc, self._policy.deadline_s) from exc
        except UpstreamUnavailableError as exc:
            span.set_attribute(attr.TOOL_ATTEMPTS, exc.attempts)
            raise upstream_unavailable(exc) from exc
        span.set_attribute(attr.TOOL_ATTEMPTS, result.attempts)
        return result


def _parse[M: BaseModel](result: RetryResult, model: type[M], *, ticket_id: str | None = None) -> M:
    response = result.response
    if not response.is_success:
        raise error_for_status(response, attempts=result.attempts, ticket_id=ticket_id)
    try:
        return model.model_validate_json(response.content)
    except ValidationError as exc:
        raise malformed_response(response.status_code, attempts=result.attempts) from exc
