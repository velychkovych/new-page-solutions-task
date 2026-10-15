"""Typed tool failures and their mapping to the error contract (CONTRACTS.md §2.4).

Error text is built only from our own wording, HTTP status codes and the mock's short
`error` codes. It never includes a raw upstream body, a stack trace or a header value.
"""

from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.tickets import ApiError
from helpdesk.shared.tools import WRITE_TOOLS

# A write that timed out may still have committed, so the model must not claim either
# outcome, and must not make up a ticket id.
WRITE_NOT_CONFIRMED_NEXT_STEP = (
    "Tell the user the change could not be confirmed and to try again later; "
    "do not make up a ticket id"
)
_MAX_VALIDATION_ISSUES = 5


class ToolCallError(Exception):
    """A failure that the tool reports to the model as `ERROR <CODE>: ...`."""

    def __init__(self, code: ErrorCode, what: str, next_step: str | None = None) -> None:
        super().__init__(what)
        self.code = code
        self.what = what
        self.next_step = next_step

    @property
    def text(self) -> str:
        return format_tool_error(self.code, self.what, self.next_step)


def invalid_arguments(tool: str, exc: ValidationError) -> ToolCallError:
    """Summarise pydantic errors by location and message. The rejected input value is left
    out: it can be long, and the model already has it."""
    issues: Sequence[str] = [
        f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    ]
    shown = "; ".join(issues[:_MAX_VALIDATION_ISSUES])
    if len(issues) > _MAX_VALIDATION_ISSUES:
        shown += f"; and {len(issues) - _MAX_VALIDATION_ISSUES} more"
    return ToolCallError(ErrorCode.INVALID_ARGUMENT, f"Invalid arguments for {tool}: {shown}")


def _api_error_code(response: httpx.Response) -> str | None:
    try:
        return ApiError.model_validate_json(response.content).error
    except ValidationError:
        return None


def error_for_status(response: httpx.Response, *, ticket_id: str | None = None) -> ToolCallError:
    """Map a final non-2xx mock response. Retryable statuses never get here."""
    status = response.status_code
    match status:
        case 404:
            subject = f"Ticket {ticket_id}" if ticket_id else "The requested ticket"
            return ToolCallError(ErrorCode.NOT_FOUND, f"{subject} does not exist")
        case 401 | 403:
            return ToolCallError(
                ErrorCode.AUTH,
                f"The ticketing system rejected the MCP server's credentials (HTTP {status})",
            )
        case 409:
            return ToolCallError(
                ErrorCode.CONFLICT,
                "This idempotency key was already used for a different change",
            )
        case 400:
            return _bad_request(_api_error_code(response))
        case 422:
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "The ticketing system rejected the arguments as invalid (HTTP 422)",
            )
        case _:
            return ToolCallError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                f"The ticketing system returned an unexpected response (HTTP {status})",
            )


def _bad_request(api_code: str | None) -> ToolCallError:
    match api_code:
        case "invalid_cursor":
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "The cursor is not valid for this query and status filter",
                "Search again without a cursor, or with the query and status the cursor came from",
            )
        case "missing_idempotency_key" | "invalid_idempotency_key":
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "The ticketing system rejected the idempotency key",
                "Do not retry; tell the user the change was not applied",
            )
        case _:
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "The ticketing system rejected the request (HTTP 400)",
            )


def malformed_response(status: int, next_step: str | None = None) -> ToolCallError:
    return ToolCallError(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        f"The ticketing system returned a malformed response (HTTP {status})",
        next_step,
    )


def unexpected_failure(tool: str) -> ToolCallError:
    """A failure no other mapping covers. The exception text is left out: it may come from
    upstream bytes. A write may still have committed, so it is reported as unconfirmed."""
    return ToolCallError(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        "The tool failed with an unexpected error",
        WRITE_NOT_CONFIRMED_NEXT_STEP if tool in WRITE_TOOLS else None,
    )
