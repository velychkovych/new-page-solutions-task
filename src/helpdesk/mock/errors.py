"""Typed mock errors and the single place they become `ApiError` bodies (CONTRACTS §1.7)."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from helpdesk.shared.tickets import ApiError


class ApiErrorCode(StrEnum):
    MISSING_IDEMPOTENCY_KEY = "missing_idempotency_key"
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    INVALID_CURSOR = "invalid_cursor"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_ERROR = "upstream_error"
    # Not in the CONTRACTS §1.7 table, which has no row for 405; still an ApiError body.
    METHOD_NOT_ALLOWED = "method_not_allowed"


class MockApiError(Exception):
    """Raised inside the mock; turned into an `ApiError` response by `install_error_handlers`."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: ApiErrorCode = ApiErrorCode.UPSTREAM_ERROR

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.code.value)
        self.detail = detail


class MissingIdempotencyKeyError(MockApiError):
    status_code = HTTPStatus.BAD_REQUEST
    code = ApiErrorCode.MISSING_IDEMPOTENCY_KEY


class InvalidIdempotencyKeyError(MockApiError):
    status_code = HTTPStatus.BAD_REQUEST
    code = ApiErrorCode.INVALID_IDEMPOTENCY_KEY


class InvalidCursorError(MockApiError):
    status_code = HTTPStatus.BAD_REQUEST
    code = ApiErrorCode.INVALID_CURSOR


class TicketNotFoundError(MockApiError):
    status_code = HTTPStatus.NOT_FOUND
    code = ApiErrorCode.NOT_FOUND


class IdempotencyConflictError(MockApiError):
    status_code = HTTPStatus.CONFLICT
    code = ApiErrorCode.IDEMPOTENCY_CONFLICT


class TicketIdsExhaustedError(MockApiError):
    """INC-9999 is the last id the `^INC-[0-9]{4}$` pattern allows."""


def error_response(
    status_code: int,
    code: ApiErrorCode,
    detail: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = ApiError(error=code.value, detail=detail)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def summarise_validation_errors(errors: Sequence[Any]) -> str:
    """Location and message only: echoing `input` could reflect a whole request body back."""
    parts: list[str] = []
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        loc = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{loc}: {error.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid request"


_STATUS_CODES: Mapping[int, ApiErrorCode] = {
    HTTPStatus.UNAUTHORIZED: ApiErrorCode.UNAUTHORIZED,
    HTTPStatus.FORBIDDEN: ApiErrorCode.FORBIDDEN,
    HTTPStatus.NOT_FOUND: ApiErrorCode.NOT_FOUND,
    HTTPStatus.METHOD_NOT_ALLOWED: ApiErrorCode.METHOD_NOT_ALLOWED,
    HTTPStatus.CONFLICT: ApiErrorCode.IDEMPOTENCY_CONFLICT,
    HTTPStatus.UNPROCESSABLE_CONTENT: ApiErrorCode.INVALID_REQUEST,
    HTTPStatus.TOO_MANY_REQUESTS: ApiErrorCode.RATE_LIMITED,
}


async def _to_api_error(_: Request, exc: Exception) -> JSONResponse:
    """One handler for every exception class, so no response body escapes the ApiError shape."""
    match exc:
        case MockApiError():
            return error_response(exc.status_code, exc.code, exc.detail)
        case RequestValidationError():
            return error_response(
                HTTPStatus.UNPROCESSABLE_CONTENT,
                ApiErrorCode.INVALID_REQUEST,
                summarise_validation_errors(exc.errors()),
            )
        case StarletteHTTPException(status_code=HTTPStatus.BAD_REQUEST):
            # FastAPI raises this when it cannot decode the body (e.g. invalid UTF-8). §1.7
            # reserves 400 for keys and cursors, so treat it like unparseable JSON: 422.
            return error_response(
                HTTPStatus.UNPROCESSABLE_CONTENT,
                ApiErrorCode.INVALID_REQUEST,
                "the request body could not be decoded",
            )
        case StarletteHTTPException():
            # Router-level errors (unknown path -> 404, wrong method -> 405) would otherwise
            # use Starlette's `{"detail": ...}` body.
            code = _STATUS_CODES.get(exc.status_code, ApiErrorCode.UPSTREAM_ERROR)
            return error_response(exc.status_code, code, headers=exc.headers)
        case _:
            # No exception text in the body: it could contain internals.
            return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, ApiErrorCode.UPSTREAM_ERROR)


def install_error_handlers(app: FastAPI) -> None:
    for exc_class in (MockApiError, RequestValidationError, StarletteHTTPException, Exception):
        app.add_exception_handler(exc_class, _to_api_error)
