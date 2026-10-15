"""Bearer and admin-token checks (CONTRACTS §1.1).

This is ASGI middleware rather than a FastAPI dependency because it must run before the
chaos middleware: a dependency would run inside the route, after chaos had already drawn.
Tokens are compared in constant time and never logged or echoed.
"""

import hmac
from http import HTTPStatus

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from helpdesk.mock.errors import ApiErrorCode, error_response
from helpdesk.shared.tickets import ADMIN_TOKEN_HEADER


def bearer_token(authorization: str | None) -> str | None:
    """The credentials of a `Bearer` header. The scheme is case-insensitive (RFC 9110 §11.1)."""
    if authorization is None:
        return None
    scheme, _, credentials = authorization.partition(" ")
    return credentials if scheme.lower() == "bearer" and credentials else None


def _same_secret(provided: str | None, expected: bytes) -> bool:
    if provided is None:
        return False
    # Starlette decodes header bytes as latin-1, so encoding back with latin-1 recovers the
    # bytes the client sent. Bytes also avoid compare_digest's ASCII-only rule for str.
    return hmac.compare_digest(provided.encode("latin-1"), expected)


def is_public(path: str) -> bool:
    return path == "/health"


def is_admin(path: str) -> bool:
    return path == "/admin" or path.startswith("/admin/")


class AuthMiddleware:
    def __init__(self, app: ASGIApp, *, api_token: str, admin_token: str) -> None:
        self.app = app
        self._api_token = api_token.encode("utf-8")
        self._admin_token = admin_token.encode("utf-8")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or is_public(scope["path"]):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not _same_secret(bearer_token(headers.get("authorization")), self._api_token):
            await error_response(HTTPStatus.UNAUTHORIZED, ApiErrorCode.UNAUTHORIZED)(
                scope, receive, send
            )
            return
        if is_admin(scope["path"]) and not _same_secret(
            headers.get(ADMIN_TOKEN_HEADER), self._admin_token
        ):
            await error_response(HTTPStatus.FORBIDDEN, ApiErrorCode.FORBIDDEN)(scope, receive, send)
            return
        await self.app(scope, receive, send)
