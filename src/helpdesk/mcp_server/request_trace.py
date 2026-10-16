"""Put each `mcp.tool` span in the trace of the HTTP request that carried the call.

Why the current context is not enough: in mcp 1.30.0 a session's tool calls run in the
session's server task, which `StreamableHTTPSessionManager` starts from the session's
first (initialize) request. That task copies the initialize request's contextvars, so
without an explicit parent every tool span of a session would hang under that long
finished request, in the wrong trace. The request that carried the call reaches the tool
as `RequestContext.request` (a Starlette `Request`), so its scope can hand the context on.
"""

from opentelemetry import context as otel_context
from opentelemetry import propagate
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send

# An ASGI scope extension key; namespaced so it cannot clash with a server's own keys.
SCOPE_KEY = "helpdesk.otel_context"


class CaptureTraceContext:
    """ASGI middleware that stores the request's OTel context (with its server span) in
    the scope. It must sit inside the OTel ASGI middleware, which `instrument_app` puts
    outside every user middleware."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope[SCOPE_KEY] = otel_context.get_current()
        await self.app(scope, receive, send)


def tool_span_parent(request: object | None) -> otel_context.Context | None:
    """The context to start a tool span in; None means the current one.

    `request` is `RequestContext.request`: None over the in-memory or stdio transports,
    where the current context is right.
    """
    if not isinstance(request, HTTPConnection):
        return None
    captured = request.scope.get(SCOPE_KEY)
    if isinstance(captured, otel_context.Context):
        return captured
    # An app served without `CaptureTraceContext`: still join the caller's trace, and
    # never the initialize request's (an empty carrier gives a new trace).
    return propagate.extract(request.headers)
