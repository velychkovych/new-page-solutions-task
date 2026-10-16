"""Builds the FastMCP server: four tools over streamable HTTP at /mcp, plus GET /health,
and the ASGI app that serves it with telemetry (CONTRACTS §2.1, §6)."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from mcp.types import ContentBlock
from opentelemetry import trace
from opentelemetry.trace import Tracer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from helpdesk.mcp_server.errors import unknown_tool
from helpdesk.mcp_server.mock_client import MockApiClient
from helpdesk.mcp_server.redaction import SecretRedactor
from helpdesk.mcp_server.request_trace import CaptureTraceContext
from helpdesk.mcp_server.tools import build_tools
from helpdesk.shared.api import HealthResponse
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import instrument_app, telemetry_lifespan

SERVER_NAME = "helpdesk-tickets"
TELEMETRY_SERVICE = "mcp"
MCP_PATH = "/mcp"
HEALTH_PATH = "/health"

logger = logging.getLogger(__name__)


class UnknownToolError(Exception):
    """Its message is the whole tool result text: the low-level server returns `str(exc)`
    of any exception from the call handler as an error result, with no prefix."""


class HelpdeskMCP(FastMCP[Any]):
    """FastMCP with a contract error text (CONTRACTS §2.4) for a tool it does not have."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        # mcp 1.30.0: ToolManager.call_tool raises ToolError("Unknown tool: <name>"), and the
        # low-level server returns that text as-is, which parse_tool_error cannot read.
        if self._tool_manager.get_tool(name) is None:
            tools = [tool.name for tool in self._tool_manager.list_tools()]
            logger.warning("call to an unknown tool rejected")
            raise UnknownToolError(unknown_tool(name, tools))
        return await super().call_tool(name, arguments)


async def health(_request: Request) -> Response:
    return JSONResponse(HealthResponse().model_dump())


def build_server(
    settings: Settings,
    client: MockApiClient,
    *,
    redactor: SecretRedactor,
    tracer: Tracer | None = None,
) -> FastMCP[Any]:
    tools: list[Tool] = list(
        build_tools(
            client,
            tracer=tracer or trace.get_tracer("helpdesk.mcp_server"),
            redactor=redactor,
            result_max_bytes=settings.trace_result_max_bytes,
        )
    )
    server: FastMCP[Any] = HelpdeskMCP(
        SERVER_NAME,
        tools=tools,
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=MCP_PATH,
        # SSE responses, so retry progress sent during a call reaches the client.
        json_response=False,
        # No transport_security: FastMCP then enables its DNS-rebinding guard only when
        # bound to a loopback host. In compose we bind 0.0.0.0, so `Host: mcp:8001` passes;
        # locally we bind 127.0.0.1 and the guard stays on.
    )
    # custom_route is untyped in mcp 1.30.0, so it is called rather than used as a
    # decorator (an untyped decorator would make `health` untyped for mypy).
    server.custom_route(HEALTH_PATH, methods=["GET"])(health)
    return server


def build_http_app(server: FastMCP[Any], settings: Settings) -> Starlette:
    """The streamable-HTTP app, traced, and setting telemetry up whenever it starts.

    FastMCP builds a plain Starlette app, so there is no `FastAPI(telemetry=...)` to pass.
    Telemetry is chained into the app's own lifespan rather than wrapped around `serve()`,
    so any server that runs the ASGI lifespan (uvicorn, or a test) gets it.
    """
    app = server.streamable_http_app()
    app.add_middleware(CaptureTraceContext)
    # Starlette 1.8.0 keeps the lifespan on the router and enters it on each ASGI
    # lifespan startup; FastMCP's own lifespan runs the session manager.
    session_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[Any]:
        # Outermost, so spans are recorded for the app's whole life and flushed last.
        async with telemetry_lifespan(TELEMETRY_SERVICE, settings), session_lifespan(app) as state:
            yield state

    app.router.lifespan_context = lifespan
    instrument_app(app)
    return app
