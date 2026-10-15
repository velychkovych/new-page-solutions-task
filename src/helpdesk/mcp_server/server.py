"""Builds the FastMCP server: four tools over streamable HTTP at /mcp, plus GET /health."""

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import Tool
from opentelemetry import trace
from opentelemetry.trace import Tracer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from helpdesk.mcp_server.mock_client import MockApiClient
from helpdesk.mcp_server.redaction import SecretRedactor
from helpdesk.mcp_server.tools import build_tools
from helpdesk.shared.api import HealthResponse
from helpdesk.shared.settings import Settings

SERVER_NAME = "helpdesk-tickets"
MCP_PATH = "/mcp"
HEALTH_PATH = "/health"


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
    server: FastMCP[Any] = FastMCP(
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
