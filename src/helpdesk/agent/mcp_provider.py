"""The real `ToolProvider`: the MCP server over streamable HTTP, via langchain-mcp-adapters.

Why a new MCP session for every call, instead of one long-lived session:
- The mcp client transport sends each request from a task started when the session opens,
  and that task keeps the context it was started in. httpx instrumentation reads the
  current span there to write `traceparent`, so a session opened at startup would put every
  tool call in the startup trace. Opened inside the calling node, the session carries the
  request's trace to the MCP server (CONTRACTS §6).
- No session state survives between calls, so there is nothing to reconnect after an MCP
  server restart.

Why the adapter's session rather than its LangChain tools: a tool call must keep the exact
MCP result, `isError` and text. The converted `StructuredTool` turns an error into an
exception or a `ToolMessage`, and the graph builds its own messages anyway.

No retries here: the MCP server is the only tool retry layer (CONTRACTS §2.5).
"""

import copy
import logging
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import (
    McpHttpClientFactory,
    StreamableHttpConnection,
    create_session,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.session import ProgressFnT
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ClientRequest,
    TextContent,
)
from pydantic import JsonValue

from helpdesk.agent.tool_provider import (
    ProgressCallback,
    ToolDefinition,
    ToolResult,
    ToolTransportError,
)
from helpdesk.shared.settings import Settings
from helpdesk.shared.tools import INJECTED_ARGS

logger = logging.getLogger(__name__)

_INJECTED: dict[str, tuple[str, ...]] = {str(tool): args for tool, args in INJECTED_ARGS.items()}


def call_timeout_s(settings: Settings) -> float:
    """How long the agent waits for one MCP request.

    The MCP server ends every tool call by its own deadline (`tool_deadline_s`), with a
    contract error when it runs out. Waiting one more attempt's timeout on top leaves it
    room to map that failure and answer, so the agent never cuts a call off first and
    turns a known outcome into a transport error.
    """
    return settings.tool_deadline_s + settings.tool_attempt_timeout_s


def model_visible_schema(tool_name: str, schema: Mapping[str, Any]) -> dict[str, JsonValue]:
    """`schema` without the arguments the agent injects (CONTRACTS §2.3).

    They go from `properties` and from `required`: a required argument the model cannot see
    would make a strict client reject every call.
    """
    visible: dict[str, Any] = copy.deepcopy(dict(schema))
    hidden = _INJECTED.get(tool_name, ())
    if not hidden:
        return visible
    properties = visible.get("properties")
    if isinstance(properties, dict):
        visible["properties"] = {k: v for k, v in properties.items() if k not in hidden}
    required = visible.get("required")
    if isinstance(required, list):
        visible["required"] = [name for name in required if name not in hidden]
    return visible


def _definition(tool: BaseTool) -> ToolDefinition:
    schema = tool.args_schema
    if not isinstance(schema, dict):
        # The adapter passes the MCP inputSchema through as a dict; anything else means
        # its conversion changed and the schema can no longer be trusted to be complete.
        raise ToolTransportError(f"{tool.name}: the MCP tool schema is not a JSON schema")
    return ToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=model_visible_schema(tool.name, schema),
    )


def _text(result: CallToolResult) -> str:
    # The contract result is one text block; other content kinds carry nothing we use.
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _progress_forwarder(on_progress: ProgressCallback) -> ProgressFnT:
    async def forward(progress: float, total: float | None, message: str | None) -> None:
        # Only the message is shown to the user; a bare counter says nothing to them.
        if message:
            await on_progress(message)

    # mcp catches and logs an exception from a progress callback, so a broken listener
    # cannot fail (or repeat) the tool call itself.
    return forward


def _failure(error: BaseException) -> str:
    """Exception type names only: messages can quote arguments, URLs or response bodies."""
    if isinstance(error, BaseExceptionGroup):
        inner = ", ".join(_failure(e) for e in error.exceptions)
        return f"{type(error).__name__}({inner})"
    return type(error).__name__


class McpToolProvider:
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        httpx_client_factory: McpHttpClientFactory | None = None,
    ) -> None:
        """`httpx_client_factory` is for tests (an in-process ASGI transport); without it
        the adapter builds a plain httpx client, which the telemetry instruments."""
        self._timeout = timedelta(seconds=timeout_s)
        connection = StreamableHttpConnection(
            transport="streamable_http",
            url=url,
            # Connect, write and pool, and also each read of the SSE response stream.
            timeout=self._timeout,
            sse_read_timeout=self._timeout,
            # Every request of the session, initialize and tools/list included. httpx's
            # read timeout does not end a request whose SSE response has started: the mcp
            # transport swallows that error and the request would wait forever.
            session_kwargs={"read_timeout_seconds": self._timeout},
        )
        if httpx_client_factory is not None:
            connection["httpx_client_factory"] = httpx_client_factory
        self._connection = connection

    @classmethod
    def from_settings(
        cls, settings: Settings, *, httpx_client_factory: McpHttpClientFactory | None = None
    ) -> McpToolProvider:
        return cls(
            settings.mcp_url,
            timeout_s=call_timeout_s(settings),
            httpx_client_factory=httpx_client_factory,
        )

    async def list_tools(self) -> Sequence[ToolDefinition]:
        try:
            # The adapter follows tools/list pagination to the end.
            tools = await load_mcp_tools(None, connection=self._connection)
        except Exception as error:
            logger.warning("listing MCP tools failed: %s", _failure(error))
            raise ToolTransportError(f"listing MCP tools failed: {_failure(error)}") from error
        return [_definition(tool) for tool in tools]

    async def call(
        self,
        name: str,
        args: Mapping[str, JsonValue],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        result: CallToolResult | None = None
        try:
            async with create_session(self._connection) as session:
                await session.initialize()
                # Not `session.call_tool`: on a fresh session it follows every result
                # with a tools/list (to validate structured output, which no tool here
                # has). A failure of that extra request would report a committed write as
                # unconfirmed, so tools/call is the last request sent.
                result = await session.send_request(
                    ClientRequest(
                        CallToolRequest(
                            params=CallToolRequestParams(name=name, arguments=dict(args))
                        )
                    ),
                    CallToolResult,
                    request_read_timeout_seconds=self._timeout,
                    progress_callback=None
                    if on_progress is None
                    else _progress_forwarder(on_progress),
                )
        except Exception as error:
            logger.warning("MCP call to %s failed: %s", name, _failure(error))
            raise ToolTransportError(f"{name}: {_failure(error)}") from error
        if result is None:
            # The adapter notes the mcp transport can swallow an error on exit
            # (langchain_mcp_adapters/tools.py, execute_tool), leaving no result at all.
            raise ToolTransportError(f"{name}: the MCP session closed without a result")
        return ToolResult(text=_text(result), is_error=result.isError)
