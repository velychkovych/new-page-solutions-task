"""The agent's only door to the ticketing tools (CONTRACTS §2, §5.3).

The graph is built against this protocol; `mcp_provider.McpToolProvider` implements it
with langchain-mcp-adapters, and the unit tests with a fake. Keeping it this narrow means
the graph never sees MCP types, and the one tool retry layer stays in the MCP server.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue


class ToolDefinition(BaseModel):
    """A tool as the model sees it. Injected arguments are already stripped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    # JSON schema of the model-visible arguments (an MCP `inputSchema` minus INJECTED_ARGS).
    input_schema: dict[str, JsonValue]


class ToolResult(BaseModel):
    """The text of one MCP tool result and whether it was an MCP tool error (`isError`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The result's text content; "" when it had none (only non-text content, or nothing).
    # The graph turns such a result into an INTERNAL error rather than pass it on.
    text: str
    is_error: bool


class ToolTransportError(Exception):
    """The tool could not be reached at all (connection, protocol, session failure).

    This is distinct from an MCP tool error, which arrives as a `ToolResult` with
    `is_error=True`. The agent maps it to UPSTREAM_UNAVAILABLE and never retries:
    retries belong to the MCP server alone.
    """


# Receives the MCP server's progress messages for one call, e.g. "Retrying (attempt 2 of 3)".
type ProgressCallback = Callable[[str], Awaitable[None]]


class ToolProvider(Protocol):
    async def list_tools(self) -> Sequence[ToolDefinition]:
        """Model-visible definitions of every ticketing tool."""
        ...

    async def call(
        self,
        name: str,
        args: Mapping[str, JsonValue],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        """Call a tool with its full MCP arguments (injected ones included for writes).

        `on_progress`, when given, receives the server's progress messages for this call
        only. Raises `ToolTransportError` when the call could not be delivered or answered.
        """
        ...
